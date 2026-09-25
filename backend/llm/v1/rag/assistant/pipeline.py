"""[assistant] 챗봇 파이프라인 — 프롬프트 · RAG · 에이전트 · 파서  (담당: 형준, 2026-09-15)

    retrieve       RAG — 질문으로 문서 DB(pgvector)를 먼저 검색한다                     LLM 0회 (임베딩 1회)
    build_prompt   시스템 규칙 + 참고 문서 + 오늘 날짜 + 화면 구장 + 최근 대화 8개
    agent          create_agent — 필요하면 도구를 부른다                                 LLM 1~5회
                     야구 DB(읽기 전용 계정): get_games · get_standings · get_ticket_prices · get_ticket_policy
                                              get_baseball_schema · execute_baseball_select
                     그 밖: search_kbo_documents · search_nearby_places · plan_course
    parse_output   답변 문자열 + (코스를 짰으면) 지도·카드용 places

    chain = retrieve | build_prompt | agent | parse_output

스위치 없음 — 모든 질문이 이 한 줄로 간다. RAG 는 매번 읽고, 야구 DB 는 에이전트가 필요할 때 읽는다.
디스패처와의 약속: answer(question, history, hint_stadium) -> {"answer","sources","route","places","coursePayload",...}
"""
import logging
import json
import os
import time
from contextlib import suppress
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from ...progress import ProgressCancelled, ProgressStorageError, config_kwargs, current, operation
from ..domain_tools import visible_text
from . import tools
from .prompts import COURSE_HINT, HINT_LINE, NEARBY_HINT, SYSTEM

log = logging.getLogger(__name__)

try:
    from langchain.agents import create_agent        # langchain>=1.0 (venue 와 같은 방식)
    READY = True
except ImportError:                                   # pragma: no cover
    create_agent = None
    READY = False

LLM_MODEL = os.getenv("LLM_MODEL") or "gpt-5.6-luna"
RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "12"))   # 도구 호출 약 4~5번까지
HISTORY_TURNS = 8
CONTEXT_K = 6
STADIUM_KO = {"JAMSIL": "잠실야구장", "GOCHEOK": "고척스카이돔", "MUNHAK": "인천 SSG 랜더스필드", "SUWON": "수원 KT 위즈 파크",
              "DAEJEON": "대전 한화생명 볼파크", "DAEGU": "대구 삼성 라이온즈 파크", "GWANGJU": "광주-KIA 챔피언스 필드",
              "SAJIK": "사직야구장", "CHANGWON": "창원 NC 파크"}
NEARBY_LABEL = {"stay": "숙박", "walk": "산책", "indoor": "실내 놀거리", "store": "편의점"}

_chain = None
_llm = None
MAX_TOOL_CALLS = 4
MAX_ANSWER_LENGTH = 8000
PLANNER_RULE = (
    "지금은 답변 작성 단계가 아니라 조회 계획 단계다. 필요한 도구가 있으면 호출하고, "
    "더 이상 호출할 도구가 없으면 사용자 답변을 작성하지 말고 READY 한 단어만 출력한다."
)


# ── 1. RAG ────────────────────────────────────────────────────────────────────
def stadium_for(question, history, hint):
    """질문 → 직전 사용자 발화 → 화면에서 고른 구장 순."""
    from ..club.router import detect_stadium
    code = detect_stadium(question)
    if not code:
        for m in reversed(history or []):
            if m.get("role") == "user" and (code := detect_stadium(m.get("content") or "")):
                break
    code = code or hint
    return code if code in STADIUM_KO else None


def retrieve(inputs: dict, _search=None, _embed=None) -> dict:
    """질문으로 문서를 먼저 찾아 inputs["context"] 에 넣는다. 검색이 실패해도 답은 계속 만든다."""
    from ..club.router import detect_categories
    question = inputs["question"]
    stadium = stadium_for(question, inputs.get("history"), inputs.get("hint"))
    cats = [c for c in detect_categories(question) if c not in ("SCHEDULE", "STANDING")] or None   # 일정·순위는 DB 가 정본
    try:
        with operation("retrieval", "retrieval", arguments={"stadium": stadium, "categories": cats}):
            rows = tools.search_documents(question, stadium, cats, k=CONTEXT_K, _search=_search, _embed=_embed)
    except (ProgressCancelled, ProgressStorageError):
        raise
    except Exception:
        log.exception("rag retrieve failed")
        rows = []
    tools.add_sources(rows)
    return {**inputs, "stadium": stadium, "context": tools.format_documents(rows), "doc_count": len(rows)}


# ── 2. 프롬프트 ───────────────────────────────────────────────────────────────
def route_hint(question) -> str:
    """코스·주변장소 질문이면 알맞은 도구를 먼저 쓰라고 한 줄 알려 준다 (분기가 아니라 힌트)."""
    from ..dispatcher import COURSE
    from ..nearby.agent import kinds_of
    if COURSE.search(question):
        return COURSE_HINT
    kinds = kinds_of(question)
    return NEARBY_HINT.format(kind=kinds[0], label=NEARBY_LABEL[kinds[0]]) if kinds else ""


def build_prompt(inputs: dict) -> dict:
    hint = inputs.get("hint")
    hint_line = HINT_LINE.format(name=STADIUM_KO[hint], code=hint) if hint in STADIUM_KO else ""
    system = (SYSTEM.replace("{today}", inputs.get("today") or date.today().isoformat())
              .replace("{stadium_hint}", hint_line)
              .replace("{context}", inputs.get("context") or "검색 결과 없음")
              .replace("{route_hint}", inputs.get("route_hint", route_hint(inputs["question"]))))
    conv = {"user": HumanMessage, "assistant": AIMessage}
    past = [conv[m["role"]](content=m["content"]) for m in (inputs.get("history") or [])
            if m.get("role") in conv and isinstance(m.get("content"), str)][-HISTORY_TURNS:]
    return {"messages": [SystemMessage(content=system), *past, HumanMessage(content=inputs["question"])]}


# ── 4. 파서 ──────────────────────────────────────────────────────────────────
def _text(content) -> str:
    return visible_text(content)


def parse_output(result: dict) -> str:
    """에이전트 결과 → 마지막 AI 메시지 본문. 도구 호출만 남기고 끝났으면 빈 문자열."""
    for msg in reversed(result.get("messages") or []):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            return _text(msg.content).strip()
    return ""


# ── 3. 에이전트 + 조립 ────────────────────────────────────────────────────────
def llm():
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(
            model=LLM_MODEL, temperature=0, timeout=25, max_retries=0,
            reasoning_effort="medium", use_responses_api=True,
        )
    return _llm


def build_chain(model=None, tool_list=None, retriever=None):
    model = model or llm()
    agent = create_agent(model=model, tools=tool_list if tool_list is not None else tools.build_tools())
    return RunnableLambda(retriever or retrieve) | RunnableLambda(build_prompt) | agent | RunnableLambda(parse_output)


def _stream_answer(question, history=None, hint_stadium=None, *, model=None, tool_list=None, retriever=None):
    """도구 계획은 숨기고 마지막 provider 응답 청크만 즉시 전달한다."""
    model = model or llm()
    st = tools.state()
    t0 = time.perf_counter()
    prepared = (retriever or retrieve)({
        "question": question, "history": history or [], "hint": hint_stadium,
    })
    messages = build_prompt(prepared)["messages"]
    exposed_tools = list(tool_list if tool_list is not None else tools.build_tools())
    allowed = {tool.name: tool for tool in exposed_tools}
    bound = model.bind_tools(exposed_tools)
    conversation = list(messages)
    calls = 0
    seen_calls = {}

    for _ in range(MAX_TOOL_CALLS + 1):
        planner = [SystemMessage(content=f"{conversation[0].content}\n\n{PLANNER_RULE}"), *conversation[1:]]
        response = bound.invoke(planner, **config_kwargs())
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        signatures = [json.dumps(
            [call.get("name"), call.get("args")], ensure_ascii=False, sort_keys=True, default=str,
        ) for call in tool_calls]
        new_count = len({signature for signature in signatures if signature not in seen_calls})
        if not new_count:
            break
        if calls + new_count > MAX_TOOL_CALLS:
            raise ValueError("tool call limit exceeded")
        conversation.append(response)
        for call, signature in zip(tool_calls, signatures):
            name, arguments, call_id = call.get("name"), call.get("args"), call.get("id")
            if name not in allowed or not isinstance(arguments, dict) or not isinstance(call_id, str) or not call_id:
                raise ValueError("malformed tool call")
            if signature in seen_calls:
                conversation.append(ToolMessage(
                    content=seen_calls[signature], tool_call_id=call_id, name=name,
                ))
                continue
            output = allowed[name].invoke(call, **config_kwargs())
            message = output if isinstance(output, ToolMessage) else ToolMessage(
                content=output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, default=str),
                tool_call_id=call_id, name=name,
            )
            conversation.append(message)
            seen_calls[signature] = message.content
        calls += new_count
    else:
        raise ValueError("tool call limit exceeded")

    provider_stream, answer, size = None, [], 0
    try:
        provider_stream = model.stream(conversation, **config_kwargs())
        for chunk in provider_stream:
            collector = current()
            if collector is not None and collector.cancelled:
                raise ProgressCancelled("chat progress cancelled")
            text = _text(getattr(chunk, "content", chunk))
            if not text:
                continue
            size += len(text)
            if size > MAX_ANSWER_LENGTH:
                raise ValueError("LLM response too long")
            answer.append(text)
            yield text
    finally:
        close = getattr(provider_stream, "close", None)
        if close:
            with suppress(Exception):
                close()
    if not answer:
        raise ValueError("agent returned no answer")
    course = st.get("course") or {}
    out = {
        "answer": "".join(answer),
        "sources": st["sources"],
        "route": "agent:rag" + ("," + ",".join(dict.fromkeys(st["tools"])) if st["tools"] else ""),
        "timing": {"agent_ms": round((time.perf_counter() - t0) * 1000), "tool_calls": calls},
    }
    for key in ("places", "coursePayload", "stadiumCode", "travel"):
        if course.get(key):
            out[key] = course[key]
    return out


def stream_answer(question, history=None, hint_stadium=None, *, model=None, tool_list=None, retriever=None):
    with tools.request_state(hint_stadium, question, history):
        return (yield from _stream_answer(
            question, history=history, hint_stadium=hint_stadium,
            model=model, tool_list=tool_list, retriever=retriever,
        ))


def chain():
    global _chain
    if _chain is None:
        _chain = build_chain()
    return _chain


def _answer(question, history=None, hint_stadium=None, _chain_obj=None):
    if _chain_obj is None:
        stream = _stream_answer(question, history=history, hint_stadium=hint_stadium)
        while True:
            try:
                next(stream)
            except StopIteration as done:
                return done.value
    st = tools.state()
    t0 = time.perf_counter()
    text = _chain_obj.invoke(
        {"question": question, "history": history or [], "hint": hint_stadium},
        **config_kwargs(recursion_limit=RECURSION_LIMIT),
    )
    course = st.get("course") or {}
    if course.get("places"):
        text = course["answer"]                          # 지도에 그린 코스와 글이 어긋나지 않게 코스 결과를 그대로 쓴다
    if not text:
        raise ValueError("agent returned no answer")    # 디스패처가 기존 도메인으로 한 번 더 시도한다
    used = list(dict.fromkeys(st["tools"]))
    out = {
        "answer": text,
        "sources": st["sources"],
        "route": "agent:rag" + ("," + ",".join(used) if used else ""),
        "timing": {"agent_ms": round((time.perf_counter() - t0) * 1000), "tool_calls": len(st["tools"])},
    }
    for key in ("places", "coursePayload", "stadiumCode", "travel"):   # 코스를 짰으면 지도·카드용 값을 그대로 넘긴다
        if course.get(key):
            out[key] = course[key]
    return out


def answer(question, history=None, hint_stadium=None, _chain_obj=None):
    with tools.request_state(hint_stadium, question, history):
        return _answer(question, history, hint_stadium, _chain_obj)
