"""RAG 파이프라인 진입점 — 백엔드가 부르는 파일은 이것 하나.

## 성호 ChatService 와 붙는 방법

ChatService 는 `self.chain` 에 두 가지만 요구한다.

    self.chain.invoke({"question": q, "chat_history": messages}) -> str
    self.chain.stream({"question": q, "chat_history": messages}) -> str 청크들

그래서 여기서 그 규격을 그대로 만족하는 Runnable(`chat_chain()`)을 만들어 준다.
ChatService 쪽 변경은 import 1줄 + chain 고르는 1줄이 전부다.

    from .rag.pipeline import chat_chain          # 추가
    if rag := chat_chain(): return rag.invoke(...) # 항상 RAG 파이프라인 (테스트 중에만 성호 체인)

스위치(CHAT_USE_RAG)는 2026-09-15 에 없앴다. 챗봇은 항상 이 파이프라인으로 답하고,
야구 DB 는 파이프라인 안의 에이전트가 읽기 전용 계정 도구로 조회한다.
`chat_chain()` 은 테스트 러너 안에서만 None 을 돌려줘 성호 회귀 테스트는 그대로 돈다.

## 풍부한 결과가 필요할 때 (코스 추천 places 등)

체인은 문자열만 주므로, 장소 목록·근거·route 가 필요하면 함수를 직접 부른다.

    from llm.v1.rag.pipeline import answer
    result = answer("잠실 주차 얼마야?", history=[...], stadium_name="잠실야구장")
    # {"answer", "sources", "route", "places", "coursePayload"}

## 입력

    question     : str   이번 질문. 프론트가 앞에 붙이는 "[선택한 구장: 잠실야구장]" 접두어는
                         여기서 떼어 stadium_name 으로 쓴다
    history      : 이전 대화 (이번 질문 제외, 오래된 것부터). 두 형식 다 받는다
                   - [{"role": "user"|"assistant", "content": "..."}, ...]   (프론트·게스트 형식)
                   - [HumanMessage(...), AIMessage(...), ...]                 (DjangoChatMessageHistory)
    stadium_name : "잠실야구장" 같은 프론트 context.stadium 값. 없으면 None
    intent       : "route" | "baseball" | "stadium". "route" 면 코스 추천. 없으면 None
                   (dispatcher 가 아직 intent 를 안 받는 버전이면 자동으로 안 넘긴다)

## 흐름

    ① history 정리 · 구장 접두어 분리                                   LLM 0회
    ② dispatcher          야구와 무관한 질문만 바로 안내                  LLM 0회
    ③ assistant 파이프라인  프롬프트 · RAG(문서 검색) · 에이전트 · 파서       LLM 1~5회
                          에이전트 도구: 야구 DB 읽기 전용 조회, 주변 장소(카카오), 코스 짜기
                          실패하면 예전 도메인(course/club/venue/nearby)으로 한 번 더
    ④ persona.finalize()  말투 통일                                     LLM 0회

LangSmith: backend/.env 에 LANGSMITH_TRACING=true · LANGSMITH_API_KEY · LANGSMITH_PROJECT 를 넣으면
           answer() 한 번이 트리 하나로 기록된다. env 가 없으면 오버헤드 0.
"""
import contextvars
import inspect
import os
import re
import sys
from typing import Any, Iterator, Optional

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig

try:                                          # langsmith 는 langchain-core 의존성이라 보통 있다
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree
except ImportError:                           # pragma: no cover
    def traceable(*_a, **_k):
        return lambda f: f

    def get_current_run_tree():
        return None

from . import dispatcher

_ROLE = {"human": "user", "ai": "assistant", "user": "user", "assistant": "assistant"}
_STADIUM_PREFIX = re.compile(r"^\s*\[선택한 구장:\s*([^\]]+)\]\s*")   # 프론트가 붙이는 접두어
# 코스 작성 화면에서 지도에 출발지를 찍으면 붙는 접두어: [출발지: 37.51234,127.07123]
_ORIGIN_PREFIX = re.compile(r"^\s*\[출발지:\s*(-?\d{1,3}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)\s*\]\s*")
# dispatcher 에 course 가 들어오기 전/후 둘 다에서 돌게 한다
_HAS_INTENT = "intent" in inspect.signature(dispatcher.answer).parameters
_HAS_ORIGIN = "origin" in inspect.signature(dispatcher.answer).parameters

# 이번 요청의 RAG 결과를 뷰가 꺼내 쓰라고 잠깐 놔두는 자리.
# 체인은 문자열만 돌려주는데(성호 규격), 뷰는 places·coursePayload 도 내려줘야 해서 필요하다.
# ContextVar 라 요청(스레드)마다 따로 놀아서 동시 요청이 섞이지 않는다.
_LAST = contextvars.ContextVar("kbo_rag_last_detail", default=None)


def last_detail():
    """직전에 이 요청에서 돌린 RAG 결과 dict (없으면 None).

    뷰에서 이렇게 쓴다:
        answer = ...체인 실행...
        d = last_detail()
        if d: 응답에 d["places"] · d["coursePayload"] 를 실어 보낸다
    """
    return _LAST.get()


def _running_tests() -> bool:
    """지금 테스트 러너 안에서 도는 중인가.

    성호 회귀 테스트는 patch("llm.v1.chat_service.ChatService.get_chain", ...) 나
    patch("llm.v1.chat_service.ChatOpenAI", ...) 로 모델을 갈아끼운다.
    RAG 를 켜면 그 자리를 우리 체인이 차지해 패치가 안 먹고 테스트가 통째로 깨진다.
    성호 테스트 파일을 건드리지 않으려고 우리 쪽에서 막는다.

    판정 (하나라도 걸리면 테스트로 본다)
      1) manage.py test ...  → sys.argv 에 "test"
      2) pytest              → PYTEST_CURRENT_TEST 환경변수 또는 argv[0]
      3) 마지막 안전망        → Django 가 만든 test_* / :memory: DB
    """
    if "test" in sys.argv or os.getenv("PYTEST_CURRENT_TEST"):
        return True
    if sys.argv and sys.argv[0].endswith(("pytest", "py.test")):
        return True
    try:
        from django.db import connection
        name = str(connection.settings_dict.get("NAME") or "")
    except Exception:
        return False
    return name.startswith("test_") or name == ":memory:"


def use_rag() -> bool:
    """챗봇은 항상 RAG 파이프라인으로 답한다 (CHAT_USE_RAG 스위치 제거, 2026-09-15).

    성호 회귀 테스트(도구 루프를 가짜 모델로 검사)가 깨지지 않도록 테스트 러너 안에서만 끈다.
    """
    return not _running_tests()


def split_context_prefix(question: str) -> tuple[str, Optional[str], Optional[dict]]:
    """'[선택한 구장: 잠실야구장][출발지: 37.5,127.0]\\n코스 짜줘' → ('코스 짜줘', '잠실야구장', {"lat": 37.5, "lng": 127.0})

    두 접두어는 어느 순서로 와도 되고, 없는 것은 None 이다.
    """
    text, stadium, origin = question or "", None, None
    while True:
        if stadium is None and (m := _STADIUM_PREFIX.match(text)):
            stadium, text = m.group(1).strip(), text[m.end():]
        elif origin is None and (m := _ORIGIN_PREFIX.match(text)):
            origin, text = {"lat": float(m.group(1)), "lng": float(m.group(2))}, text[m.end():]
        else:
            return text.strip(), stadium, origin


def split_stadium_prefix(question: str) -> tuple[str, Optional[str]]:
    """'[선택한 구장: 잠실야구장]\\n잠실 주차 얼마야?' → ('잠실 주차 얼마야?', '잠실야구장') · 출발지 접두어도 함께 뗀다"""
    text, stadium, _ = split_context_prefix(question)
    return text, stadium


def normalize_history(history) -> list[dict]:
    """LangChain 메시지든 dict 든 [{"role","content"}] 로 맞춘다.

    회원 경로는 DjangoChatMessageHistory.messages (HumanMessage/AIMessage),
    게스트 경로(GuestChatView)도 HumanMessage/AIMessage 로 만들어 넘겨준다.
    system 등 그 외 역할은 버리고, 구장 접두어도 뗀다.
    """
    out = []
    for m in history or []:
        if isinstance(m, BaseMessage):
            role, content = _ROLE.get(m.type), m.content
        elif isinstance(m, dict):
            role, content = _ROLE.get(m.get("role")), m.get("content")
        else:
            continue
        if role and isinstance(content, str):
            out.append({"role": role, "content": split_stadium_prefix(content)[0]})
    return out


def _tag(result: dict, extra: dict):
    """LangSmith 가 켜져 있으면 route 를 run metadata 로 남긴다 (꺼져 있으면 no-op)"""
    try:
        rt = get_current_run_tree()
        if rt is not None:
            rt.add_metadata({"route": result.get("route", ""),
                             "n_sources": len(result.get("sources") or []),
                             "n_places": len(result.get("places") or []), **extra})
    except Exception:
        pass


@traceable(run_type="chain", name="kbo_rag.answer")
def answer(question: str, history=None, stadium_name: Optional[str] = None,
           intent: Optional[str] = None, origin: Optional[dict] = None) -> dict:
    """RAG 실행. 반환 {"answer", "sources", "route", "places", "coursePayload", "question"}

    question 키에는 구장 접두어를 뗀 질문이 들어간다 (대화 기록에 저장할 때 쓰라고).
    """
    q, prefixed, prefixed_origin = split_context_prefix(question)
    stadium_name = stadium_name or prefixed
    origin = origin or prefixed_origin
    kwargs = {"intent": intent} if _HAS_INTENT else {}
    if origin and _HAS_ORIGIN:
        kwargs["origin"] = origin
    result = dispatcher.answer(q, history=normalize_history(history), stadium_name=stadium_name, **kwargs)
    result.setdefault("places", [])
    result.setdefault("coursePayload", None)
    result["question"] = q
    _tag(result, {"stadium_name": stadium_name or "", "intent": intent or ""})
    return result


# ── 성호 ChatService 의 self.chain 자리에 그대로 꽂히는 Runnable ──────────────────
class RagChatChain(Runnable[dict, str]):
    """{"question", "chat_history"} → 답변 문자열. invoke 와 stream 둘 다 지원한다.

    ChatService.invoke_with_messages 는 .invoke() 를,
    ChatService.stream_with_history 는 .stream() 을 부른다 — 둘 다 여기로 온다.

    .stream() 은 마지막 provider 모델의 실제 text delta 만 흘린다.
    """

    name = "kbo_rag_chain"

    @staticmethod
    def _args(inputs: Any) -> dict:
        if isinstance(inputs, str):
            question, stadium_name, origin = split_context_prefix(inputs)
            return {"question": question, "history": [], "stadium_name": stadium_name, "intent": None, "origin": origin}
        inputs = inputs or {}
        # 스트리밍 경로도 출발지 접두어를 잃지 않게 구장·출발지를 함께 뗀다
        question, prefixed_stadium, prefixed_origin = split_context_prefix(inputs.get("question") or "")
        return {
            "question": question,
            # chat_history 는 성호 체인 키, history 는 우리 키 — 둘 다 받는다
            "history": normalize_history(
                inputs.get("chat_history") if inputs.get("chat_history") is not None
                else inputs.get("history")
            ),
            "stadium_name": inputs.get("stadium_name") or prefixed_stadium,
            "intent": inputs.get("intent"),
            "origin": inputs.get("origin") or prefixed_origin,
        }

    def detail(self, inputs: Any) -> dict:
        """places·sources 까지 필요할 때. 결과를 last_detail() 로도 꺼낼 수 있게 놔둔다."""
        result = answer(**self._args(inputs))
        _LAST.set(result)
        return result

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs) -> str:
        return self.detail(input)["answer"]

    def stream(self, input: Any, config: Optional[RunnableConfig] = None,
               **kwargs) -> Iterator[str]:
        result = yield from dispatcher.stream(**self._args(input))
        _LAST.set(result)


rag_chain = RagChatChain()


def chat_chain() -> Optional[RagChatChain]:
    """RAG 체인을 돌려준다 (테스트 러너 안에서만 None → 성호 체인).

    ChatService 는 `self.chain = chat_chain() or self.get_chain()` 한 줄로 쓴다.
    """
    return rag_chain if use_rag() else None
