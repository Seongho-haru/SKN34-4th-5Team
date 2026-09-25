"""[venue] 구장 안팎 도메인 — 먹거리·편의시설·교통·포토존·주변 맛집 (담당: 현준)

현준 test.py(LangChain Agent + search_documents_tool)를 Django 로 옮긴 것.
흐름: 구장 슬롯(별칭 사전·직전 대화·프론트 힌트) → 되묻기 가드 → Agent(create_agent)가 search_documents_tool 호출 여부 판단
      → 도구 안에서 검색어 변환(LLM) → 벡터 검색(구장·카테고리·안/밖 필터) → 부족하면 키워드 fallback → 답변 생성
디스패처와의 약속: answer(question, history, hint_stadium) -> {"answer", "sources", "route"} · READY

test.py 에서 바뀐 것:
  - psycopg2 풀 → django.db.connection (검색 SQL 은 club/retrieval.search 재사용, fallback 은 여기 구현)
  - "다른 Search Agent 담당" 차단(순위·일정·티켓) 삭제 — 디스패처가 그 질문을 여기로 안 보낸다
  - 제외 카테고리(SCHEDULE·PRICE·SEAT·TICKET_POLICY) 대신 담당 카테고리 화이트리스트(VENUE_CATEGORIES)
  - 프롬프트 말투는 ../persona 에서, 근거 등급(grade)을 sources 에 넣음
  - 구단 미지정 되묻기는 프롬프트에만 맡기지 않고 코드로도 막는다(채점 가능하게)
"""
import contextvars
import json
import logging
import os
import re
import time

from django.db import connection, transaction
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from ...progress import config_kwargs

from ..club.retrieval import embed                # 질문 임베딩만 재사용 (같은 임베딩 모델)
from ..domain_tools import tools_for, visible_text
from ..persona import FIXED
from .prompts import NO_DOCUMENTS_MESSAGE, QUERY_TRANSFORM, SYSTEM

log = logging.getLogger(__name__)

try:
    from langchain.agents import create_agent      # langchain>=1.0
    READY = True
except ImportError:                                 # langchain 패키지가 없으면 디스패처가 club 으로 보낸다
    create_agent = None
    READY = False
    log.warning("langchain.agents.create_agent 를 불러올 수 없어 venue 도메인을 비활성화합니다 (pip install langchain)")

LLM_MODEL = os.getenv("LLM_MODEL") or "gpt-5.6-luna"
TOP_K = int(os.getenv("VENUE_TOP_K", "10"))
MAX_DISTANCE = float(os.getenv("VENUE_MAX_DISTANCE", "0.5"))   # cosine distance 초기 threshold (test.py 와 동일)
VENUE_CATEGORIES = ["FOOD_IN", "FOOD_OUT", "CAFE", "SPOT", "FACILITY", "CONTENT", "TRANSPORT", "STADIUM", "OPERATION"]

# ── 별칭 → (team_code, stadium_code)  (test.py TEAM_ALIASES 그대로) ──────────
TEAM_ALIASES: dict[str, tuple[str, str]] = {
    "SSG랜더스": ("SSG", "MUNHAK"), "인천SSG랜더스필드": ("SSG", "MUNHAK"), "SSG랜더스필드": ("SSG", "MUNHAK"),
    "랜더스필드": ("SSG", "MUNHAK"), "쓱랜더스": ("SSG", "MUNHAK"), "문학경기장": ("SSG", "MUNHAK"),
    "문학구장": ("SSG", "MUNHAK"), "인천야구장": ("SSG", "MUNHAK"), "인천구장": ("SSG", "MUNHAK"),
    "랜더스": ("SSG", "MUNHAK"), "SSG": ("SSG", "MUNHAK"), "문학": ("SSG", "MUNHAK"), "쓱": ("SSG", "MUNHAK"),
    "LG트윈스": ("LG", "JAMSIL"), "엘지트윈스": ("LG", "JAMSIL"), "트윈스": ("LG", "JAMSIL"),
    "LG": ("LG", "JAMSIL"), "엘지": ("LG", "JAMSIL"),
    "두산베어스": ("DOOSAN", "JAMSIL"), "베어스": ("DOOSAN", "JAMSIL"), "두산": ("DOOSAN", "JAMSIL"),
    "잠실야구장": ("LG", "JAMSIL"), "잠실구장": ("LG", "JAMSIL"), "잠실": ("LG", "JAMSIL"),
    "KIA타이거즈": ("KIA", "GWANGJU"), "기아타이거즈": ("KIA", "GWANGJU"), "챔피언스필드": ("KIA", "GWANGJU"),
    "광주야구장": ("KIA", "GWANGJU"), "광주구장": ("KIA", "GWANGJU"), "무등구장": ("KIA", "GWANGJU"),
    "타이거즈": ("KIA", "GWANGJU"), "챔필": ("KIA", "GWANGJU"), "KIA": ("KIA", "GWANGJU"), "기아": ("KIA", "GWANGJU"),
    "삼성라이온즈": ("SAMSUNG", "DAEGU"), "라이온즈파크": ("SAMSUNG", "DAEGU"), "대구야구장": ("SAMSUNG", "DAEGU"),
    "대구구장": ("SAMSUNG", "DAEGU"), "라이온즈": ("SAMSUNG", "DAEGU"), "삼성": ("SAMSUNG", "DAEGU"), "라팍": ("SAMSUNG", "DAEGU"),
    "롯데자이언츠": ("LOTTE", "SAJIK"), "사직야구장": ("LOTTE", "SAJIK"), "사직구장": ("LOTTE", "SAJIK"),
    "부산야구장": ("LOTTE", "SAJIK"), "부산구장": ("LOTTE", "SAJIK"), "자이언츠": ("LOTTE", "SAJIK"),
    "롯데": ("LOTTE", "SAJIK"), "사직": ("LOTTE", "SAJIK"),
    "한화이글스": ("HANWHA", "DAEJEON"), "한화생명볼파크": ("HANWHA", "DAEJEON"), "이글스파크": ("HANWHA", "DAEJEON"),
    "대전야구장": ("HANWHA", "DAEJEON"), "대전구장": ("HANWHA", "DAEJEON"), "이글스": ("HANWHA", "DAEJEON"), "한화": ("HANWHA", "DAEJEON"),
    "케이티위즈": ("KT", "SUWON"), "KT위즈": ("KT", "SUWON"), "수원야구장": ("KT", "SUWON"),
    "수원구장": ("KT", "SUWON"), "위즈파크": ("KT", "SUWON"), "케이티": ("KT", "SUWON"), "위즈": ("KT", "SUWON"), "KT": ("KT", "SUWON"),
    "NC다이노스": ("NC", "CHANGWON"), "엔씨다이노스": ("NC", "CHANGWON"), "창원야구장": ("NC", "CHANGWON"),
    "창원구장": ("NC", "CHANGWON"), "마산구장": ("NC", "CHANGWON"), "다이노스": ("NC", "CHANGWON"),
    "엔씨파크": ("NC", "CHANGWON"), "NC파크": ("NC", "CHANGWON"), "엔씨": ("NC", "CHANGWON"), "엔팍": ("NC", "CHANGWON"), "NC": ("NC", "CHANGWON"),
    "키움히어로즈": ("KIWOOM", "GOCHEOK"), "고척스카이돔": ("KIWOOM", "GOCHEOK"), "히어로즈": ("KIWOOM", "GOCHEOK"),
    "고척돔": ("KIWOOM", "GOCHEOK"), "서울돔": ("KIWOOM", "GOCHEOK"), "키움": ("KIWOOM", "GOCHEOK"), "고척": ("KIWOOM", "GOCHEOK"),
}
_ALIASES_LONGEST_FIRST = sorted(TEAM_ALIASES, key=len, reverse=True)   # "잠실야구장" 이 "잠실" 보다 먼저

# ── 질문 → 필터 추론 (test.py infer_keyword_filters) ─────────────────────────
CATEGORY_RULES = [
    (("교통", "대중교통", "버스", "지하철", "주차", "주차장", "오는길", "가는 법"), ["TRANSPORT"]),
    (("포토존", "포토", "사진", "포토부스", "최정존", "기념존", "전시"), ["CONTENT", "FACILITY", "SPOT"]),
    (("수유실", "수유", "유모차", "의무실", "의무", "물품보관소", "보관소", "짐보관", "화장실", "휠체어", "충전소", "편의시설", "흡연"), ["FACILITY", "STADIUM", "CONTENT"]),
    (("카페", "커피", "음료", "디저트"), ["CAFE", "FOOD_IN", "FOOD_OUT"]),
]
OUTSIDE_WORDS = ("근처", "주변", "외부", "밖")
INSIDE_WORDS = ("내부", "구장 안", "구장안", "안에서", "안에", "1루", "3루", "내야", "외야")
FOOD_WORDS = ("먹거리", "음식점", "식당", "매장", "치킨", "피자", "버거", "스낵", "맛집", "밥", "떡볶이", "만두")


def infer_slots(question: str) -> dict:
    """구장·팀·카테고리·안/밖 추론"""
    slots: dict = {}
    compact = question.replace(" ", "")
    for alias in _ALIASES_LONGEST_FIRST:
        if alias.upper() in compact.upper():
            slots["team_code"], slots["stadium_code"] = TEAM_ALIASES[alias]
            break
    for terms, cats in CATEGORY_RULES:
        if any(t in question for t in terms):
            slots["categories"] = cats
            break
    else:
        if any(t in question for t in OUTSIDE_WORDS) and any(t in question for t in FOOD_WORDS):
            slots["categories"] = ["FOOD_OUT", "CAFE"]
        elif any(t in question for t in FOOD_WORDS):
            # 내부 먹거리 질문에 외부 매장과 카페가 섞이지 않도록 원본 규칙을 유지한다.
            slots["categories"] = ["FOOD_IN", "CONTENT", "FACILITY"]
    if any(t in question for t in OUTSIDE_WORDS):
        slots["in_stadium_flag"] = "N"
    elif any(t in question for t in INSIDE_WORDS):
        slots["in_stadium_flag"] = "Y"
    return slots


# ── 근거 등급 (club 과 같은 기준) ─────────────────────────────────────────────
def grade(meta: dict) -> str:
    ev = str(meta.get("evidence_type") or "").upper()
    st = str(meta.get("status") or "").upper()
    if ev == "UNOFFICIAL":
        return "UNOFFICIAL"
    if ev == "THIRD_PARTY_API":
        return "THIRD_PARTY"
    if st in {"CONFIRMED", "CONFIRMED_OFFICIAL", "CONFIRMED_BASELINE"} or (not st and ev in {"", "OFFICIAL", "DERIVED"}):
        return "OFFICIAL"
    return "UNCERTAIN"


# ── 검색 ────────────────────────────────────────────────────────────────────
def _metadata_dict(metadata) -> dict:
    """DB driver가 반환한 metadata를 항상 dict로 정규화한다."""
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            value = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def vector_search(query: str, stadium: str | None, categories: list[str] | None,
                  in_stadium_flag: str | None = None) -> list[dict]:
    """구장·카테고리·안팎 조건을 적용한 벡터 검색."""
    qvec = embed(query)
    conds = ["metadata->>'category' = ANY(%(cats)s)"]
    params: dict = {"cats": categories or VENUE_CATEGORIES, "v": "[" + ",".join(map(str, qvec)) + "]", "k": TOP_K}
    if stadium:
        conds.append("(metadata->>'stadium_code' = %(st)s OR metadata->>'stadium_code' IS NULL)")
        params["st"] = stadium
    if in_stadium_flag:
        conds.append("metadata->>'in_stadium_flag' = %(flag)s")
        params["flag"] = in_stadium_flag
    sql = f"""
        SELECT content, metadata, embedding <=> %(v)s::vector AS dist
        FROM llm_documentchunk WHERE {' AND '.join(conds)}
        ORDER BY embedding <=> %(v)s::vector LIMIT %(k)s"""
    with transaction.atomic(), connection.cursor() as cur:
        cur.execute("SET LOCAL hnsw.ef_search = 200")
        cur.execute(sql, params)
        rows = cur.fetchall()
    out = []
    for content, meta, dist in rows:
        if float(dist) > MAX_DISTANCE:
            continue
        meta = _metadata_dict(meta)
        out.append({"content": content, "metadata": {**meta, "distance": float(dist), "match_type": "vector"}})
    return out


def _keyword_patterns(query: str) -> list[str]:
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", query)
    aliases = {"포토존": "PHOTOZONE", "사진": "PHOTOZONE", "교통": "TRANSPORT", "대중교통": "TRANSPORT"}
    tokens.extend(aliases[t] for t in tokens if t in aliases)
    return [f"%{t}%" for t in dict.fromkeys(tokens) if len(t) >= 2]


def keyword_fallback_search(query: str, stadium: str | None, categories: list[str] | None,
                            in_stadium_flag: str | None, top_k: int) -> list[dict]:
    """벡터가 부족할 때 content/metadata 를 ILIKE 로 보조 검색 (test.py keyword_fallback_search)"""
    patterns = _keyword_patterns(query)
    if not patterns:
        return []
    conds, params = ["(content ILIKE ANY(%s) OR metadata::text ILIKE ANY(%s))"], [patterns, patterns]
    conds.append("metadata->>'category' = ANY(%s)")
    params.append(categories or VENUE_CATEGORIES)
    if in_stadium_flag:
        conds.append("metadata->>'in_stadium_flag' = %s")
        params.append(in_stadium_flag)
    if stadium:
        conds.append("(metadata->>'stadium_code' = %s OR metadata->>'stadium_code' IS NULL)")
        params.append(stadium)
    params.append(max(top_k * 100, 500))
    with connection.cursor() as cur:
        cur.execute(f"SELECT content, metadata FROM llm_documentchunk WHERE {' AND '.join(conds)} ORDER BY id LIMIT %s", params)
        rows = cur.fetchall()

    def score(row):
        text = f"{row[0]} {json.dumps(row[1] or {}, ensure_ascii=False)}".lower()
        return sum(p.strip("%").lower() in text for p in patterns)

    rows = sorted(rows, key=score, reverse=True)[:top_k]
    return [{"content": c, "metadata": {**_metadata_dict(m), "distance": None, "match_type": "keyword_fallback"}}
            for c, m in rows]


def search_documents(query: str, slots: dict) -> dict:
    """검색어 변환 → 벡터 검색 → 소프트 필터 → 키워드 fallback (test.py search_documents)"""
    transformed = transform_query(query)
    slots = {**infer_slots(f"{query} {transformed}"), **{k: v for k, v in slots.items() if v}}
    stadium, cats, flag = slots.get("stadium_code"), slots.get("categories"), slots.get("in_stadium_flag")

    docs = vector_search(transformed, stadium, cats, flag)
    method = "vector" if docs else "none"

    fetch_k = max(TOP_K * 2, 10)
    if not docs or len(docs) < 3:
        # 내부/외부가 명시된 경우 fallback에서도 그 조건을 풀지 않는다.
        fb = keyword_fallback_search(transformed, stadium, cats, flag, fetch_k)
        if not fb and transformed != query:
            fb = keyword_fallback_search(query, stadium, cats, flag, fetch_k)
        if fb:
            docs, method = fb, "keyword_fallback"
    docs = docs[:fetch_k]
    return {"original_query": query, "transformed_query": transformed, "documents": docs, "search_method": method}


def format_documents(docs: list[dict]) -> str:
    if not docs:
        return NO_DOCUMENTS_MESSAGE
    return "\n\n".join(
        f"[문서 {i}] (등급={grade(d['metadata'])} · 구장={d['metadata'].get('stadium_code')} · 카테고리={d['metadata'].get('category')})\n{d['content']}"
        for i, d in enumerate(docs, 1)
    )


# ── LLM · Agent (서버 기동 후 1회 생성) ──────────────────────────────────────
_llm = None
_transformer = None
_agent = None
_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar("venue_ctx")   # 도구가 슬롯·검색결과를 주고받는 통로


def llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=LLM_MODEL, temperature=0, timeout=25, max_retries=0, reasoning_effort="medium", use_responses_api=True)
    return _llm


def transform_query(query: str) -> str:
    """검색어만 다듬는 보조 LLM 호출이며 답변 에이전트/도구 루프가 아니다."""
    global _transformer
    if _transformer is None:
        prompt = ChatPromptTemplate.from_messages([("system", QUERY_TRANSFORM), ("human", "{query}")])
        _transformer = prompt | llm() | StrOutputParser()
    return _transformer.invoke({"query": query}).strip() or query


@tool
def search_documents_tool(query: str) -> str:
    """구장 안팎 문서(먹거리·편의시설·교통·포토존·주변 맛집)에서 질문과 관련된 근거를 검색한다."""
    ctx = _ctx.get({})
    result = search_documents(query, ctx.get("slots", {}))
    ctx["last"] = result
    return format_documents(result["documents"])


def agent():
    global _agent
    if _agent is None:
        _agent = create_agent(model=llm(), tools=tools_for("venue"), system_prompt=SYSTEM)
    return _agent


# ── 진입점 ──────────────────────────────────────────────────────────────────
def _stadium_from_history(history):
    for m in reversed(history or []):
        if m.get("role") == "user":
            s = infer_slots(m["content"])
            if s.get("stadium_code"):
                return s
    return {}


def _answer(question, history=None, hint_stadium=None):
    if not READY:
        raise RuntimeError("venue 도메인 비활성화 (langchain 패키지 없음)")
    history = history or []
    timing, route = {}, []

    slots = infer_slots(question)
    if not slots.get("stadium_code"):
        prev = _stadium_from_history(history)
        if prev:
            slots["stadium_code"], slots["team_code"] = prev["stadium_code"], prev.get("team_code")
            route.append(f"carry:{prev['stadium_code']}")
        elif hint_stadium:
            slots["stadium_code"] = hint_stadium
            route.append(f"hint:{hint_stadium}")
    if not slots.get("stadium_code"):
        return {"answer": FIXED["clarify"], "sources": [], "route": "guard:clarify", "timing": timing}

    ctx = {"slots": slots}
    token = _ctx.set(ctx)
    try:
        messages = [*[{"role": m["role"], "content": m["content"]} for m in history[-6:]],
                    {"role": "user", "content": question}]
        t0 = time.perf_counter()
        result = agent().invoke(
            {"messages": messages},
            **config_kwargs(recursion_limit=6),
        )
        timing["agent_ms"] = round((time.perf_counter() - t0) * 1000)
    finally:
        _ctx.reset(token)

    final = result["messages"][-1]
    text = visible_text(final.content)
    tool_called = any(type(m).__name__ == "ToolMessage" for m in result["messages"])

    last = ctx.get("last") or {}
    docs = last.get("documents", [])
    sources = [{"doc_id": d["metadata"].get("doc_id"), "grade": grade(d["metadata"]),
                "category": d["metadata"].get("category"), "stadium": d["metadata"].get("stadium_code"),
                "match_type": d["metadata"].get("match_type")} for d in docs]
    route.append(f"agent:{'tool' if tool_called else 'no_tool'}:{last.get('search_method', '-')}"
                 f":{slots.get('stadium_code')}:{','.join(slots.get('categories') or []) or '-'}")
    return {"answer": text, "sources": sources, "route": " ".join(route), "timing": timing}


def answer(question, history=None, hint_stadium=None):
    from ..assistant.tools import request_state
    with request_state(hint_stadium, question, history):
        return _answer(question, history, hint_stadium)
