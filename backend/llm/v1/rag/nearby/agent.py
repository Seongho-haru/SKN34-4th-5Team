"""[nearby] 구장 주변 숙박·산책·실내놀거리·편의점 — RAG 에 없는 종류를 카카오 실시간 조회로 답한다 (담당: 형준)

흐름 (LLM 1회)
    ① 종류   질문 → stay / walk / indoor / store (키워드 규칙)                         LLM 0회
    ② 구장   질문 → 직전 사용자 발화 → 프론트가 넘긴 구장                              LLM 0회
    ③ 조회   kakao.nearby(구장, 종류) — 지도와 같은 조건(실제 구장 좌표·2.5km), 6시간 캐시  LLM 0회
    ④ 답변   가까운 순 + 세부 종류(호텔·볼링장 등) 필터 → 후보 8곳을 LLM 에 주고 3~5곳 소개     LLM 1회
             LLM 이 실패하면 코드가 목록으로 답한다

디스패처와의 약속: answer(question, history, hint_stadium) -> {"answer","sources","route"} · READY
places 는 돌려주지 않는다 — 프론트가 places 를 받으면 "추천 코스"로 보고 지도 코스를 바꾸기 때문.
"""
import logging
import os
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ...progress import ProgressCancelled, ProgressStorageError, operation
from ..club.router import detect_stadium
from ..domain_tools import run_model, visible_text
from ..persona import FIXED
from . import kakao
from .prompts import NO_KEY, NO_PLACES, SYSTEM, USER

log = logging.getLogger(__name__)

READY = True
LLM_MODEL = os.getenv("LLM_MODEL") or "gpt-5.6-luna"
WALK_M_PER_MIN = 80
SHOW = 8

KINDS = {
    "stay": re.compile(r"숙박|숙소|호텔|모텔|게스트\s*하우스|펜션|리조트|잘\s*곳|잘\s*데|묵을|1박|하룻밤|자고\s*갈"),
    "walk": re.compile(r"산책|공원|걷기|걸을\s*(곳|데|만한)|둘레길|산책로"),
    "indoor": re.compile(r"실내|볼링|보드\s*게임|방탈출|영화관|영화\s*볼|오락실|비\s*(가\s*)?오면\s*.{0,12}(갈|놀|할|볼)"),
    "store": re.compile(r"편의점"),
}
# 세부 종류 — 질문에 있으면 그 말이 카테고리에 들어간 곳만 (없으면 전체)
SUBTYPES = {
    "stay": ["호텔", "모텔", "게스트하우스", "펜션", "리조트"],
    "indoor": ["볼링", "보드게임", "방탈출", "영화", "오락실", "박물관", "미술관"],
}

STADIUM_KO = kakao.STADIUM_QUERY
_llm = None


def llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=LLM_MODEL, temperature=0, timeout=25, max_retries=0, reasoning_effort="medium", use_responses_api=True)
    return _llm


def kinds_of(question: str) -> list[str]:
    return [k for k, rx in KINDS.items() if rx.search(question or "")]


def wants(question: str) -> bool:
    """디스패처용 — 이 도메인이 답할 질문인가."""
    return bool(kinds_of(question))


def stadium_of(question, history, hint):
    code = detect_stadium(question)
    if code:
        return code, "question"
    for m in reversed(history or []):
        if m.get("role") == "user" and (c := detect_stadium(m.get("content") or "")):
            return c, "carry"
    return (hint, "hint") if hint else (None, "")


def narrow(places, kind, question):
    """'호텔' 처럼 세부 종류를 말했으면 그것만. 결과가 없으면 전체로 되돌린다. 같은 브랜드는 한 곳만."""
    words = [w for w in SUBTYPES.get(kind, []) if w in question.replace(" ", "")]
    picked = [p for p in places if any(w in f"{p['detail']} {p['name']}" for w in words)] if words else places
    picked = picked or places
    out, brands = [], set()
    for p in picked:
        brand = p["name"].split()[0]
        if brand in brands:
            continue
        brands.add(brand)
        out.append(p)
    return out


def walk_min(meters):
    return max(1, round(meters / WALK_M_PER_MIN))


def places_text(places):
    return "\n".join(f"- {p['name']} · {p['detail'].split(' > ')[-1] or p['kindLabel']} · 구장에서 {p['distance']}m"
                     f"(도보 {walk_min(p['distance'])}분) · {p['address']}" for p in places)


def template_answer(places, stadium_ko, kind_label):
    lines = [f"{stadium_ko} 가까운 {kind_label} 몇 곳이에요."]
    lines += [f"- {p['name']} (구장에서 도보 {walk_min(p['distance'])}분) — {p['detail'].split(' > ')[-1]}" for p in places[:5]]
    lines.append("카카오맵 기준이라 가시기 전에 영업·예약 여부를 한 번 확인해 보세요.")
    lines.append(f"옆 지도의 '{kind_label}' 카테고리에서 위치를 바로 볼 수 있어요.")
    return "\n".join(lines)


def _answer(question, history=None, hint_stadium=None):
    timing, route = {}, []
    kinds = kinds_of(question) or ["walk"]
    kind = kinds[0]                                   # 한 번에 한 종류 (여러 개면 앞의 것)
    code, how = stadium_of(question, history, hint_stadium)
    if not code or code == "OTHER":
        return {"answer": FIXED["clarify"], "sources": [], "route": "nearby:clarify", "timing": timing}
    route.append(f"{how}:{code}")
    stadium_ko, kind_label = STADIUM_KO.get(code, code), kakao.KIND_LABEL[kind]

    if not kakao.enabled():
        return {"answer": NO_KEY.format(stadium=stadium_ko, kind_label=kind_label), "sources": [],
                "route": f"nearby:{kind}:no_key", "timing": timing}
    t0 = time.perf_counter()
    with operation("tool", "search_nearby_places", arguments={"stadium_code": code, "kind": kind}):
        places = narrow(kakao.nearby(code, kind), kind, question)[:SHOW]
    timing["kakao_ms"] = round((time.perf_counter() - t0) * 1000)
    if not places:
        return {"answer": NO_PLACES.format(stadium=stadium_ko, kind_label=kind_label), "sources": [],
                "route": f"nearby:{kind}:empty", "timing": timing}

    sources = [{"doc_id": f"kakao:{p['placeId']}", "grade": "THIRD_PARTY", "category": kind.upper(), "stadium": code}
               for p in places]
    try:
        t0 = time.perf_counter()
        system = SYSTEM.replace("{kinds}", kind_label).replace("{kind_label}", kind_label)
        out = run_model(llm(), [SystemMessage(content=system), HumanMessage(content=USER.format(
            stadium=stadium_ko, places=places_text(places), question=question,
        ))], "nearby").content
        timing["llm_ms"] = round((time.perf_counter() - t0) * 1000)
        text = visible_text(out)
        text = text.strip() or template_answer(places, stadium_ko, kind_label)
    except (ProgressCancelled, ProgressStorageError):
        raise
    except Exception:
        log.exception("nearby llm failed")
        text = template_answer(places, stadium_ko, kind_label)
        route.append("template")
    route.append(f"nearby:{kind}:{len(places)}")
    return {"answer": text, "sources": sources, "route": " ".join(route), "timing": timing}


def answer(question, history=None, hint_stadium=None):
    from ..assistant.tools import request_state
    with request_state(hint_stadium, question, history):
        return _answer(question, history, hint_stadium)
