"""안내데스크 — 질문을 보고 course(직관 코스, 형준) / club(구단·야구, 형준) / venue(구장 안팎, 현준) 중 누가 답할지 정한다. LLM 호출 0회.

두 도메인 모듈이 지켜야 하는 약속은 딱 하나:
    answer(question: str, history: list[dict] | None, hint_stadium: str | None) -> dict
        반환 {"answer": str, "sources": list[dict], "route": str}
    READY: bool   # False 면 디스패처가 이 도메인으로 보내지 않는다 (아직 채우는 중)

history 는 [{"role": "user"|"assistant", "content": str}, ...] (이번 질문은 제외, 오래된 것부터).
hint_stadium 은 프론트 context.stadium 을 구장 코드(JAMSIL 등)로 바꾼 값. 없으면 None.
"""
import logging
import re

from . import persona
from .assistant import pipeline as assistant
from .club import agent as club
from .course import agent as course
from .nearby import agent as nearby
from .venue import agent as venue
from ..progress import ProgressCancelled, ProgressStorageError, operation

log = logging.getLogger(__name__)

# ── 도메인 판별 키워드 (디스패처 전용 · 각 도메인의 상세 사전과는 별개) ───────────
# 구장 안팎: 먹거리·시설·교통·주변 → venue
VENUE_WORDS = [
    "먹거리", "매점", "맛집", "음식점", "식당", "밥집", "치킨", "피자", "버거", "떡볶이", "스낵",
    "카페", "커피", "디저트", "음료수",
    "주차", "지하철", "버스", "셔틀", "교통", "출구", "오는길", "가는 법", "어떻게 가",
    "수유실", "화장실", "보관함", "물품보관", "보관소", "짐보관", "의무실", "휠체어", "충전소", "흡연", "편의시설", "유모차",
    "굿즈", "팀스토어", "포토", "포토존", "사진", "전시", "이벤트", "체험",
    "근처", "주변", "놀거리", "명소", "가볼", "관광", "주소", "전화", "문의처",
]
# 구단·야구: 일정·순위·가격·예매·좌석·반입·규칙 → club
CLUB_WORDS = [
    "순위", "몇 위", "승률", "게임차", "꼴찌", "선두",
    "경기", "일정", "몇 시", "결과", "스코어", "몇대몇", "상대", "홈경기", "다음",
    "예매", "선예매", "티켓", "가격", "얼마", "요금",
    "좌석", "시야", "응원석", "내야", "외야", "테이블석", "휠체어석",
    "반입", "가져가", "들고", "가방", "캔", "페트", "소주", "맥주", "술", "주류", "우산", "돗자리", "도시락", "외부 음식",
    "재입장", "나갔다",
    "규칙", "룰", "이닝", "스트라이크", "볼넷", "삼진", "홈런", "ABS", "피치클락", "비디오 판독", "연장", "우천", "취소", "노게임",
]

# 직관 코스 짜기 → course (메인 기능). 프론트가 intent="route" 를 주거나, 아래 표현이 있으면 코스로 간다.
# "맛집 추천해줘" 만으로는 venue (맛집 목록) — "코스·루트·동선·경기 전후·하루" 처럼 "순서를 짜 달라"는 뜻이 있어야 course.
COURSE = re.compile(r"코스|루트|동선|일정\s*짜|계획\s*(짜|세워)|경기\s*전\s*후|전후\s*(로|에|코스)|하루\s*(를|를\s*)?(짜|계획|보내)|"
                    r"경기\s*전에?\s*.{0,12}(경기\s*(끝|후)|끝나고)|뭐\s*하고\s*(놀|보내)|어떻게\s*보내|첫\s*직관")

# 야구 직관과 무관한 주제 — 어느 도메인으로도 보내지 않고 바로 범위 안내
OFF_TOPIC = re.compile(r"축구|K리그|농구|배구|골프|e스포츠|롤드컵|올림픽|월드컵|"
                       r"날씨|주식|코인|부동산|영화|드라마|아이돌|연예인|다이어트|"
                       r"코딩|파이썬|숙제|레시피|요리법")
# 야구 단어가 섞여 있어도 절대 답하지 않는 주제 — 직관 준비와 접점이 없어서 BASEBALL 예외를 안 준다.
# ("야구 좋아하는데 코딩 알려줘", "야구장 갈 때 탈 자동차 추천" 이 에이전트 LLM 으로 새는 것을 막는다 · 2026-09-16)
HARD_OFF = re.compile(r"코딩|파이썬|프로그래밍|숙제|과제\s*좀|레시피|요리법|주식|코인|비트코인|부동산|로또|"
                      r"(?:자동차|차량)\s*(?:추천|뭐\s*살|살까|구매|바꾸|바꿀)")
BASEBALL = re.compile(r"야구|KBO|구장|직관|경기|반입|재입장|좌석|예매|순위|선수|"
                      r"잠실|고척|문학|수원|대전|대구|광주|사직|창원|포항|"
                      r"LG|두산|키움|SSG|KT|한화|삼성|KIA|기아|롯데|NC", re.I)

# 프론트 context.stadium("잠실야구장") → 구장 코드
STADIUM_NAME_TO_CODE = {
    "잠실": "JAMSIL", "고척": "GOCHEOK", "문학": "MUNHAK", "인천": "MUNHAK", "랜더스": "MUNHAK",
    "수원": "SUWON", "대전": "DAEJEON", "대구": "DAEGU", "광주": "GWANGJU",
    "사직": "SAJIK", "부산": "SAJIK", "창원": "CHANGWON", "포항": "OTHER",
}


def _hits(question: str, words: list[str]) -> list[str]:
    q = question.replace(" ", "").upper()
    return [w for w in words if w.replace(" ", "").upper() in q]


# "얼마·요금" 은 주차 요금처럼 구장 쪽 질문에도 붙는다 → 구장 단어와 같이 나오면 club 쪽 신호로 세지 않는다
WEAK_CLUB_WORDS = {"얼마", "요금", "가격", "다음", "취소"}


def route(question: str, intent: str | None = None) -> str:
    """'course' | 'nearby' | 'venue' | 'club' | 'both' | 'scope'. intent 는 프론트 context.intent ("route"|"baseball"|"stadium")"""
    if HARD_OFF.search(question):
        return "scope"
    if OFF_TOPIC.search(question) and not BASEBALL.search(question):
        return "scope"
    if intent == "route" or COURSE.search(question):
        return "course"
    # 숙박·산책·실내놀거리·편의점 — RAG 에 없는 종류라 카카오 실시간 조회(nearby)로 보낸다 (2026-09-15)
    if nearby.READY and nearby.wants(question):
        return "nearby"
    v, c = _hits(question, VENUE_WORDS), _hits(question, CLUB_WORDS)
    if v:
        c = [w for w in c if w not in WEAK_CLUB_WORDS]
    if v and c:
        return "both"
    if v:
        return "venue"
    return "club"          # 아무 데도 안 걸리면 club — 되묻기·거절 가드가 거기 있다


def stadium_code_from_name(name: str | None) -> str | None:
    if not name:
        return None
    for key, code in STADIUM_NAME_TO_CODE.items():
        if key in name:
            return code
    return None


def _call(domain, question, history, hint_stadium, **extra):
    try:
        name = domain.__name__.split(".")[-2]
        with operation("phase", name):
            return domain.answer(question, history=history, hint_stadium=hint_stadium, **extra)
    except (ProgressCancelled, ProgressStorageError):
        raise
    except Exception:            # 한 도메인이 죽어도 챗봇 전체가 죽지 않게
        log.exception("rag domain failed: %s", domain.__name__)
        if domain is nearby:     # 카카오 조회가 죽으면 venue(RAG)라도 답하게
            r = _call(venue if venue.READY else club, question, history, hint_stadium)
            r["route"] = f"nearby:error>{r['route']}"
            return r
        if domain is course:     # course 가 죽으면 venue(준비됐으면) → club 순으로 맛집 목록이라도 준다
            r = _call(venue if venue.READY else club, question, history, hint_stadium)
            r["route"] = f"course:error>{r['route']}"
            return r
        if domain is venue:      # venue 가 죽으면 club 이 대신 답한다 (같은 DB 라 답은 나온다)
            try:
                r = club.answer(question, history=history, hint_stadium=hint_stadium)
                r["route"] = f"venue:error>club>{r['route']}"
                return r
            except (ProgressCancelled, ProgressStorageError):
                raise
            except Exception:
                log.exception("rag fallback failed")
        return {"answer": persona.FIXED["error"], "sources": [], "route": f"{domain.__name__}:error"}


def answer(question: str, history: list[dict] | None = None, stadium_name: str | None = None,
           intent: str | None = None, origin: dict | None = None) -> dict:
    """진입점 — 모든 질문이 assistant 파이프라인(프롬프트 · RAG · 에이전트[DB 조회 도구] · 파서)으로 간다.

    스위치 없음 (2026-09-15). 야구와 무관한 질문만 여기서 바로 돌려보내고,
    에이전트가 실패하면 같은 질문을 예전 도메인(course/nearby/club/venue)으로 한 번 더 답한다.
    반환 {"answer","sources","route","places","coursePayload"} — places·coursePayload 는 코스를 짰을 때만 채워진다.

    origin: 코스 작성 화면에서 지도에 찍은 출발지 {"lat","lng"}. 에이전트는 출발지를 모르므로,
    출발지가 있는 코스 질문은 출발지 기준으로 단계별로 장소를 찾는 course 도메인이 바로 답한다.
    """
    history = history or []
    hint = stadium_code_from_name(stadium_name)
    kind = route(question, intent)

    if kind == "scope":
        return {"answer": persona.FIXED["scope"], "sources": [], "route": "dispatcher:scope", "places": []}

    if _origin_course(kind, origin):
        result = _domain_answer(kind, question, history, hint, origin)
    else:
        try:
            with operation("phase", "assistant"):
                result = assistant.answer(question, history=history, hint_stadium=hint)
        except (ProgressCancelled, ProgressStorageError):
            raise
        except Exception:
            log.exception("assistant pipeline failed — falling back to domain")
            result = _domain_answer(kind, question, history, hint, origin)
            result["route"] = f"agent:error>{result['route']}"

    result["answer"] = persona.finalize(result["answer"])
    result.setdefault("places", [])
    result.setdefault("coursePayload", None)   # 코스를 짰을 때만 — 프론트 지도·"이 코스 저장하기" 용
    return result


def _origin_course(kind: str, origin: dict | None) -> bool:
    """출발지가 찍힌 코스 질문인가 — 이때는 course 도메인이 출발지부터 이어서 코스를 짠다."""
    return bool(origin) and kind == "course" and course.READY


def stream(question: str, history: list[dict] | None = None, stadium_name: str | None = None,
           intent: str | None = None, origin: dict | None = None):
    """assistant의 마지막 provider 응답만 흘리고 완료 메타데이터를 반환한다."""
    history = history or []
    hint = stadium_code_from_name(stadium_name)
    kind = route(question, intent)
    if kind == "scope":
        result = {"answer": persona.FIXED["scope"], "sources": [], "route": "dispatcher:scope", "places": []}
        yield result["answer"]
        return result
    if _origin_course(kind, origin):
        result = _domain_answer(kind, question, history, hint, origin)
        result["answer"] = persona.finalize(result["answer"])
        yield result["answer"]
        result.setdefault("places", [])
        result.setdefault("coursePayload", None)
        return result
    emitted = False
    try:
        with operation("phase", "assistant"):
            stream = assistant.stream_answer(question, history=history, hint_stadium=hint)
            while True:
                try:
                    chunk = next(stream)
                except StopIteration as done:
                    result = done.value
                    break
                emitted = True
                yield chunk
    except (ProgressCancelled, ProgressStorageError):
        raise
    except Exception:
        if emitted:
            raise
        log.exception("assistant stream failed — falling back to domain")
        result = _domain_answer(kind, question, history, hint, origin)
        result["route"] = f"agent:error>{result['route']}"
        result["answer"] = persona.finalize(result["answer"])
        yield result["answer"]
    result.setdefault("places", [])
    result.setdefault("coursePayload", None)
    return result


def _domain_answer(kind, question, history, hint, origin=None) -> dict:
    """예전 도메인 라우팅 — 에이전트 파이프라인이 실패했을 때만 쓴다."""
    use_venue = venue.READY
    if kind == "course" and course.READY:
        result = _call(course, question, history, hint, **({"origin": origin} if origin else {}))
        result["route"] = f"course>{result['route']}"
    elif kind == "nearby":
        result = _call(nearby, question, history, hint)
        result["route"] = f"nearby>{result['route']}"
    elif kind == "venue":
        result = _call(venue if use_venue else club, question, history, hint)
        result["route"] = f"venue>{result['route']}" if use_venue else f"venue(not ready)>club>{result['route']}"
    elif kind == "both" and use_venue:
        # 복합 질문: 두 도메인에 각각 묻고 이어 붙인다 (3차 범위. LLM 최대 2회)
        a, b = _call(club, question, history, hint), _call(venue, question, history, hint)
        result = {
            "answer": f"{a['answer']}\n\n{b['answer']}",
            "sources": a["sources"] + b["sources"],
            "route": f"both>club[{a['route']}]+venue[{b['route']}]",
        }
    else:
        result = _call(club, question, history, hint)
        result["route"] = f"club>{result['route']}"
    return result
