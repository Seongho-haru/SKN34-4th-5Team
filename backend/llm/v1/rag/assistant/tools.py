"""[assistant] 에이전트 도구 — 야구 DB 조회(읽기 전용 계정) + 문서 추가 검색 + 주변 장소 + 코스 짜기.

DB 조회 (BASEBALL_DB_USER 읽기 전용 계정 · 성호 BaseballQueryService 를 그대로 거친다)
    get_games               경기 일정·결과·남은 경기 수        (SQL 은 코드에 고정)
    get_standings           팀 순위                           (고정)
    get_ticket_prices       구단별 좌석 가격                    (고정)
    get_ticket_policy       예매 오픈·최대 매수·예매처          (고정)
    get_baseball_schema     위 4개로 안 되는 질문용 — 19개 테이블 스키마
    execute_baseball_select 위 4개로 안 되는 질문용 — LLM 이 쓴 SELECT 1문장 (스키마를 먼저 봐야 실행)
그 밖
    search_kbo_documents    RAG 추가 검색 (파이프라인이 이미 문서를 넣어 주지만, 다른 구장·다른 주제가 더 필요할 때)
    search_nearby_places    카카오 실시간 — 숙박·산책·실내놀거리·편의점 (지도와 같은 조건)
    plan_course             직관 코스 짜기 — 결과가 옆 지도·코스 카드로 이어진다

요청마다 상태(화면 구장·대화·스키마 확인 여부·쓴 도구·근거·코스 결과)를 ContextVar 에 둔다 → 동시 요청끼리 안 섞인다.
테스트는 _service/_search/_embed 같은 밑줄 인자로 가짜를 넣는다 (LLM 에게 보이는 도구 인자에는 없음).
"""
import contextvars
import json
from contextlib import contextmanager
from datetime import date, timedelta
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

STADIUMS = {"JAMSIL", "GOCHEOK", "MUNHAK", "SUWON", "DAEJEON", "DAEGU", "GWANGJU", "SAJIK", "CHANGWON"}
CATEGORIES = {"TRANSPORT", "FOOD_IN", "FOOD_OUT", "CAFE", "SPOT", "FACILITY", "CONTENT", "SEAT", "PRICE",
              "TICKET_POLICY", "CARRY_IN", "REENTRY", "RULE", "STADIUM", "OPERATION", "SCHEDULE", "STANDING"}
DOC_K = 5
MAX_SQL_ROWS = 50
FIXED_ROWS = 200                 # 고정 조회는 한 팀 한 시즌 경기(144)까지 한 번에 센다
OFFICIAL_STATUS = {"CONFIRMED", "CONFIRMED_OFFICIAL", "CONFIRMED_BASELINE", ""}

# 질문에 나오는 팀 표현 → TEAM.team_code (data/raw/team_stadium_code_map.csv)
TEAM_ALIASES = {
    "LG": ["LG", "엘지", "트윈스"], "DOOSAN": ["DOOSAN", "두산", "베어스"], "KIWOOM": ["KIWOOM", "키움", "히어로즈"],
    "SSG": ["SSG", "랜더스", "쓱"], "KT": ["KT", "케이티", "위즈"], "HANWHA": ["HANWHA", "한화", "이글스"],
    "SAMSUNG": ["SAMSUNG", "삼성", "라이온즈"], "KIA": ["KIA", "기아", "타이거즈"], "LOTTE": ["LOTTE", "롯데", "자이언츠"],
    "NC": ["NC", "엔씨", "다이노스"],
}
STATUS = {"upcoming": "PREV", "finished": "END", "canceled": "CANCEL"}

_STATE: contextvars.ContextVar[dict] = contextvars.ContextVar("assistant_state")


def _new_state(hint_stadium=None, question="", history=None):
    return {"hint": hint_stadium, "question": question, "history": history or [], "schema_seen": False,
            "tools": [], "sources": [], "course": None}


def new_state(hint_stadium=None, question="", history=None) -> dict:
    """테스트/직접 도구 호출용 상태를 만든다. 답변 진입점은 request_state를 쓴다."""
    st = _new_state(hint_stadium, question, history)
    _STATE.set(st)
    return st


@contextmanager
def request_state(hint_stadium=None, question="", history=None):
    """한 답변 요청 동안만 assistant 도구 상태를 공유하고 부모 상태를 복원한다."""
    st = _new_state(hint_stadium, question, history)
    token = _STATE.set(st)
    try:
        yield st
    finally:
        _STATE.reset(token)


def state() -> dict:
    try:
        return _STATE.get()
    except LookupError:
        return new_state()


def grade(row) -> str:
    ev, st = str(row.get("evidence_type") or "").upper(), str(row.get("status") or "").upper()
    if ev == "UNOFFICIAL":
        return "UNOFFICIAL"
    if ev == "THIRD_PARTY_API":
        return "THIRD_PARTY"
    return "OFFICIAL" if st in OFFICIAL_STATUS else "UNCERTAIN"


def team_code(value) -> str | None:
    v = str(value or "").strip().replace(" ", "").upper()
    if not v:
        return None
    for code, words in TEAM_ALIASES.items():
        if any(w.upper() == v or w.upper() in v for w in words):
            return code
    return None


def to_stadium_code(value) -> str | None:
    v = str(value or "").strip().upper()
    if v in STADIUMS:
        return v
    from ..club.router import detect_stadium        # "잠실", "챔피언스 필드" 같은 한글도 받는다
    code = detect_stadium(str(value or ""))
    return code if code in STADIUMS else None


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


# ── 야구 DB 조회 ─────────────────────────────────────────────────────────────
def _service(service=None):
    if service is not None:
        return service
    from baseball.query_service import BaseballQueryService
    return BaseballQueryService()


def _errors():
    try:
        from baseball.query_repository import BaseballQueryError
        from baseball.query_service import BaseballQueryValidationError
        return (BaseballQueryError, BaseballQueryValidationError)
    except Exception:                                  # 테스트 환경
        return (ValueError,)


def _run_fixed(name, sql, params, service=None) -> dict | str:
    """코드에 고정한 SELECT 를 읽기 전용 계정으로 실행 (성호 서비스의 검증·타임아웃을 그대로 거친다)."""
    s = state()
    s["tools"].append(name)
    try:
        result = _service(service).execute_baseball_select(sql, params, FIXED_ROWS)
    except _errors() as exc:
        return f"조회 오류: {exc}"
    s["sources"].append({"doc_id": f"baseball_db:{name}", "grade": "OFFICIAL", "category": "DB", "stadium": None})
    return result


def _rows(result) -> list[dict]:
    cols = result.get("columns") or []
    return [dict(zip(cols, r)) for r in result.get("rows") or []]


class GamesInput(BaseModel):
    team: str | None = Field(default=None, description="팀 이름 (예: KIA, 두산). 모르면 비운다")
    stadium: str | None = Field(default=None, description="구장 (예: 잠실, JAMSIL). 모르면 비운다")
    date_from: str | None = Field(default=None, description="YYYY-MM-DD. 비우면 오늘")
    date_to: str | None = Field(default=None, description="YYYY-MM-DD. 비우면 date_from 부터 시즌 끝까지")
    status: Literal["all", "upcoming", "finished", "canceled"] = Field(default="all", description="경기 전/끝난 경기/취소")
    home_away: Literal["all", "home", "away"] = Field(default="all", description="team 기준 홈경기만/원정만")
    limit: int = Field(default=10, description="보여줄 경기 수 (1~30). count 는 조건에 맞는 전체 수")


def get_games(team=None, stadium=None, date_from=None, date_to=None, status="all", home_away="all", limit=10,
              _service_obj=None) -> str:
    """경기 일정·결과·남은 경기 수를 야구 DB 에서 찾는다. count 는 조건에 맞는 전체 경기 수."""
    try:
        start = date.fromisoformat(date_from) if date_from else date.today()
        end = date.fromisoformat(date_to) if date_to else start + timedelta(days=400)
    except ValueError:
        return "날짜는 YYYY-MM-DD 형식으로 넣어 주세요."
    d_from, d_to = start.isoformat(), end.isoformat()
    conds, params = ["g.game_date >= %(d_from)s", "g.game_date <= %(d_to)s"], {"d_from": d_from, "d_to": d_to}
    code = team_code(team)
    if team and not code:
        return f"'{team}' 팀을 찾지 못했어요. 팀 이름으로 다시 조회하세요."
    if code:
        params["team"] = code
        conds.append({"home": "h.team_code = %(team)s", "away": "a.team_code = %(team)s"}.get(
            home_away, "(h.team_code = %(team)s OR a.team_code = %(team)s)"))
    st = to_stadium_code(stadium) if stadium else None
    if stadium and not st:
        return f"'{stadium}' 구장을 찾지 못했어요."
    if st:
        params["st"] = st
        conds.append("s.stadium_code = %(st)s")
    if status in STATUS:
        params["status"] = STATUS[status]
        conds.append("g.status_code = %(status)s")
    sql = ('SELECT g.game_date, g.game_time, h.team_name_ko AS home_team, a.team_name_ko AS away_team, '
           's.stadium_name_ko AS stadium, g.home_score, g.away_score, g.status_code '
           'FROM "GAME" g JOIN "TEAM" h ON h.id = g.home_team_id JOIN "TEAM" a ON a.id = g.away_team_id '
           'JOIN "STADIUM" s ON s.id = g.stadium_id WHERE ' + " AND ".join(conds) +
           " ORDER BY g.game_date, g.game_time")
    result = _run_fixed("games", sql, params, _service_obj)
    if isinstance(result, str):
        return result
    rows = _rows(result)
    n = max(1, min(int(limit or 10), 30))
    return _dumps({"count": len(rows), "count_is_partial": bool(result.get("truncated")),
                   "status_code 뜻": {"PREV": "경기 전", "END": "종료", "CANCEL": "취소"},
                   "games": rows[:n]})


class StandingsInput(BaseModel):
    as_of: str | None = Field(default=None, description="YYYY-MM-DD. 비우면 가장 최근 순위")


def get_standings(as_of=None, _service_obj=None) -> str:
    """팀 순위(승·패·무·게임차)를 야구 DB 에서 찾는다. as_of 이전 가장 최근 날짜 기준."""
    sql = ('SELECT sh.snapshot_date, sh.rank, t.team_name_ko, sh.wins, sh.losses, sh.draws, sh.games_behind '
           'FROM "STANDING_HISTORY" sh JOIN "TEAM" t ON t.id = sh.team_id '
           'WHERE sh.snapshot_date = (SELECT MAX(x.snapshot_date) FROM "STANDING_HISTORY" x WHERE x.snapshot_date <= %(d)s) '
           'ORDER BY sh.rank')
    try:
        as_of = date.fromisoformat(as_of).isoformat() if as_of else date.today().isoformat()
    except ValueError:
        return "날짜는 YYYY-MM-DD 형식으로 넣어 주세요."
    result = _run_fixed("standings", sql, {"d": as_of}, _service_obj)
    return result if isinstance(result, str) else _dumps({"standings": _rows(result)})


class PricesInput(BaseModel):
    team: str | None = Field(default=None, description="홈 구단 이름. 팀을 모르면 비운다")
    stadium: str | None = Field(default=None, description="구장 이름. 팀을 모를 때 사용한다")
    zone_keyword: str | None = Field(default=None, description="좌석 이름 일부 (예: 테이블, 외야, 1루). 모르면 비운다")
    cheapest_first: bool = Field(default=True, description="싼 순서로 정렬")
    limit: int = Field(default=15, description="보여줄 행 수 (1~30)")


def get_ticket_prices(team=None, stadium=None, zone_keyword=None, cheapest_first=True, limit=15, _service_obj=None) -> str:
    """구단 홈구장의 좌석 구역별 티켓 가격을 야구 DB 에서 찾는다 (요일·권종별)."""
    code = team_code(team) if team else None
    stadium_code = to_stadium_code(stadium) if stadium else None
    if team and not code:
        return f"'{team}' 팀을 찾지 못했어요."
    if stadium and not stadium_code:
        return f"'{stadium}' 구장을 찾지 못했어요."
    if not code and not stadium_code:
        return "구단이나 구장을 알려 주세요."
    conds, params = [], {}
    if code:
        conds.append("t.team_code = %(team)s")
        params["team"] = code
    if stadium_code:
        conds.append("s.stadium_code = %(stadium)s")
        params["stadium"] = stadium_code
    if zone_keyword:
        conds.append("sz.zone_name_ko ILIKE %(zone)s")
        params["zone"] = f"%{zone_keyword.strip()}%"
    sql = ('SELECT t.team_name_ko, sz.zone_name_ko, tp.day_type, tp.customer_type, tp.price_tier, tp.price_krw, tp.discount_condition '
           'FROM "TICKET_PRICE" tp JOIN "SEAT_ZONE" sz ON sz.id = tp.seat_zone_id '
           'JOIN "HOME_CONTEXT" hc ON hc.id = sz.home_context_id JOIN "TEAM" t ON t.id = hc.team_id '
           'JOIN "STADIUM" s ON s.id = hc.stadium_id '
           'WHERE ' + " AND ".join(conds) +
           (" ORDER BY tp.price_krw, sz.zone_name_ko" if cheapest_first else " ORDER BY sz.zone_name_ko, tp.price_krw"))
    result = _run_fixed("prices", sql, params, _service_obj)
    if isinstance(result, str):
        return result
    rows = _rows(result)
    n = max(1, min(int(limit or 15), 30))
    return _dumps({"count": len(rows), "prices": rows[:n],
                   "참고": "day_type·price_tier 는 구단마다 이름이 다르다 (WEEKDAY, WEEKEND_HOLIDAY, 색상 등급 등)"})


class PolicyInput(BaseModel):
    team: str = Field(description="구단 이름")


def get_ticket_policy(team, _service_obj=None) -> str:
    """구단 예매 정책(선예매·일반예매 조건, 최대 매수, 예매처)을 야구 DB 에서 찾는다."""
    code = team_code(team)
    if not code:
        return f"'{team}' 팀을 찾지 못했어요."
    sql = ('SELECT tp.policy_type, tp.subtype, tp.max_tickets, tp.booking_channel, tp.channel_condition '
           'FROM "TICKET_POLICY" tp JOIN "TEAM" t ON t.id = tp.team_id WHERE t.team_code = %(team)s '
           'ORDER BY tp.policy_type, tp.id')
    result = _run_fixed("policy", sql, {"team": code}, _service_obj)
    return result if isinstance(result, str) else _dumps({"policies": _rows(result)[:12]})


class NoInput(BaseModel):
    pass


class SelectInput(BaseModel):
    sql: str = Field(description="get_baseball_schema 에서 확인한 quoted 테이블만 쓰는 PostgreSQL SELECT 1문장")
    params: dict[str, Any] | None = Field(default=None, description="%(name)s 자리에 넣을 값")
    max_rows: int = Field(default=20, description=f"최대 행 수 (1~{MAX_SQL_ROWS})")


def get_baseball_schema(_service_obj=None) -> str:
    """야구 DB 19개 테이블의 컬럼·관계. 전용 조회 도구로 안 되는 질문에서 SQL 을 쓰기 전에 먼저 부른다."""
    s = state()
    s["tools"].append("schema")
    out = _service(_service_obj).get_baseball_schema()
    s["schema_seen"] = True
    return out if isinstance(out, str) else _dumps(out)


def execute_baseball_select(sql, params=None, max_rows=20, _service_obj=None) -> str:
    """읽기 전용 계정으로 SELECT 한 문장을 실행한다 (전용 조회 도구로 안 될 때만)."""
    s = state()
    s["tools"].append("sql")
    if not s["schema_seen"]:
        return "먼저 get_baseball_schema 로 테이블과 컬럼을 확인한 뒤 SQL 을 작성하세요."
    try:
        result = _service(_service_obj).execute_baseball_select(sql, params, max(1, min(int(max_rows or 20), MAX_SQL_ROWS)))
    except _errors() as exc:
        return f"조회 오류: {exc}"
    s["sources"].append({"doc_id": "baseball_db:sql", "grade": "OFFICIAL", "category": "DB", "stadium": None})
    return _dumps(result)


# ── RAG 추가 검색 ─────────────────────────────────────────────────────────────
class SearchInput(BaseModel):
    query: str = Field(description="검색할 내용 (예: '고척 주차 요금', '보조배터리 반입')")
    stadium_code: str | None = Field(default=None, description="구장 코드나 이름. 모르면 비운다")
    categories: list[str] | None = Field(default=None, description="TRANSPORT, FOOD_IN, FOOD_OUT, CAFE, SPOT, FACILITY, "
                                                                  "SEAT, CARRY_IN, REENTRY, RULE 중에서. 모르면 비운다")


def search_documents(query, stadium=None, categories=None, k=DOC_K, _search=None, _embed=None) -> list[dict]:
    """pgvector 검색 + 키워드 재정렬. 카테고리로 0건이면 카테고리를 풀고 한 번 더."""
    cats = [c.upper() for c in (categories or []) if c and c.upper() in CATEGORIES] or None
    if _search is None:
        from ..club.retrieval import embed, keyword_rerank, search
        _embed = embed
        _search = lambda v, kk, st, ct: keyword_rerank(query, search(v, k=kk, stadium=st, categories=ct)[0], k=k)  # noqa: E731
    vec = _embed(query)
    rows = _search(vec, k * 3, stadium, cats)
    if not rows and cats:
        rows = _search(vec, k * 3, stadium, None)
    return list(rows)[:k]


def format_documents(rows) -> str:
    if not rows:
        return "검색 결과 없음"
    return "\n\n".join(f"[{i}] 등급={grade(r)} · 구장={r.get('stadium') or '공통'} · 분류={r.get('category')}\n{r.get('content')}"
                       for i, r in enumerate(rows, 1))


def add_sources(rows):
    for r in rows:
        state()["sources"].append({"doc_id": r.get("doc_id"), "grade": grade(r), "category": r.get("category"),
                                   "stadium": r.get("stadium")})


def search_kbo_documents(query, stadium_code=None, categories=None, _search=None, _embed=None) -> str:
    """구장 안내 문서를 더 찾는다 (이미 받은 참고 문서에 없을 때만)."""
    s = state()
    s["tools"].append("rag")
    code = to_stadium_code(stadium_code) if stadium_code else s.get("hint")
    rows = search_documents(query, code, categories, _search=_search, _embed=_embed)
    add_sources(rows)
    return format_documents(rows)


# ── 카카오 주변 장소 · 코스 짜기 ───────────────────────────────────────────────
class NearbyInput(BaseModel):
    kind: Literal["stay", "walk", "indoor", "store"] = Field(description="stay 숙박 · walk 산책/공원 · indoor 실내놀거리 · store 편의점")
    stadium_code: str | None = Field(default=None, description="구장 코드나 이름. 모르면 비운다")
    keyword: str | None = Field(default=None, description="세부 종류 (예: 호텔, 게스트하우스, 볼링장)")


def search_nearby_places(kind, stadium_code=None, keyword=None, _nearby=None) -> str:
    """구장 반경 2.5km 의 숙박·산책·실내놀거리·편의점을 카카오맵에서 실시간으로 찾는다 (가까운 순)."""
    s = state()
    s["tools"].append(f"nearby:{kind}")
    code = to_stadium_code(stadium_code) if stadium_code else s.get("hint")
    if not code:
        return "어느 구장인지 모릅니다. 사용자에게 구장을 물어보세요."
    from ..nearby import agent as nearby_agent, kakao
    if _nearby is None and not kakao.enabled():
        return "카카오 장소 조회가 설정되지 않았습니다. 옆 지도의 해당 카테고리에서 확인하라고 안내하세요."
    places = nearby_agent.narrow((_nearby or kakao.nearby)(code, kind), kind, keyword or "")[:8]
    for p in places:
        s["sources"].append({"doc_id": f"kakao:{p['placeId']}", "grade": "THIRD_PARTY", "category": kind.upper(), "stadium": code})
    if not places:
        return "반경 2.5km 안에서 찾지 못했습니다."
    return _dumps([{"name": p["name"], "type": p["detail"].split(" > ")[-1], "walk_min": max(1, round(p["distance"] / 80)),
                    "address": p["address"]} for p in places])


class CourseInput(BaseModel):
    request: str = Field(description="사용자의 코스 요청을 그대로 (구장·일행·경기 전후·이동수단·숙박 여부가 들어가게)")


def plan_course(request, _course=None) -> str:
    """직관 코스(경기 전 → 구장 → 경기 후, 요청 시 숙소)를 짠다. 결과는 옆 지도와 코스 카드에 그대로 표시된다."""
    s = state()
    s["tools"].append("course")
    from ..domain_tools import active_domain
    if active_domain() == "course":
        return "현재 코스 생성 중에는 plan_course를 다시 호출할 수 없습니다. 주어진 후보로 답하세요."
    if _course is None:
        from ..course import agent as _course
    question = request if request and len(request) >= len(s.get("question") or "") else (s.get("question") or request)
    result = _course.answer(question, history=s.get("history"), hint_stadium=s.get("hint"))
    s["course"] = result
    s["sources"].extend(result.get("sources") or [])
    return result.get("answer") or "코스를 짜지 못했습니다."


def build_specialized_tools():
    def tool(fn, schema):
        return StructuredTool.from_function(fn, name=fn.__name__, args_schema=schema, description=fn.__doc__,
                                            handle_validation_error="도구 인자 형식이 올바르지 않습니다. 설명을 보고 다시 부르세요.")
    return (
        tool(get_games, GamesInput),
        tool(get_standings, StandingsInput),
        tool(get_ticket_prices, PricesInput),
        tool(get_ticket_policy, PolicyInput),
        tool(get_baseball_schema, NoInput),
        tool(execute_baseball_select, SelectInput),
        tool(search_kbo_documents, SearchInput),
        tool(search_nearby_places, NearbyInput),
        tool(plan_course, CourseInput),
    )


def build_tools():
    from ..domain_tools import tools_for
    return list(tools_for("assistant"))
