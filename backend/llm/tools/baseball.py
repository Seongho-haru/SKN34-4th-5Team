from typing import Any

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field, StrictInt

from baseball.query_repository import BaseballQueryError
from baseball.query_service import BaseballQueryService, BaseballQueryValidationError


class ExecuteBaseballSelectInput(BaseModel):
    sql: str = Field(
        description=(
            "get_baseball_schema로 확인한 public 야구 테이블의 quoted_name만 사용하는 "
            "단일 PostgreSQL SELECT. JOIN, 집계, 서브쿼리, 비재귀 CTE를 포함할 수 있다."
        )
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description="SQL의 %(name)s named parameter 값. 문자열 보간 대신 사용한다.",
    )
    max_rows: StrictInt = Field(
        default=100,
        description="반환할 최대 행 수. 서비스 설정 범위 안의 정수여야 한다.",
    )


def create_baseball_tools(service: BaseballQueryService | None = None):
    """기존 BaseballQueryService를 호출하는 LangChain 도구 두 개를 만든다."""
    service = service or BaseballQueryService()

    def schema() -> dict:
        """SQL 작성 전에 public 야구 테이블, quoted_name, 컬럼, FK 관계를 조회한다."""
        try:
            return service.get_baseball_schema()
        except Exception as exc:
            raise ToolException(f"[조회 실패] 스키마 조회 오류: {type(exc).__name__}") from None

    def select(sql: str, params: dict[str, Any] | None = None, max_rows: int = 100) -> dict:
        """스키마에 있는 quoted 테이블만 대상으로 단일 읽기 SELECT를 실행한다."""
        try:
            return service.execute_baseball_select(sql, params, max_rows)
        except (BaseballQueryValidationError, BaseballQueryError) as exc:
            raise ToolException(str(exc)) from None

    invalid_input = "도구 입력 형식이 올바르지 않습니다. 스키마와 인자 설명을 확인하세요."
    return (
        StructuredTool.from_function(
            schema,
            name="get_baseball_schema",
            description=(
                "야구 SQL 작성의 첫 단계로 호출한다. public 스키마의 허용 테이블 quoted_name, "
                "컬럼 타입, PK/null 여부, FK 관계를 반환한다."
            ),
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            select,
            name="execute_baseball_select",
            description=(
                "get_baseball_schema 결과를 바탕으로 PostgreSQL 단일 SELECT를 실행한다. "
                "quoted 테이블 JOIN, 집계, 서브쿼리, 비재귀 CTE와 %(name)s named params를 "
                "지원하며 결과 행 수는 max_rows로 제한한다."
            ),
            args_schema=ExecuteBaseballSelectInput,
            handle_tool_error=True,
            handle_validation_error=invalid_input,
        ),
    )


get_baseball_schema, execute_baseball_select = create_baseball_tools()


"""baseball domain tools."""
from datetime import date, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo
from pydantic import AwareDatetime, Field, StrictBool, StrictInt, model_validator

from .common import LimitInput, _json, _result, _rows, _tool, db_team_code, is_team_code, tving_team_code

class StandingsInput(LimitInput):
    snapshot_date: date | None = None
    as_of: date | None = Field(default=None, description="기존 V1 날짜 별칭. 해당 날짜 이전 가장 최근 순위.")
    team_code: str | None = None
    include_detail: StrictBool = Field(default=False, description="현재 저장된 구단 기록·팀내순위·선수단 상세 포함. 과거 순위와 별도 시점이다.")

    @model_validator(mode="after")
    def validate_team(self):
        if self.team_code and not is_team_code(self.team_code):
            raise ValueError("올바른 팀 코드가 아닙니다.")
        return self

def _game_range(start_date=None, end_date=None, upcoming=False):
    today = datetime.now(ZoneInfo("Asia/Seoul")).date()
    explicit = start_date is not None or end_date is not None
    start_date = start_date or (end_date if explicit else today)
    end_date = end_date or (start_date if explicit else today + timedelta(days=366 if upcoming else 6))
    return start_date, end_date


class GamesInput(LimitInput):
    start_date: date | None = None
    end_date: date | None = None
    date_from: date | None = None
    date_to: date | None = None
    team: str | None = None
    stadium: str | None = None
    status: Literal["all", "upcoming", "finished", "canceled"] = "all"
    home_away: Literal["all", "home", "away"] = "all"
    team_code: str | None = Field(default=None, pattern="^[A-Z]{2,7}$")
    stadium_id: StrictInt | None = Field(default=None, ge=1)
    upcoming_only: StrictBool = Field(default=False, description="as_of 이후 저장된 시작 시각의 예정 경기만 조회.")
    as_of: AwareDatetime | None = None
    offset: StrictInt = Field(default=0, ge=0, le=1_000_000)

    @model_validator(mode="after")
    def validate_range(self):
        self.start_date, self.end_date = _game_range(self.start_date or self.date_from,
                                                    self.end_date or self.date_to,
                                                    self.upcoming_only or self.status == "upcoming")
        if self.start_date > self.end_date or (self.end_date - self.start_date).days > 366:
            raise ValueError("날짜 범위는 순서대로 최대 366일이어야 합니다.")
        if self.team_code and not is_team_code(self.team_code):
            raise ValueError("올바른 팀 코드가 아닙니다.")
        return self

class PlayerInput(LimitInput):
    team_code: str | None = Field(default=None, pattern="^(SS|KT|LG|HT|OB|NC|HH|LT|SK|WO|SAMSUNG|KIA|DOOSAN|HANWHA|LOTTE|SSG|KIWOOM)$")
    player_code: str | None = Field(default=None, min_length=1, max_length=40)
    name: str | None = Field(default=None, min_length=1, max_length=80)
    include_detail: StrictBool | None = Field(default=None, description="프로필·시즌·통산 기록 포함. 선수 코드 조회는 기본 포함한다.")
    offset: StrictInt = Field(default=0, ge=0, le=1_000_000)

    @model_validator(mode="after")
    def any_filter(self):
        if not any((self.team_code, self.player_code, self.name)):
            raise ValueError("구단, 선수 코드, 이름 중 하나가 필요합니다.")
        return self

def create_baseball_domain_tools():
    from django.db.models import Q
    from baseball.models import Game, StandingHistory
    from tving import service as tving_service

    def get_standings(snapshot_date=None, limit=20, team_code=None, include_detail=False, as_of=None):
        """저장된 순위와 선택 구단의 현재 상세(기록·팀내순위·선수단)를 조회한다."""
        from baseball.models import TeamProfile
        from tving.relational import read_team, team_sync_time
        requested = snapshot_date or as_of
        try:
            freshness = tving_service.get_standings_freshness(requested)
        except tving_service.TvingError:
            freshness = {"stale": True, "warning": "최신 순위를 확인하지 못해 저장된 자료만 조회합니다."}
        snapshots = StandingHistory.objects.all()
        if as_of and not snapshot_date:
            snapshots = snapshots.filter(snapshot_date__lte=as_of)
        actual = snapshot_date or snapshots.order_by("-snapshot_date").values_list("snapshot_date", flat=True).first()
        query = StandingHistory.objects.filter(snapshot_date=actual)
        if team_code:
            query = query.filter(team__team_code=db_team_code(team_code))
        rows = [] if actual is None else _rows(query.order_by("rank", "team__team_code"), (
            "team__team_code", "team__team_name_ko", "snapshot_date", "rank", "wins", "losses", "draws", "games_behind",
            "played", "win_rate", "winning_streak", "batting_average", "era", "last_ten", "source", "collected_at", "source_fetched_at", "last_synced_at",
        ), limit)
        for row in rows:
            row["detailPath"] = f"/standings/teams/{tving_team_code(row['team__team_code'])}"
        details = []
        if include_detail:
            codes = [tving_team_code(team_code)] if team_code else [tving_team_code(row["team__team_code"]) for row in rows]
            for code in codes:
                profile = TeamProfile.objects.filter(external_code=code).select_related("team").first()
                detail = read_team(code) if profile else None
                entry = {"teamCode": code, "data": detail, "detailPath": f"/standings/teams/{code}",
                         "source": "tving", "sourceUrl": f"https://www.tving.com/sports/kbo/team/{code}",
                         "collected_at": _json(team_sync_time(profile.team)) if profile else None,
                         "detail_available": detail is not None, "time_basis": "current_saved_detail_not_requested_snapshot",
                         "schedule_scope": "earliest_30_saved_games_not_upcoming; use get_games"}
                match = next((row for row in rows if tving_team_code(row["team__team_code"]) == code), None)
                if match is not None:
                    match["teamDetail"] = entry
                else:
                    details.append(entry)
        return _result(rows, standings=[{**row, "team_name_ko": row["team__team_name_ko"]} for row in rows], requested_date=_json(requested), actual_date=_json(actual) if rows else None, team_details=details, **freshness)

    def get_games(start_date=None, end_date=None, team_code=None, stadium_id=None, limit=20, upcoming_only=False, as_of=None, offset=0,
                  date_from=None, date_to=None, team=None, stadium=None, status="all", home_away="all"):
        """날짜 범위의 일정과 결과를 팀/구장으로 필터링한다."""
        from .assistant import team_code as resolve_team, to_stadium_code, STATUS
        upcoming_only = upcoming_only or status == "upcoming"
        start_date, end_date = _game_range(start_date or date_from, end_date or date_to, upcoming_only)
        code = db_team_code(team_code) if team_code else resolve_team(team)
        if team and not code:
            raise ToolException("팀 이름을 확인해 주세요.")
        query = Game.objects.filter(game_date__range=(start_date, end_date))
        if code:
            if home_away == "home":
                query = query.filter(home_team__team_code=code)
            elif home_away == "away":
                query = query.filter(away_team__team_code=code)
            else:
                query = query.filter(Q(home_team__team_code=code) | Q(away_team__team_code=code))
        if stadium:
            stadium_code = to_stadium_code(stadium)
            if not stadium_code:
                raise ToolException("구장 이름을 확인해 주세요.")
            query = query.filter(stadium__stadium_code=stadium_code)
        if status in STATUS:
            states = {"upcoming": ("PREV", "READY", "scheduled"), "finished": ("END", "final"), "canceled": ("CANCEL", "cancelled")}
            query = query.filter(status_code__in=states[status])
        if stadium_id is not None:
            query = query.filter(stadium_id=stadium_id)
        from django.utils import timezone
        cutoff = (as_of or timezone.now()).astimezone(ZoneInfo("Asia/Seoul"))
        # Stored times, including midnight, are exact; no unknown precision is recorded.
        if upcoming_only:
            query = query.filter(status_code__in=("scheduled", "PREV", "READY")).filter(
                Q(game_date__gt=cutoff.date()) | Q(game_date=cutoff.date(), game_time__gt=cutoff.time().replace(tzinfo=None))
            )
        # 다음 경기 날짜까지만 최신성을 확인한 뒤 최종 행을 읽는다.
        checked_end = (query.order_by("game_date", "game_time", "game_code").values_list("game_date", flat=True).first() or end_date) if upcoming_only else end_date
        try:
            freshness = tving_service.get_game_range_freshness(start_date, checked_end)
        except tving_service.TvingError:
            freshness = {"stale": True, "warning": "최신 일정을 확인하지 못해 저장된 자료만 조회합니다."}
        total = query.count()
        rows = _rows(query.order_by("game_date", "game_time", "game_code")[offset:], (
            "id", "game_code", "game_date", "game_time", "home_team__team_code", "home_team__team_name_ko",
            "away_team__team_code", "away_team__team_name_ko", "stadium_id", "stadium__stadium_name_ko", "stadium__stadium_code",
            "home_score", "away_score", "status_code", "game_type", "home_starting_pitcher", "away_starting_pitcher",
            "source", "source_external_code", "source_stadium_name", "source_status_label", "source_home_code", "source_home_name",
            "source_away_code", "source_away_name", "collected_at", "source_fetched_at", "last_synced_at",
        ), limit)
        for row in rows:
            row["time_precision"] = "time"
        games = [{**row, "home_team": row["home_team__team_name_ko"], "away_team": row["away_team__team_name_ko"],
                  "stadium": row["stadium__stadium_name_ko"]} for row in rows]
        return _result(rows, games=games, count_is_partial=offset + len(rows) < total,
                       total_count=total, offset=offset, has_more=offset + len(rows) < total,
                       start_date=start_date.isoformat(), end_date=end_date.isoformat(), limit=limit,
                       availability="unknown" if not total and freshness.get("stale") else "available" if total else "empty",
                       as_of=cutoff.isoformat() if upcoming_only else _json(as_of), **freshness)

    def search_players(team_code=None, player_code=None, name=None, limit=20, include_detail=None, offset=0):
        """저장된 선수 명단과 선택한 프로필·시즌·통산 기록을 조회한다(자동 수집 없음)."""
        stale, warning = False, None
        team_code = tving_team_code(team_code) if team_code else None  # TVING 검색은 약어 기준
        try:
            teams = [team_code] if team_code else []
            if name and not teams:
                from baseball.models import Player
                teams = [tving_team_code(code) for code in Player.objects.filter(name__icontains=name).values_list("team__team_code", flat=True).distinct()]
            for code in teams:
                refreshed = tving_service.refresh_team(code)
                stale, warning = stale or refreshed["stale"], warning or refreshed["warning"]
            if player_code:
                refreshed = tving_service.refresh_athlete(player_code)
                stale, warning = stale or refreshed["stale"], warning or refreshed["warning"]
        except tving_service.TvingError:
            stale, warning = True, "최신 선수 정보를 확인하지 못해 저장된 자료만 조회합니다."
        rows, total = tving_service.search_entities(kind="player", team=team_code, player=player_code, name=name, page_size=limit, offset=offset)
        want_detail = bool(player_code) if include_detail is None else include_detail
        players = {}
        if want_detail:
            from baseball.models import Player
            from tving.relational import read_athlete
            players = {player.external_code: player for player in Player.objects.filter(external_code__in=[row["externalCode"] for row in rows]).select_related("team").prefetch_related("season_records", "career_records")}
        for row in rows:
            code = row["externalCode"]
            row.update(detailPath=f"/standings/players/{code}", source="tving", sourceUrl=f"https://www.tving.com/sports/kbo/athlete/{code}")
            if want_detail:
                player = players.get(code)
                detail = read_athlete(code, player=player) if player else None
                row.update(detail=detail, detail_available=detail is not None,
                           collected_at=_json(player.detail_last_synced_at) if player else None,
                           profile_collected_at=_json(player.profile_last_synced_at) if player else None)
        return _result(rows, total_count=total, offset=offset, has_more=offset + len(rows) < total, stale=stale, warning=warning)

    specs = (
        (get_standings, 'get_standings', '저장된 순위와 실제 날짜를 반환한다. team_code와 include_detail로 현재 구단 기록·팀내순위·선수단을 조회한다. 상세는 과거 순위 시점이 아니다.', StandingsInput),
        (get_games, 'get_games', '저장된 일정/결과를 조회한다. 날짜 미정은 KST 오늘 포함 7일 모든 상태, 명시 날짜 하나는 그날만, 명시 범위는 그대로 조회한다. 다음/다가오는 경기는 upcoming_only=True 또는 status=upcoming으로 as_of 이후의 예정 경기만 조회하며 날짜 미정은 최대 366일을 탐색한다. limit/offset으로 페이지 조회하고 start_date/end_date/total_count/has_more를 안내한다. stale+저장 행은 경고와 함께 사용하고 availability=unknown은 경기 없음이 아니다. 저장된 시작 시각은 정확한 시각이다.', GamesInput),
        (search_players, 'search_players', '저장된 구단/코드/이름 선수 목록을 페이지 조회하며 자동 수집하지 않는다. 이름으로 선수 소개·프로필·상세 기록을 물으면 include_detail=True로 detail의 프로필·시즌·통산 기록을 함께 조회한다(선수 코드 조회는 기본 포함). items의 imageUrl과 detailPath는 상세 포함 여부와 무관하게 반환되며, 소개에 유용하면 제공된 값을 Markdown 이미지·상세 링크로 사용한다. 없는 값이나 URL은 만들지 않는다.', PlayerInput),
    )
    return tuple(_tool(*spec) for spec in specs)
