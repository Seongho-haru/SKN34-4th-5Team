"""도메인 조회 도구 모음.

기본 채팅 경로에서 사용한다. ``CHAT_USE_RAG=1``이면 기존 RAG 경로가 먼저
응답하므로 이 도구 루프를 거치지 않는다는 기존 opt-in 동작은 의도적으로 유지한다.
저장된 코스/커뮤니티/외부 제공자 문자열은 명령이 아닌 신뢰하지 않는 데이터다.
외부 조회는 travel/tving의 기존 공개 서비스에만 위임하며 여기서 제공자 호출을 재구현하지 않는다.
"""

import math
from datetime import date
from decimal import Decimal
from uuid import UUID

from django.db import DatabaseError
from django.db.models import Q
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from baseball.models import (
    Facility,
    FoodStore,
    Game,
    HomeContext,
    SeatMap,
    SeatScope,
    SeatZone,
    Stadium,
    StadiumContent,
    StandingHistory,
    TicketPolicy,
    TicketPrice,
    Transport,
)
from community.models import (
    FREE_CATEGORIES,
    TEAM_CATEGORIES,
    TEAM_CODES,
    CommunityPost,
    PredictionGame,
)
from travel.models import Course
from tving import service as tving_service


INVALID_INPUT = "도구 입력 형식이 올바르지 않습니다. 인자 설명을 확인하세요."
DB_ERROR = "저장된 정보를 조회하지 못했습니다. 잠시 후 다시 시도해 주세요."
KAKAO_CATEGORIES = ("FD6", "CE7", "AT4", "CT1", "CS2", "AD5")


def _json(value):
    if isinstance(value, (UUID, Decimal)):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _rows(queryset, fields, limit):
    return [{key: _json(value) for key, value in row.items()} for row in queryset.values(*fields)[:limit]]


def _result(items, **metadata):
    return {**metadata, "count": len(items), "items": items}


def _safe(function):
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ToolException:
            raise
        except DatabaseError:
            raise ToolException(DB_ERROR) from None

    wrapped.__name__ = function.__name__
    wrapped.__doc__ = function.__doc__
    wrapped.__annotations__ = function.__annotations__
    return wrapped


def _tool(function, name, description, schema):
    return StructuredTool.from_function(
        _safe(function), name=name, description=description, args_schema=schema,
        handle_tool_error=True, handle_validation_error=INVALID_INPUT,
    )


class ToolInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)


class LimitInput(ToolInput):
    limit: StrictInt = Field(default=20, ge=1, le=100)


class StandingsInput(LimitInput):
    snapshot_date: date | None = None


class GamesInput(LimitInput):
    start_date: date
    end_date: date
    team_code: str | None = Field(default=None, pattern="^[A-Z]{2}$")
    stadium_id: StrictInt | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_range(self):
        if self.start_date > self.end_date or (self.end_date - self.start_date).days > 366:
            raise ValueError("날짜 범위는 순서대로 최대 366일이어야 합니다.")
        if self.team_code and self.team_code not in TEAM_CODES:
            raise ValueError("올바른 팀 코드가 아닙니다.")
        return self


class StadiumInput(ToolInput):
    stadium_id: StrictInt | None = Field(default=None, ge=1)
    stadium_code: str | None = Field(default=None, min_length=1, max_length=40)

    @model_validator(mode="after")
    def exactly_one(self):
        if (self.stadium_id is None) == (self.stadium_code is None):
            raise ValueError("stadium_id와 stadium_code 중 하나만 입력하세요.")
        return self


class ContextInput(LimitInput):
    season: StrictInt = Field(ge=1982, le=2100)
    team_code: str = Field(pattern="^[A-Z]{2}$")
    stadium_id: StrictInt | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_team(self):
        if self.team_code not in TEAM_CODES:
            raise ValueError("올바른 팀 코드가 아닙니다.")
        return self


class TicketPricesInput(ContextInput):
    as_of: date | None = None
    zone_code: str | None = Field(default=None, min_length=1, max_length=80)


class TicketPoliciesInput(LimitInput):
    team_code: str = Field(pattern="^[A-Z]{2}$")
    game_id: StrictInt | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_team(self):
        if self.team_code not in TEAM_CODES:
            raise ValueError("올바른 팀 코드가 아닙니다.")
        return self


class StadiumListInput(LimitInput):
    stadium_id: StrictInt = Field(ge=1)


class FacilityInput(StadiumListInput):
    facility_type: str | None = Field(default=None, min_length=1, max_length=80)


class ContentInput(StadiumListInput):
    content_type: str | None = Field(default=None, min_length=1, max_length=80)


class PlacesInput(ToolInput):
    method: str = Field(pattern="^(keyword|category)$")
    query: str | None = Field(default=None, min_length=1, max_length=100)
    category: str | None = None
    latitude: StrictFloat = Field(ge=-90, le=90)
    longitude: StrictFloat = Field(ge=-180, le=180)
    radius: StrictInt | None = Field(default=None, ge=1, le=20_000)
    page: StrictInt = Field(default=1, ge=1, le=3)
    limit: StrictInt = Field(default=15, ge=1, le=15)
    sort: str = Field(default="distance", pattern="^(accuracy|distance)$")

    @model_validator(mode="after")
    def validate_search(self):
        if not math.isfinite(self.latitude) or not math.isfinite(self.longitude):
            raise ValueError("좌표는 유한한 숫자여야 합니다.")
        if self.category is not None and self.category not in KAKAO_CATEGORIES:
            raise ValueError("허용되지 않은 카테고리입니다.")
        if self.method == "keyword" and not self.query:
            raise ValueError("키워드 검색에는 query가 필요합니다.")
        if self.method == "category" and self.category is None:
            raise ValueError("카테고리 검색에는 category가 필요합니다.")
        return self


class CourseSearchInput(LimitInput):
    query: str | None = Field(default=None, min_length=1, max_length=100)
    stadium: str | None = Field(default=None, min_length=1, max_length=120)
    tag: str | None = Field(default=None, min_length=1, max_length=80)

    @model_validator(mode="after")
    def any_filter(self):
        if not any((self.query, self.stadium, self.tag)):
            raise ValueError("검색 조건이 하나 이상 필요합니다.")
        return self


class CourseInput(ToolInput):
    course_id: UUID


class CommunitySearchInput(LimitInput):
    query: str = Field(min_length=1, max_length=100)
    board: str | None = Field(default=None, pattern="^(free|teams)$")
    team_code: str | None = Field(default=None, pattern="^[A-Z]{2}$")
    category: str | None = None

    @model_validator(mode="after")
    def validate_filters(self):
        if self.team_code and self.team_code not in TEAM_CODES:
            raise ValueError("올바른 팀 코드가 아닙니다.")
        if self.board == "free" and self.team_code:
            raise ValueError("자유게시판에는 팀 필터를 쓸 수 없습니다.")
        allowed_categories = FREE_CATEGORIES if self.board == "free" else TEAM_CATEGORIES
        if self.category and self.category not in allowed_categories:
            raise ValueError("올바른 카테고리가 아닙니다.")
        return self


class PredictionInput(LimitInput):
    game_date: date
    team_code: str | None = Field(default=None, pattern="^[A-Z]{2}$")
    status: str | None = Field(default=None, pattern="^(scheduled|live|final|cancelled|postponed|suspended|unknown)$")

    @model_validator(mode="after")
    def validate_team(self):
        if self.team_code and self.team_code not in TEAM_CODES:
            raise ValueError("올바른 팀 코드가 아닙니다.")
        return self


class PlayerInput(LimitInput):
    team_code: str | None = Field(default=None, pattern="^(SS|KT|LG|HT|OB|NC|HH|LT|SK|WO)$")
    player_code: str | None = Field(default=None, min_length=1, max_length=40)
    name: str | None = Field(default=None, min_length=1, max_length=80)

    @model_validator(mode="after")
    def any_filter(self):
        if not any((self.team_code, self.player_code, self.name)):
            raise ValueError("구단, 선수 코드, 이름 중 하나가 필요합니다.")
        return self


class DirectionsInput(ToolInput):
    mode: str = Field(pattern="^(car|walk|transit)$")
    points: list[dict[str, StrictFloat]] = Field(min_length=2, max_length=13)

    @model_validator(mode="after")
    def validate_points(self):
        for point in self.points:
            if set(point) != {"lat", "lng"} or not math.isfinite(point["lat"]) or not math.isfinite(point["lng"]):
                raise ValueError("각 지점에는 유한한 lat, lng만 있어야 합니다.")
            if abs(point["lat"]) > 90 or abs(point["lng"]) > 180:
                raise ValueError("좌표 범위를 확인하세요.")
        return self


class TourismInput(ToolInput):
    stadium_code: str = Field(pattern="^(JAMSIL|GOCHEOK|MUNHAK|SUWON|DAEJEON|DAEGU|GWANGJU|SAJIK|CHANGWON)$")
    latitude: StrictFloat = Field(ge=-90, le=90)
    longitude: StrictFloat = Field(ge=-180, le=180)


class WeatherInput(ToolInput):
    stadium_code: str = Field(pattern="^(JAMSIL|GOCHEOK|MUNHAK|SUWON|DAEJEON|DAEGU|GWANGJU|SAJIK|CHANGWON)$")
    game_date: date
    game_time: str = Field(pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")


def _context(season, team_code, stadium_id=None):
    query = HomeContext.objects.filter(season=season, team__team_code=team_code)
    return query.filter(stadium_id=stadium_id) if stadium_id is not None else query


def _course_item(course, include_stops=False):
    item = {
        "id": str(course.pk), "route_number": course.route_number, "title": course.title,
        "stadium": course.stadium, "description": course.description, "content": course.content,
        "content_format": course.content_format, "duration": course.duration, "cover": course.cover,
        "tags": course.tags, "start_lat": course.start_lat, "start_lng": course.start_lng,
        "author": course.author, "likes": course.likes, "views": course.views,
        "is_sample": course.is_sample, "created_at": course.created_at.isoformat(),
        "updated_at": course.updated_at.isoformat(),
    }
    if include_stops:
        item["stops"] = list(course.stops.order_by("position").values(
            "position", "name", "lat", "lng", "category", "place_id", "address",
            "tour_content_id", "is_map_point", "is_drawn_point",
        ))
    return item


def create_domain_tools():
    """도메인 에이전트와 기본 채팅이 공유하는 typed 조회 도구를 만든다."""

    def get_standings(snapshot_date=None, limit=20):
        """정확한 날짜 또는 저장된 최신 날짜의 KBO 순위를 조회한다."""
        freshness = tving_service.ensure_standings_fresh(snapshot_date)
        actual = snapshot_date or StandingHistory.objects.order_by("-snapshot_date").values_list("snapshot_date", flat=True).first()
        rows = [] if actual is None else _rows(
            StandingHistory.objects.filter(snapshot_date=actual).order_by("rank", "team__team_code"),
            ("team__team_code", "team__team_name_ko", "snapshot_date", "rank", "wins", "losses", "draws", "games_behind"), limit,
        )
        return _result(rows, requested_date=_json(snapshot_date), actual_date=_json(actual) if rows else None, **freshness)

    def get_games(start_date, end_date, team_code=None, stadium_id=None, limit=20):
        """날짜 범위의 일정과 결과를 팀/구장으로 필터링한다."""
        freshness = tving_service.ensure_game_range_fresh(start_date, end_date)
        query = Game.objects.filter(game_date__range=(start_date, end_date))
        if team_code:
            query = query.filter(Q(home_team__team_code=team_code) | Q(away_team__team_code=team_code))
        if stadium_id is not None:
            query = query.filter(stadium_id=stadium_id)
        return _result(_rows(query.order_by("game_date", "game_time", "game_code"), (
            "id", "game_code", "game_date", "game_time", "home_team__team_code", "home_team__team_name_ko",
            "away_team__team_code", "away_team__team_name_ko", "stadium_id", "stadium__stadium_name_ko",
            "home_score", "away_score", "status_code", "game_type",
        ), limit), **freshness)

    def get_stadium(stadium_id=None, stadium_code=None):
        """ID 또는 코드로 공개 구장 정보를 조회한다."""
        query = Stadium.objects.filter(id=stadium_id) if stadium_id is not None else Stadium.objects.filter(stadium_code=stadium_code)
        item = _rows(query, ("id", "stadium_code", "stadium_name_ko", "address", "longitude", "latitude", "facility_manager", "game_operator", "phone_general", "phone_facility", "phone_ticket"), 1)
        return {"item": item[0] if item else None}

    def get_seat_zones(season, team_code, stadium_id=None, limit=20):
        """팀·시즌 홈 컨텍스트의 좌석 구역을 조회한다."""
        query = SeatZone.objects.filter(home_context__in=_context(season, team_code, stadium_id)).order_by("home_context__stadium_id", "zone_code")
        return _result(_rows(query, ("id", "home_context_id", "home_context__stadium_id", "zone_code", "zone_name_ko", "level", "side", "seat_type", "group_size", "accessible"), limit))

    def get_seat_views(season, team_code, stadium_id=None, limit=20):
        """팀·시즌 홈 컨텍스트의 좌석 시야 특성을 조회한다."""
        query = SeatScope.objects.filter(home_context__in=_context(season, team_code, stadium_id)).order_by("home_context__stadium_id", "scope_code", "seat_views__id")
        return _result(_rows(query, ("id", "home_context_id", "home_context__stadium_id", "scope_code", "scope_name", "seat_views__view_characteristic", "seat_views__roof_coverage", "seat_views__evidence_scope"), limit))

    def get_ticket_prices(season, team_code, stadium_id=None, limit=20, as_of=None, zone_code=None):
        """팀·시즌 좌석 가격을 선택한 유효일 기준으로 조회한다."""
        query = TicketPrice.objects.filter(seat_zone__home_context__in=_context(season, team_code, stadium_id))
        if as_of:
            query = query.filter(Q(valid_from__isnull=True) | Q(valid_from__lte=as_of), Q(valid_to__isnull=True) | Q(valid_to__gte=as_of))
        if zone_code:
            query = query.filter(seat_zone__zone_code=zone_code)
        rows = _rows(query.order_by("seat_zone__zone_code", "price_krw", "id"), ("id", "seat_zone__zone_code", "seat_zone__zone_name_ko", "price_tier", "day_type", "customer_type", "group_size", "price_krw", "valid_from", "valid_to", "discount_condition"), limit)
        return _result(rows, as_of=_json(as_of))

    def get_ticket_policies(team_code, game_id=None, limit=20):
        """팀과 선택한 경기의 공개 예매 정책을 조회한다."""
        query = TicketPolicy.objects.filter(team__team_code=team_code)
        if game_id is not None:
            query = query.filter(Q(game_id=game_id) | Q(game_id__isnull=True))
        return _result(_rows(query.order_by("policy_code", "channel_no", "id"), ("policy_code", "team__team_code", "game_id", "policy_type", "subtype", "open_at", "max_tickets", "channel_no", "booking_channel", "channel_condition"), limit))

    def get_transport(stadium_id, limit=20):
        """구장의 교통·주차 정보를 조회한다."""
        return _result(_rows(Transport.objects.filter(stadium_id=stadium_id).order_by("mode", "access_code"), ("access_code", "mode", "title", "details", "parking_spaces", "reservation_required"), limit))

    def get_food_stores(stadium_id, limit=20):
        """구장 공식 매점과 위치·메뉴 분류를 조회한다."""
        items = []
        for store in FoodStore.objects.filter(stadium_id=stadium_id).prefetch_related("locations", "menus").order_by("record_code")[:limit]:
            items.append({"record_code": store.record_code, "store_facility": store.store_facility, "location_qty": store.location_qty, "locations": list(store.locations.order_by("location_no").values("location_no", "floor", "zone_location")), "menus": list(store.menus.order_by("menu_category_official").values_list("menu_category_official", flat=True))})
        return _result(items)

    def get_facilities(stadium_id, limit=20, facility_type=None):
        """구장의 편의시설을 조회한다."""
        query = Facility.objects.filter(stadium_id=stadium_id)
        if facility_type:
            query = query.filter(facility_type=facility_type)
        return _result(_rows(query.order_by("facility_type", "record_code"), ("record_code", "facility_type", "floor", "side", "nearby_section", "gate", "gender", "indoor_outdoor", "location_detail"), limit))

    def get_stadium_contents(stadium_id, limit=20, content_type=None):
        """구장의 공개 부가 콘텐츠를 조회한다."""
        query = StadiumContent.objects.filter(stadium_id=stadium_id)
        if content_type:
            query = query.filter(content_type=content_type)
        return _result(_rows(query.order_by("content_type", "record_code"), ("record_code", "content_type", "name", "floor", "location", "official_description", "operating_condition"), limit))

    def get_seat_maps(season, team_code, stadium_id=None, limit=20):
        """팀·시즌 홈 컨텍스트의 공식 좌석도와 자산을 조회한다."""
        items = []
        for seat_map in SeatMap.objects.filter(home_context__in=_context(season, team_code, stadium_id)).select_related("home_context").prefetch_related("assets").order_by("home_context__stadium_id", "id")[:limit]:
            items.append({"home_context_id": seat_map.home_context_id, "stadium_id": seat_map.home_context.stadium_id, "map_title": seat_map.map_title, "page_url": seat_map.page_url, "assets": list(seat_map.assets.order_by("asset_no").values("asset_no", "asset_url", "asset_role"))})
        return _result(items)

    def search_places(method, latitude, longitude, query=None, category=None, radius=None, page=1, limit=15, sort="distance"):
        """기존 장소 서비스로 검색하고 그 서비스가 제공자 결과를 동기화한다."""
        try:
            from travel.place_service import PlaceError, search_and_sync_places
        except ModuleNotFoundError as error:
            if error.name != "travel.place_service":
                raise
            raise ToolException("장소 검색 서비스 통합이 필요합니다.") from None
        payload = {
            "method": method, "lat": latitude, "lng": longitude,
            "page": page, "size": limit, "sort": sort,
        }
        if query is not None:
            payload["keyword"] = query
        if category is not None:
            payload["category"] = category
        if radius is not None:
            payload["radius"] = radius
        try:
            return search_and_sync_places(payload)
        except PlaceError as error:
            raise ToolException(error.message) from None

    def search_courses(query=None, stadium=None, tag=None, limit=20):
        """공개 코스를 제목·설명·구장·태그로 검색한다."""
        courses = Course.objects.all()
        if query:
            courses = courses.filter(Q(title__icontains=query) | Q(description__icontains=query) | Q(content__icontains=query))
        if stadium:
            courses = courses.filter(stadium__icontains=stadium)
        if tag:
            courses = courses.filter(tags__contains=[tag])
        items = [_course_item(course) for course in courses.order_by("-created_at", "id")[:limit]]
        return _result(items)

    def get_course(course_id):
        """UUID로 공개 코스와 방문 장소를 조회한다."""
        course = Course.objects.prefetch_related("stops").filter(pk=course_id).first()
        return {"item": _course_item(course, True) if course else None}

    def search_community_posts(query, board=None, team_code=None, category=None, limit=20):
        """공개 커뮤니티 글을 제목·본문으로 검색한다."""
        posts = CommunityPost.objects.filter(Q(title__icontains=query) | Q(content__icontains=query))
        if board:
            posts = posts.filter(board=board)
        if team_code:
            posts = posts.filter(team_code=team_code)
        if category:
            posts = posts.filter(category=category)
        return _result(_rows(posts.order_by("-created_at", "post_number"), ("post_number", "board", "team_code", "author", "title", "content", "category", "created_at", "views", "recommendations", "comment_count", "is_sample"), limit))

    def get_prediction_games(game_date, team_code=None, status=None, limit=20):
        """저장된 승부예측 대상 경기와 익명 팬 투표 집계를 조회한다(개인 선택 제외)."""
        from community.predictions import _counts

        games = PredictionGame.objects.filter(game_date=game_date)
        if team_code:
            games = games.filter(Q(home_team_code=team_code) | Q(away_team_code=team_code))
        if status:
            games = games.filter(status=status)
        items = [{
            "game_id": game.source_id, "date": game.game_date.isoformat(),
            "starts_at": _json(game.starts_at), "stadium": game.stadium,
            "away": {"code": game.away_team_code, "name": game.away_team_name, "score": game.away_score},
            "home": {"code": game.home_team_code, "name": game.home_team_name, "score": game.home_score},
            "status": game.status, "result": game.result or None,
            "locked": game.locked_at is not None, "voided": game.voided_at is not None,
            "source_fetched_at": _json(game.source_fetched_at),
            "fan_votes": _counts(game),
            "fan_vote_notice": "이용자 팬 투표 집계이며 실제 승리 확률이나 경기 결과 예측이 아닙니다.",
        } for game in games.order_by("starts_at", "source_id")[:limit]]
        return _result(items)

    def search_players(team_code=None, player_code=None, name=None, limit=20):
        """TVING 공통 DB-first 경로로 선수 명단/상세를 갱신한 뒤 공개 선수 정보를 찾는다."""
        stale, warning = False, None
        try:
            teams = [team_code] if team_code else []
            if name and not teams:
                saved, _ = tving_service.search_entities(kind="player", page_size=100)
                needle = name.casefold()
                teams = sorted({row["teamCode"] for row in saved if needle in row["name"].casefold()})
            for code in teams:
                refreshed = tving_service.refresh_team(code)
                stale, warning = stale or refreshed["stale"], warning or refreshed["warning"]
            if player_code:
                refreshed = tving_service.refresh_athlete(player_code)
                stale, warning = stale or refreshed["stale"], warning or refreshed["warning"]
        except tving_service.TvingError:
            stale, warning = True, "최신 선수 정보를 확인하지 못해 저장된 자료만 조회합니다."
        rows, _ = tving_service.search_entities(kind="player", team=team_code, player=player_code, page_size=100)
        if name:
            needle = name.casefold()
            rows = [row for row in rows if needle in row["name"].casefold()]
        return _result(rows[:limit], stale=stale, warning=warning)

    def get_directions(mode, points):
        """기존 공개 길찾기 서비스로 선택 지점 사이 경로를 조회한다."""
        from travel.directions_provider import DirectionsError, fetch_directions
        try:
            return fetch_directions(mode, points)
        except DirectionsError as error:
            message = "길찾기 요청이 많아요. 잠시 후 다시 시도해 주세요." if error.status == 429 else "길찾기 정보를 불러오지 못했습니다."
            raise ToolException(message) from None

    def search_tourism(stadium_code, latitude, longitude):
        """기존 한국관광공사 연동 서비스로 구장 주변 관광지를 조회한다."""
        from rest_framework.exceptions import ValidationError
        from travel.tourism_provider import TourismProviderError
        from travel.tourism_service import search_tourism as service
        try:
            return service({"stadium": stadium_code, "lat": latitude, "lng": longitude})
        except ValidationError:
            raise ToolException("관광지 검색 위치를 확인해 주세요.") from None
        except TourismProviderError:
            raise ToolException("관광지 정보를 불러오지 못했습니다.") from None

    def get_weather(stadium_code, game_date, game_time):
        """기존 기상청 서비스로 구장 경기 시각의 단기예보를 조회한다."""
        from travel.weather_service import WeatherError, get_stadium_weather
        try:
            return get_stadium_weather(stadium_code, game_date.isoformat(), game_time)
        except WeatherError as error:
            messages = {
                "invalid_request": "예보 날짜와 구장을 확인해 주세요.",
                "weather_not_configured": "날씨 데이터 연결 설정이 필요합니다.",
                "weather_busy": "날씨 요청이 많아요. 잠시 후 다시 시도해 주세요.",
            }
            raise ToolException(messages.get(error.code, "날씨 정보를 불러오지 못했습니다.")) from None

    specs = (
        (get_standings, "get_standings", "정확한 날짜 또는 최신 저장 스냅샷의 순위와 실제 날짜를 반환한다.", StandingsInput),
        (get_games, "get_games", "날짜 범위의 실제 일정/결과를 팀 또는 구장으로 좁힌다.", GamesInput),
        (get_stadium, "get_stadium", "구장 ID 또는 코드로 공개 상세를 조회한다.", StadiumInput),
        (get_seat_zones, "get_seat_zones", "팀·시즌·선택 구장의 좌석 구역을 조회한다.", ContextInput),
        (get_seat_views, "get_seat_views", "팀·시즌·선택 구장의 좌석 시야를 조회한다.", ContextInput),
        (get_ticket_prices, "get_ticket_prices", "팀·시즌 좌석 가격을 선택 유효일 기준으로 조회한다.", TicketPricesInput),
        (get_ticket_policies, "get_ticket_policies", "팀과 선택 경기의 예매 정책을 조회한다.", TicketPoliciesInput),
        (get_transport, "get_transport", "구장의 교통·주차 정보를 조회한다.", StadiumListInput),
        (get_food_stores, "get_food_stores", "구장 공식 매점, 위치와 메뉴를 조회한다.", StadiumListInput),
        (get_facilities, "get_facilities", "구장 편의시설을 조회한다.", FacilityInput),
        (get_stadium_contents, "get_stadium_contents", "구장 부가 콘텐츠를 조회한다.", ContentInput),
        (get_seat_maps, "get_seat_maps", "팀·시즌·선택 구장의 좌석도를 조회한다.", ContextInput),
        (search_places, "search_places", "기존 장소 서비스로 주변 장소를 검색하고 최신 결과를 동기화한다.", PlacesInput),
        (search_courses, "search_courses", "공개 코스를 검색한다.", CourseSearchInput),
        (get_course, "get_course", "UUID로 공개 코스 상세를 조회한다.", CourseInput),
        (search_community_posts, "search_community_posts", "공개 커뮤니티 글을 검색한다.", CommunitySearchInput),
        (get_prediction_games, "get_prediction_games", "승부예측 대상 경기와 실제 확률이 아닌 익명 팬 투표 집계를 조회한다.", PredictionInput),
        (search_players, "search_players", "TVING DB-first 최신성 경로로 구단/코드/이름에 맞는 선수를 조회한다.", PlayerInput),
        (get_directions, "get_directions", "기존 길찾기 서비스로 2~13개 지점의 경로를 조회한다.", DirectionsInput),
        (search_tourism, "search_tourism", "한국관광공사 연동 서비스로 구장 주변 관광지를 조회한다.", TourismInput),
        (get_weather, "get_weather", "기상청 연동 서비스로 구장 경기 시각의 날씨를 조회한다.", WeatherInput),
    )
    return tuple(_tool(*spec) for spec in specs)
