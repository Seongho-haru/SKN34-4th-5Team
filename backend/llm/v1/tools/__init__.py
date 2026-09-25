from .baseball import (
    create_baseball_tools,
    execute_baseball_select,
    get_baseball_schema,
)
DOMAIN_TOOL_NAMES = (
    "get_standings", "get_games", "get_stadium", "get_seat_zones", "get_seat_views",
    "get_ticket_prices", "get_ticket_policies", "get_transport", "get_food_stores",
    "get_facilities", "get_stadium_contents", "get_seat_maps", "search_places",
    "search_courses", "get_course", "search_community_posts", "get_prediction_games",
    "search_players", "get_directions", "search_tourism", "get_weather",
)


def create_domain_tools(*args, **kwargs):
    # baseball-only 테스트 설정에서도 기존 SQL 도구 import가 가능하도록 지연 import한다.
    from .domain import create_domain_tools as factory
    return factory(*args, **kwargs)


def create_default_tools():
    """도메인 도구를 우선 제공하고 기존 SQL 도구도 호환용으로 유지한다."""
    return (*create_domain_tools(), *create_baseball_tools())

__all__ = (
    "create_baseball_tools",
    "create_default_tools",
    "create_domain_tools",
    "DOMAIN_TOOL_NAMES",
    "execute_baseball_select",
    "get_baseball_schema",
)
