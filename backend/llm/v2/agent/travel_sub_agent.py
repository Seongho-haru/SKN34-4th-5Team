"""주변 후보 조사 전문 Agent: 구장 주변 맛집·카페·관광·실내활동 후보 조사."""
from .common import build_agent

TRAVEL_RESEARCH_RULES = """역할: 구장 주변 후보 조사 전문 에이전트.
get_stadium 으로 좌표를 확인한 뒤 식당·카페는 search_places(method=category, category=FD6/CE7), 숙박·산책·실내·편의점은 search_nearby_places, 관광은 search_tourism·search_documents_tool 로 후보를 찾는다.
날씨 조건은 get_weather 로 확인한다. 후보와 보유 데이터 근거만 돌려준다. 외부 후기·메뉴·분위기 추가 리서치·새 출처 검색은 하지 않는다. 보유 근거가 부족한 조건은 미확인으로 메인에 반환한다.
최종 추천·일정은 확정하지 않는다."""

TOOLS = ("get_stadium", "search_places", "search_nearby_places", "search_tourism", "search_documents_tool", "get_weather")


def build(model, tools_by_name):
    return build_agent(model, [tools_by_name[n] for n in TOOLS], TRAVEL_RESEARCH_RULES)
