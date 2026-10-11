"""하위 Agent 를 메인 Agent 의 ask_* 도구로 감싼다. 결과 content 는 최종 답(문자열), artifact 는 하위 대화 전체."""
from typing import Annotated

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables.config import merge_configs

from . import baseball_sub_agent, course_sub_agent, place_sub_agent, travel_sub_agent, web_sub_agent
from .common import final_text

SPECIALISTS = {
    "ask_web_research": (web_sub_agent, "정해진 공개 페이지 원문 확인 전문 에이전트: 기존·사용자 제공 URL의 사실·출처·미확인 조건을 반환한다. 키워드 검색·새 출처 발굴은 하지 않는다. 본문 실패 시 필요한 사용자 입력을 메인에 반환한다."),
    "ask_course": (course_sub_agent, "직관 코스 전문 에이전트: 실제 사용자 요청과 현재 지도·서버 기억으로 코스 생성·수정을 검증해 돌려준다. 턴당 한 번만 실행한다."),
    "ask_baseball": (baseball_sub_agent, "야구 전문 에이전트: 경기 일정·시각, 구장 확인, 구장 안 정보, 야구 커뮤니티를 조회해 돌려준다."),
    "ask_travel_research": (travel_sub_agent, "주변 후보 조사 에이전트: 구장 주변 맛집·카페·관광·실내활동 후보를 찾아 돌려준다."),
    "ask_place_data": (place_sub_agent, "장소 확인 에이전트: 기존 공개 코스와 특정 장소가 실제 검색되는지 확인해 돌려준다."),
}


def _delegate(name, description, agent):
    @tool(name, description=description)
    def ask(task: str, runtime: ToolRuntime, summary: Annotated[str, "사용자 화면에 공개되는 40자 이내 짧은 작업명(예: 두산 다음 경기 확인). 비밀·개인정보·식별자·긴 지시 금지"] = ""):
        """task: 하위 Agent 에게 줄 실제 작업 지시.
        summary: 사용자 화면에 보일 40자 이내 짧은 작업명(예: 두산 다음 경기 확인). 비밀·개인정보·긴 지시 금지. 수행 입력에는 쓰이지 않는다."""
        # 하위 Agent 에는 작업 문자열과 선택 context 만 넘긴다. 승인(decision)은 작업 문자열이 아닌 내부 state 로 전달.
        course = name == "ask_course"
        messages = list(runtime.state.get("messages") or []) if course else [HumanMessage(task)]
        records = [] if course else list(messages)
        try:
            for update in agent.stream({"messages": messages, "context": runtime.state.get("context"),
                                        "decision": runtime.state.get("decision"),
                                        "tool_group_ids": list(runtime.state.get("tool_group_ids") or ()),
                                        **({"course_memory": runtime.state.get("course_memory", {})} if course else {})}, stream_mode="updates",
                                       config=merge_configs(runtime.config, {"metadata": {"parent_id": runtime.tool_call_id}})):
                for node in update.values():
                    records.extend((node or {}).get("messages") or [] if isinstance(node, dict) else ())
        except Exception as exc:  # 한도 초과/조회 실패를 결과 없음과 구분해 메인 Agent 에 알린다
            return ToolMessage(f"[조회 실패] {name}: {type(exc).__name__}", name=name,
                               tool_call_id=runtime.tool_call_id, status="error", artifact=records)
        if course:
            result = next((m for m in reversed(records) if isinstance(m, ToolMessage)), None)
            return ToolMessage(str(result.content) if result is not None else "[조회 실패] ask_course: 답변 없음",
                               name=name, tool_call_id=runtime.tool_call_id,
                               status=result.status if result is not None else "error", artifact=records)
        return ToolMessage(final_text({"messages": records}) or f"[조회 실패] {name}: 답변 없음",
                           name=name, tool_call_id=runtime.tool_call_id, artifact=records)
    return ask


def build(model, tools_by_name):
    return [_delegate(name, desc, module.build(model, tools_by_name))
            for name, (module, desc) in SPECIALISTS.items()]
