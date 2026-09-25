"""모든 답변 에이전트가 공유하는 도구 목록과 bounded tool 실행."""

import contextvars
import json

from langchain_core.messages import ToolMessage

from ..progress import config_kwargs

SUPPORTED_DOMAINS = frozenset({"assistant", "chat", "club", "venue", "course", "nearby"})
_ACTIVE_DOMAIN = contextvars.ContextVar("active_answer_domain", default=None)


def active_domain():
    return _ACTIVE_DOMAIN.get()


def _all_tools():
    from ..tools import create_default_tools
    from .assistant.tools import build_specialized_tools
    from .venue.agent import search_documents_tool

    registered = {tool.name: tool for tool in build_specialized_tools()}
    for tool in (*create_default_tools(), search_documents_tool):
        registered.setdefault(tool.name, tool)
    return tuple(registered.values())


def visible_text(content):
    """Responses API content에서 사용자에게 보여도 되는 텍스트 블록만 합친다."""
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") for part in content or []
        if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
        and isinstance(part.get("text"), str)
    )


def tools_for(domain):
    if domain not in SUPPORTED_DOMAINS:
        raise KeyError(domain)
    return _all_tools()


def invoke(domain, name, arguments):
    if domain not in SUPPORTED_DOMAINS:
        raise KeyError(domain)
    from ..tools import create_default_tools
    tool = next((tool for tool in create_default_tools() if tool.name == name), None)
    if tool is None:
        raise ValueError(f"{name} is not allowed for {domain}")
    return tool.invoke(arguments, **config_kwargs())


def run_model(model, messages, domain, max_tool_rounds=2, require_first_tool=False):
    """전체 지원 도구를 노출하고, 도구 결과를 다음 모델 호출에 넣어 유한 횟수로 답한다."""
    tools = tools_for(domain)
    allowed = {tool.name: tool for tool in tools}
    bound = model.bind_tools(tools)
    first = model.bind_tools(tools, tool_choice="required") if require_first_tool else bound
    conversation = list(messages)
    token = _ACTIVE_DOMAIN.set(domain)
    try:
        for round_number in range(max_tool_rounds):
            response = (first if round_number == 0 else bound).invoke(
                conversation, **config_kwargs()
            )
            conversation.append(response)
            if not response.tool_calls:
                return response
            for call in response.tool_calls:
                tool = allowed.get(call["name"])
                result = tool.invoke(call["args"], **config_kwargs()) if tool else "허용되지 않은 도구입니다."
                conversation.append(ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=call["id"], name=call["name"],
                ))
        return model.invoke(conversation, **config_kwargs())  # 도구 라운드 소진 뒤 답변만 받는 tool-free finalization
    finally:
        _ACTIVE_DOMAIN.reset(token)
