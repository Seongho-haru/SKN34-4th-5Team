"""Offline specialist delegation/allowlist contract; no paid calls."""
import os
import unittest
from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from llm.v2.tests import test_chain
from llm.v2.tests.test_chain import call, decision
from llm.v2.agent import travel_sub_agent, web_sub_agent


def transport_cases():
    import asyncio
    import httpcore
    import httpx
    import anyio
    from unittest.mock import AsyncMock

    async def mapped_error(kind):
        transport = httpx.AsyncHTTPTransport()
        with patch.object(transport._pool, "handle_async_request",
                          AsyncMock(side_effect=kind("offline transport"))):
            async with httpx.AsyncClient(transport=transport) as client:
                try:
                    await client.get("https://example.com/article")
                except httpx.TransportError as error:
                    assert isinstance(error.__cause__, kind)
                    return error
        raise AssertionError("installed HTTP mapping did not raise")

    from mcp.shared.exceptions import McpError
    from mcp.types import ErrorData, CONNECTION_CLOSED
    denied = httpx.ConnectError("denied")
    denied.__cause__ = PermissionError("denied")
    request = httpx.Request("GET", "https://example.com/article")
    cases = [(TimeoutError(), True), (httpx.ReadTimeout("slow"), True),
             (httpx.ConnectError("temporary"), True), (anyio.ClosedResourceError(), True),
             (httpx.UnsupportedProtocol("unsupported"), False), (httpx.LocalProtocolError("invalid"), False),
             (denied, False), (PermissionError(), False), (ValueError(), False),
             (ExceptionGroup("temporary", [TimeoutError(), anyio.EndOfStream()]), True),
             (ExceptionGroup("mixed", [TimeoutError(), httpx.UnsupportedProtocol("unsupported")]), False)]
    for kind in (httpcore.ReadTimeout, httpcore.ConnectError, httpcore.RemoteProtocolError,
                 httpcore.UnsupportedProtocol, httpcore.LocalProtocolError):
        cases.append((asyncio.run(mapped_error(kind)), kind not in
                      {httpcore.UnsupportedProtocol, httpcore.LocalProtocolError}))
    for field in ("__cause__", "__context__"):
        group = ExceptionGroup("permanent root", [TimeoutError(), httpx.ReadTimeout("slow")])
        setattr(group, field, PermissionError("denied"))
        cases.append((group, False))
    for cause in (RuntimeError("unknown cause"), asyncio.CancelledError()):
        error = httpx.ReadTimeout("slow")
        error.__cause__ = cause
        cases.append((error, False))
    for code in (400, 401, 403, 429, 500, 501, 502, 503, 504):
        cases.append((httpx.HTTPStatusError(str(code), request=request,
                      response=httpx.Response(code, request=request)), code in {500, 502, 503, 504}))
    for code in (408, CONNECTION_CLOSED, 503, 403, -32601):
        cases.append((McpError(ErrorData(code=code, message=str(code))), code in {408, CONNECTION_CLOSED, 503}))
    return cases


class BrowserResearchTest(unittest.TestCase):
    run_graph = test_chain.ChainTest.run_graph
    def test_nearby_evidence_returns_to_main(self):
        with patch.dict(os.environ, {"WEB_RESEARCH_ENABLED": "false"}):
            out = self.run_graph([
                call("ask_travel_research", {"task": "Find food evidence"}, "research"),
                AIMessage("후보 근거: https://example.com/menu; 영업시간 미확인"),
                AIMessage("메인 추천: 후보와 https://example.com/menu 출처, 영업시간 미확인"),
            ], decision(capabilities=["nearby_places"]), [HumanMessage("잠실 맛집 추천")])
        evidence = next(m for m in out["messages"] if isinstance(m, ToolMessage))
        self.assertIn("https://example.com/menu", evidence.content)
        self.assertIsInstance(evidence.artifact, list)
        self.assertTrue(out["messages"][-1].content.startswith("메인 추천"))

    def test_specialist_is_available_without_its_capability(self):
        out = self.run_graph([
            call("ask_travel_research", {"task": "후보 확인"}, "research"),
            AIMessage("후보 없음"), AIMessage("메인 안내"),
        ], decision(capabilities=["schedule"]), [HumanMessage("경기 일정")])
        result = next(m for m in out["messages"] if isinstance(m, ToolMessage))
        self.assertEqual(result.status, "success")
        self.assertEqual(result.content, "후보 없음")
        self.assertEqual(set(self.model_calls[1]["tools"]), set(travel_sub_agent.TOOLS))
        self.assertEqual(self.executed, [])

    def test_browser_without_token_fails_before_connecting(self):
        import asyncio
        from llm.v2.agent.browser_research import browse
        with patch.dict(os.environ, {"JEV_MCP_TOKEN": ""}):
            with self.assertRaisesRegex(RuntimeError, "JEV_MCP_TOKEN"):
                asyncio.run(browse("https://example.com", "read"))

    def test_research_model_params_satisfy_meter_output_cap(self):
        from django.conf import settings
        from llm.service.usage import Meter, UnsafeOutputLimit
        from llm.v2.agent.browser_research import research_model
        research_model.cache_clear()
        self.addCleanup(research_model.cache_clear)
        with patch.dict(os.environ, {"RESEARCH_API_KEY": "offline-test-key"}):
            params = research_model()._get_invocation_params()
        self.assertEqual(params["max_completion_tokens"], settings.USAGE_MAX_CALL_OUTPUT_TOKENS)
        meter = Meter()
        meter.on_chat_model_start({}, [[]], run_id="research", invocation_params=params)
        self.assertEqual(meter.totals(), (0, 0, 1, 1))
        uncapped = {k: v for k, v in params.items()
                    if k not in ("max_tokens", "max_completion_tokens", "max_output_tokens")}
        with self.assertRaises(UnsafeOutputLimit):
            meter.on_chat_model_start({}, [[]], invocation_params=uncapped)

    def test_v2_direct_and_specialist_search_is_blocked_and_context_restored(self):
        from langchain_core.tools import ToolException
        from llm.v2.agent.browser_research import allow_new_source_search, structured_search
        from llm.v2.middleware.dynamic_tools import DynamicToolMiddleware
        from types import SimpleNamespace
        middleware = DynamicToolMiddleware(["search_places"])
        request = SimpleNamespace(tool_call={"name": "search_places", "id": "lookup", "args": {}}, state={})
        def handler(_):
            self.assertFalse(allow_new_source_search.get())
            with self.assertRaisesRegex(ToolException, "추가 리서치"):
                structured_search(model="never-called")
            return "existing lookup"
        self.assertEqual(middleware.wrap_tool_call(request, handler), "existing lookup")
        self.assertTrue(allow_new_source_search.get())
        with self.assertRaises(RuntimeError):
            middleware.wrap_tool_call(request, lambda _: (_ for _ in ()).throw(RuntimeError()))
        self.assertTrue(allow_new_source_search.get())

    def test_body_retries_only_transient_transport_once_not_blocked_or_empty(self):
        from unittest.mock import AsyncMock
        from llm.v2.agent import browser_research as browser
        from llm.v2.tests.test_web_specialist import observed
        url = "https://example.com/article"
        with patch.object(
                browser, "browse", AsyncMock(side_effect=[TimeoutError(), observed("body")])) as fetch:
            self.assertEqual(browser.web_body(url)["status"], "ok")
            self.assertEqual(fetch.call_count, 2)
        with patch.object(
                browser, "browse", AsyncMock(side_effect=TimeoutError())) as fetch:
            self.assertEqual(browser.web_body(url)["status"], "timeout")
            self.assertEqual(fetch.call_count, 2)
        for value in ({"status": "blocked"}, observed("")):
            with patch.object(browser, "browse", AsyncMock(return_value=value)) as fetch:
                self.assertNotEqual(browser.web_body(url)["status"], "ok")
                self.assertEqual(fetch.call_count, 1)

    def test_actual_transport_hierarchy_controls_body_attempts(self):
        from unittest.mock import AsyncMock
        from llm.v2.agent import browser_research as browser
        from llm.v2.tests.test_web_specialist import observed
        for error, retryable in transport_cases():
            with self.subTest(error=repr(error)), patch.object(browser, "browse",
                    AsyncMock(side_effect=[error, observed("body")])) as fetch:
                if retryable:
                    self.assertEqual(browser.web_body("https://example.com/article")["status"], "ok")
                else:
                    with self.assertRaises(type(error)):
                        browser.web_body("https://example.com/article")
                self.assertEqual(fetch.call_count, 2 if retryable else 1)

    def test_completed_task_cancellation_retrieves_exception(self):
        import asyncio
        import gc
        from llm.v2.agent import browser_research as browser
        from llm.service.chat_runs import Stopped
        async def run():
            warnings = []
            asyncio.get_running_loop().set_exception_handler(lambda loop, context: warnings.append(context))
            async def failed():
                raise TimeoutError()
            with patch("llm.service.chat_runs.check_cancelled", side_effect=Stopped), self.assertRaises(Stopped):
                await browser.cancellable_call(failed())
            gc.collect()
            await asyncio.sleep(0)
            self.assertEqual(warnings, [])
        asyncio.run(run())

    def test_body_cancellation_prevents_retry(self):
        from unittest.mock import AsyncMock
        from llm.v2.agent import browser_research as browser
        from llm.service.chat_runs import Stopped
        with patch.object(browser, "browse", AsyncMock(side_effect=TimeoutError())) as fetch, patch(
                "llm.service.chat_runs.check_cancelled", side_effect=[None, Stopped]), self.assertRaises(Stopped):
            browser.web_body("https://example.com/article")
        self.assertEqual(fetch.call_count, 1)
        self.assertFalse(browser._admission.locked())

    def test_specialist_cannot_repeat_failed_body_read(self):
        from llm.v2.middleware.dynamic_tools import DynamicToolMiddleware
        from types import SimpleNamespace
        args = {"url": "https://example.com/article"}
        messages = [HumanMessage("read"), call("jev_read_body", args, "first"),
                    ToolMessage("blocked", tool_call_id="first", name="jev_read_body", status="error")]
        request = SimpleNamespace(tool_call={"name": "jev_read_body", "id": "repeat", "args": args},
                                  state={"messages": messages})
        from unittest.mock import Mock
        handler = Mock()
        result = DynamicToolMiddleware(["jev_read_body"]).wrap_tool_call(request, handler)
        self.assertEqual(result.status, "error")
        handler.assert_not_called()
        request.tool_call["args"] = {"url": "https://example.com/other-known-page"}
        DynamicToolMiddleware(["jev_read_body"]).wrap_tool_call(request, handler)
        handler.assert_called_once()

    def test_policy_instructions_are_present_not_semantic_enforcement(self):
        from llm.v2.middleware.jev_guidelines import GUARD_INSTRUCTIONS, GROUNDING_RULES, CONTENT_RULES
        from llm.v2.agent.chain import MAIN_RULES
        self.assertIn("전체 NON_PASS", GUARD_INSTRUCTIONS)
        self.assertIn("확인 질문만", GROUNDING_RULES)
        self.assertIn("전체 일정 조회", GROUNDING_RULES)
        self.assertIn("관련 대화의 확정 조건 → 화면 기본값", CONTENT_RULES)
        self.assertIn("기존 날짜를 폐기하지", MAIN_RULES)
        self.assertIn("개인정보·인증 정보", GROUNDING_RULES)

    def test_enabled_specialist_has_body_read_only_no_native_search(self):
        from types import SimpleNamespace
        model = object()
        with patch.dict(os.environ, {"WEB_RESEARCH_ENABLED": "true"}), patch.object(
                web_sub_agent, "research_model", return_value=model), patch.object(
                web_sub_agent, "direct_tools", return_value=[SimpleNamespace(name="jev_browse"), SimpleNamespace(name="jev_read_body")]), patch.object(
                web_sub_agent, "build_agent") as build:
            web_sub_agent.build(object(), {})
        self.assertEqual([t.name for t in build.call_args.args[1]], ["jev_read_body"])
        self.assertFalse(hasattr(web_sub_agent, "WebSearchMiddleware"))

    def test_enabled_specialist_uses_research_model_and_tool(self):
        from llm.v2.tests.test_chain import ScriptedModel, fake_tools
        model = ScriptedModel(script=[], calls=[])
        with patch.dict(os.environ, {"WEB_RESEARCH_ENABLED": "true"}), patch.object(web_sub_agent, "research_model", return_value=model), patch.object(web_sub_agent, "direct_tools", return_value=[]), patch.object(web_sub_agent, "build_agent") as build:
            web_sub_agent.build(object(), fake_tools([]))
        self.assertIs(build.call_args.args[0], model)
        self.assertEqual(build.call_args.args[1], [])
