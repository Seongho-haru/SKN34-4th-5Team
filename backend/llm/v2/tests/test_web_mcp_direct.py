"""Real create_agent loop with the installed MCP conversion boundary, offline."""
import asyncio
import json
import os
import threading
from unittest.mock import patch

from django.test import SimpleTestCase
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
from mcp.types import Tool, CallToolResult, TextContent

from llm.v2.agent import browser_research, chain, web_sub_agent
from llm.v2.tests.test_chain import ScriptedModel, fake_tools, call


class DirectMCPTests(SimpleTestCase):
    def setUp(self):
        self.executed = []
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()
        self.loops = []

    async def boundary(self, request, handler):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            self.loops.append(asyncio.get_running_loop())
            await asyncio.sleep(.02)
            self.executed.append((request.name, request.args))
            payload = ({"source_url": request.args["url"], "result": {"status": "blocked"}}
                       if "denied" in request.args["url"] else
                       {"status": "ok", "body": "verified menu", "source_url": request.args["url"],
                        "source_kind": "rendered_dom_snapshot"})
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(payload))],
                                  structuredContent=payload)
        finally:
            with self.lock:
                self.active -= 1

    def discovered(self, interceptor=None):
        tools = []
        for name in ("jev_browse", "jev_read_body"):
            props = {"url": {"type": "string"}}
            if name == "jev_browse":
                props["goal"] = {"type": "string"}
            tools.append(convert_mcp_tool_to_langchain_tool(None,
                Tool(name=name, description="Actual MCP schema", inputSchema={"type": "object",
                     "properties": props, "required": list(props)}),
                connection={"transport": "streamable_http", "url": "http://offline.test/mcp"},
                tool_interceptors=[interceptor or self.boundary]))
        return tools

    def graph(self, script, capabilities=("web_research",)):
        calls = []
        model = ScriptedModel(script=script, calls=calls)
        with patch.dict(os.environ, {"WEB_RESEARCH_ENABLED": "true"}), patch.object(
                browser_research, "discover_tools", side_effect=self.async_discover), patch.object(
                web_sub_agent, "research_model", return_value=model), patch(
                "llm.v2.middleware.jev_guidelines.classify", return_value={"allowed": True, "capabilities": list(capabilities)}):
            out = chain.build_graph(model, fake_tools([])).invoke({"messages": [HumanMessage("메뉴 근거 확인", id="q")]})
        return out, calls

    async def async_discover(self):
        return self.discovered()

    def test_results_drive_alternative_url_then_main_and_history(self):
        def assess(messages):
            result = messages[-1]
            if isinstance(result, ToolMessage) and result.name == "jev_read_body":
                self.assertIn("verified menu", str(result.content))
                return None
            self.assertIsInstance(result, ToolMessage)
            self.assertEqual(result.status, "error")
            self.assertIn("blocked", str(result.content))
            return call("jev_read_body", {"url": "https://example.com/alternative"}, "body")
        out, calls = self.graph([
            call("ask_web_research", {"task": "메뉴 확인", "summary": "메뉴 근거"}, "web"),
            call("jev_browse", {"url": "https://example.com/denied", "goal": "menu"}, "browse"),
            assess, AIMessage("확인 메뉴 https://example.com/alternative; 원래 URL 차단"), AIMessage("메인 최종 안내")])
        self.assertEqual([n for n, _ in self.executed], ["jev_browse", "jev_read_body"])
        self.assertEqual(self.peak, 1)
        self.assertEqual(len(set(self.loops)), 2)
        self.assertNotIn("web_search", calls[0]["tools"])
        self.assertIn("jev_browse", calls[1]["tools"])
        self.assertIn("jev_read_body", calls[1]["tools"])
        self.assertIn("web_search", calls[1]["tools"])
        self.assertNotIn("research_public_web", calls[1]["tools"])
        parent = next(m for m in out["messages"] if isinstance(m, ToolMessage))
        self.assertTrue(any(isinstance(m, ToolMessage) and m.name == "jev_read_body" and m.artifact for m in parent.artifact))
        from llm.serializer.message import project_history, project_event
        out["messages"][-1].id = "answer"
        history = project_history(out["messages"], {"q": {"status": "completed", "answer_id": "answer"}}, detail=True)
        tools = history[-1]["tools"]
        self.assertEqual(tools[0]["kind"], "sub_agent")
        self.assertEqual(tools[0]["summary"], "메뉴 근거")
        self.assertEqual(tools[1]["parent_id"], "web")
        live = project_event({"event": "on_tool_end", "name": "ask_web_research", "data": {"output": parent}})
        self.assertEqual(live[1]["kind"], "sub_agent")
        self.assertEqual(out["messages"][-1].text, "메인 최종 안내")

    def test_parallel_batch_is_sequential_and_total_calls_bounded(self):
        batch = AIMessage("", tool_calls=[{"name": "jev_read_body", "args": {"url": f"https://example.com/{i}"}, "id": str(i)} for i in range(5)])
        out, _ = self.graph([call("ask_web_research", {"task": "확인"}, "web"), batch,
                             AIMessage("호출 한도 이후 미확인"), AIMessage("메인")])
        self.assertEqual(len(self.executed), 3)
        self.assertEqual(self.peak, 1)
        parent = next(m for m in out["messages"] if isinstance(m, ToolMessage))
        self.assertEqual(sum(isinstance(m, ToolMessage) and m.status == "error" for m in parent.artifact), 2)

    def test_empty_capabilities_still_delegate_to_narrow_web_tools(self):
        out, calls = self.graph([call("ask_web_research", {"task": "메뉴 확인"}, "web"),
                                 call("jev_read_body", {"url": "https://example.com/menu"}, "body"),
                                 AIMessage("메뉴 확인"), AIMessage("메인")], ())
        self.assertEqual(set(calls[0]["tools"]), set(chain.sub_agents.SPECIALISTS) | {"present_planning_questions"})
        self.assertEqual(set(calls[1]["tools"]), {"jev_browse", "jev_read_body", "web_search"})
        self.assertEqual(self.executed, [("jev_read_body", {"url": "https://example.com/menu"})])
        self.assertEqual(next(m for m in out["messages"] if isinstance(m, ToolMessage)).status, "success")

    def test_adapter_busy_private_overflow_timeout_and_cancellation(self):
        from llm.service.attachments import MAX_TEXT
        from llm.service.chat_runs import Stopped
        with patch.object(browser_research, "discover_tools", side_effect=self.async_discover):
            tools = browser_research.direct_tools()
        body = next(t for t in tools if t.name == "jev_read_body")
        tool_call = {"name": body.name, "id": "body", "type": "tool_call", "args": {"url": "https://example.com/"}}
        browser_research._admission.acquire()
        try:
            self.assertEqual(body.invoke(tool_call).status, "error")
        finally:
            browser_research._admission.release()
        with self.assertRaises(Exception):
            body.invoke({**tool_call, "args": {"url": "http://127.0.0.1/"}})
        self.assertEqual(self.executed, [])
        async def oversized(request, handler):
            return CallToolResult(content=[TextContent(type="text", text=json.dumps({"status": "ok", "body": "x" * MAX_TEXT}))])
        async def timeout(request, handler):
            raise TimeoutError()
        for interceptor, status in ((oversized, "overflow"), (timeout, "timeout")):
            original = self.discovered()[1]
            from unittest.mock import AsyncMock
            async def coroutine(_interceptor=interceptor, **kwargs):
                from types import SimpleNamespace
                result = await _interceptor(SimpleNamespace(args=kwargs), None)
                return [{"type": "text", "text": result.content[0].text}], None
            original = original.model_copy(update={"coroutine": coroutine})
            with patch.object(browser_research, "discover_tools", AsyncMock(return_value=[original])):
                adapted = browser_research.direct_tools()[0]
            result = adapted.invoke(tool_call)
            self.assertEqual(result.status, "error")
            self.assertIn(status, str(result.content))
        async def waiting():
            await asyncio.sleep(10)
        with patch("llm.service.chat_runs.check_cancelled", side_effect=Stopped), self.assertRaises(Stopped):
            asyncio.run(browser_research.cancellable_call(waiting()))

    def test_real_converter_checks_text_and_structured_payloads(self):
        from langchain_mcp_adapters.tools import _convert_call_tool_result
        from llm.service.attachments import MAX_TEXT
        from unittest.mock import AsyncMock
        ok = {"status": "ok"}
        cases = [(ok, {"status": status}, status) for status in
                 ("blocked", "busy", "timeout", "error", "partial", "overflow", "cancelled")]
        cases += [({"status": "blocked"}, ok, "blocked"),
                  (ok, {"status": "ok", "result": {"status": "blocked"}}, "blocked"),
                  (ok, {"body": "unverified"}, "error"),
                  (ok, {"status": "ok", "body": "가" * (MAX_TEXT // 3 + 1)}, "overflow"),
                  ({"status": "ok", "body": "x" * MAX_TEXT}, ok, "overflow"),
                  (ok, ok, "ok")]
        for text, structured, expected in cases:
            with self.subTest(expected=expected, structured_status=structured.get("status")):
                payload = CallToolResult(content=[TextContent(type="text", text=json.dumps(text))],
                                         structuredContent=structured)
                _, artifact = _convert_call_tool_result(payload)
                self.assertIsInstance(artifact, dict)
                self.assertEqual(artifact["structured_content"], structured)
                async def boundary(request, handler):
                    return payload
                with patch.object(browser_research, "discover_tools",
                                  AsyncMock(return_value=self.discovered(boundary))):
                    body = browser_research.direct_tools()[1]
                result = body.invoke({"name": body.name, "id": "body", "type": "tool_call",
                                      "args": {"url": "https://example.com/"}})
                self.assertEqual(result.status, "success" if expected == "ok" else "error")
                if expected != "ok":
                    self.assertEqual(json.loads(result.content)["status"], expected)

    def test_nested_status_is_not_invented_success(self):
        for status in ("blocked", "busy", "timeout", "error", "partial", "overflow"):
            self.assertEqual(browser_research.result_status([{"type": "text", "text": json.dumps({"status": "ok", "result": {"status": status}})}]), status)
        self.assertEqual(browser_research.result_status({"result": {"final_text": "invented"}}), "error")
