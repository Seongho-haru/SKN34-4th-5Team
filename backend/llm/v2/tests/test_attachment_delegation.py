"""Offline compiled-main-graph attachment reads: no attachment specialist/provider hop."""
import json
import os
import uuid
from unittest.mock import AsyncMock, patch

from django.test import TransactionTestCase
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rest_framework.exceptions import ValidationError

from llm.models import ChatAttachment, ChatSession
from llm.serializer.message import project_history
from llm.service import attachments, chat_v2
from llm.service.chat_runs import Stopped
from llm.v2.agent import browser_research, chain, baseball_sub_agent, course_sub_agent, place_sub_agent, travel_sub_agent
from llm.v2.middleware.attachment_context import DirectContextLimit
from llm.v2.tests.test_chain import ScriptedModel, call, fake_tools
from llm.v2.tests.test_web_specialist import observed


class AttachmentDelegationTests(TransactionTestCase):
    def setUp(self):
        self.session = ChatSession.objects.create(guest=uuid.uuid4())
        self.rows = [ChatAttachment.objects.create(session=self.session, kind="url", name=f"page {i}",
                     source_url=f"https://example.com/page{i}") for i in range(3)]
        self.guard = patch("llm.v2.middleware.jev_guidelines.classify", return_value={"allowed": True, "capabilities": []})
        self.classify = self.guard.start()
        self.addCleanup(self.guard.stop)
        self.env = patch.dict(os.environ, {"WEB_RESEARCH_ENABLED": "false"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def input(self, rows=None):
        return {"messages": [HumanMessage("PRIVATE_QUESTION 잠실 휴게시간?", id="q", additional_kwargs={
            "attachment_ids": [str(r.id) for r in (self.rows if rows is None else rows)]})],
            "attachment_session_id": str(self.session.id)}

    def body(self, url, **kwargs):
        return observed("BEGIN baseball MIDDLE 14:00 ~ 15:00 END", requested_url=url, final_url=url, source_url=url,
                        frames=[{"url": url + "/iframe", "role": "article", "status": "ok", "title": "Article frame", "chars": 43}])

    def graph(self, script=None):
        self.calls = []
        return chain.build_graph(ScriptedModel(script=script or [AIMessage("MAIN_FINAL")], calls=self.calls), fake_tools([]))

    def test_forced_main_tool_and_iframe_evidence_without_specialist_model(self):
        from types import SimpleNamespace
        reader = AsyncMock(side_effect=lambda args: self.body(args["url"]))
        unused = AsyncMock(side_effect=AssertionError("attachment must not use jev_browse"))
        with patch.object(browser_research, "discover_tools", AsyncMock(return_value=[
                SimpleNamespace(name="jev_read_body", ainvoke=reader),
                SimpleNamespace(name="jev_browse", ainvoke=unused)])):
            out = self.graph().invoke(self.input())
        self.classify.assert_called_once()
        self.assertEqual(reader.call_count, 3)
        self.assertEqual(reader.call_args_list[0].args, ({"url": self.rows[0].source_url},))
        unused.assert_not_called()
        self.assertEqual(len(self.calls), 1)
        forced = next(m for m in out["messages"] if isinstance(m, AIMessage) and m.tool_calls)
        self.assertEqual(forced.tool_calls[0]["name"], "jev_read_body")
        self.assertEqual(forced.tool_calls[0]["args"], {})
        self.assertFalse(any(m.tool_calls and m.tool_calls[0]["name"] == "ask_web_research"
                             for m in out["messages"] if isinstance(m, AIMessage)))
        self.assertEqual([m.name for m in out["messages"] if isinstance(m, ToolMessage)], ["jev_read_body"])
        current = next(m for m in self.calls[0]["messages"] if m.id == "q")
        for token in ("BEGIN", "MIDDLE 14:00 ~ 15:00", "END", "Article frame", '"original_body": true'):
            self.assertIn(token, current.text)
        self.assertEqual(set(self.calls[0]["tools"]), set(chain.sub_agents.SPECIALISTS) | {"present_planning_questions"})
        self.assertNotIn("BEGIN", json.dumps([m.model_dump() for m in out["messages"]], default=str))

    def test_cache_followup_and_reattach_keep_main_tool_without_browser_fetch(self):
        graph = self.graph([AIMessage("FIRST"), AIMessage("FOLLOWUP"), AIMessage("REATTACH")])
        with patch.object(browser_research, "web_body", side_effect=self.body) as reader:
            first = graph.invoke(self.input(self.rows[:1]))
            second = graph.invoke({**self.input(self.rows[:1]), "messages": [*first["messages"], HumanMessage("followup", id="next")]})
            third = graph.invoke({**self.input(self.rows[:1]), "messages": [*second["messages"], HumanMessage("again", id="new",
                                 additional_kwargs={"attachment_ids": [str(self.rows[0].id)]})]})
        reader.assert_called_once()
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.classify.call_count, 3)
        for invocation in self.calls:
            self.assertIn("MIDDLE 14:00 ~ 15:00", "\n".join(m.text for m in invocation["messages"]))
        self.assertEqual(len({r["attachment_web_call_id"] for r in (first, second, third)}), 3)

    def test_guard_ownership_url_validation_and_hidden_tool(self):
        with patch.object(browser_research, "web_body") as reader:
            self.classify.return_value = {"allowed": False, "capabilities": []}
            out = self.graph().invoke(self.input())
            self.assertEqual(self.calls, [])
            self.assertFalse(any(isinstance(m, ToolMessage) for m in out["messages"]))
            self.classify.return_value = {"allowed": True, "capabilities": []}
            with self.assertRaisesRegex(ValueError, "missing conversation attachment"):
                self.graph().invoke({**self.input(), "attachment_session_id": str(uuid.uuid4())})
            self.rows[0].source_url = "http://127.0.0.1/"
            self.rows[0].save(update_fields=["source_url"])
            with self.assertRaises(ValidationError):
                self.graph().invoke(self.input(self.rows[:1]))
            out = self.graph([call("jev_read_body", {"url": "https://example.com/override"}, "forged"),
                              AIMessage("FINAL")]).invoke(self.input([]))
            self.assertEqual(next(m for m in out["messages"] if isinstance(m, ToolMessage)).status, "error")
            reader.assert_not_called()

    def clear_read_result(self):
        ChatAttachment.objects.filter(session=self.session).update(url_read_result={}, extracted_text="")

    def test_current_and_historical_unreadable_are_explicit_not_body_cached(self):
        for old in (False, True):
            for status in ("busy", "timeout", "blocked", "error", "partial"):
                with self.subTest(old=old, status=status), patch.object(browser_research, "web_body", return_value={"status": status}):
                    self.clear_read_result()
                    data = self.input(self.rows[:1])
                    if old:
                        data["messages"] += [AIMessage("previous"), HumanMessage("followup", id="next")]
                    out = self.graph().invoke(data)
                    result = next(m for m in out["messages"] if isinstance(m, ToolMessage))
                    self.assertEqual(result.status, "error")
                    self.assertTrue(json.loads(result.content)["incomplete"])
                    text = "\n".join(m.text for m in self.calls[0]["messages"])
                    self.assertIn(f'"status": "{status}"', text)
                    self.assertIn('"available_body": false', text)
                    self.rows[0].refresh_from_db()
                    self.assertEqual(self.rows[0].extracted_text, "")

    def test_historical_failure_aggregates_only_retained_sources(self):
        old = self.input(self.rows[:2])["messages"][0]
        current = HumanMessage("current source", id="next", additional_kwargs={"attachment_ids": [str(self.rows[2].id)]})
        for pruned in (True, False):
            with self.subTest(pruned=pruned):
                self.clear_read_result()
                def body(url, **kwargs):
                    if url == self.rows[0].source_url:
                        return {"status": "busy"}
                    if pruned and url == self.rows[1].source_url:
                        return observed("word " * 21000, "partial")
                    return self.body(url)
                with patch.object(browser_research, "web_body", side_effect=body):
                    out = self.graph().invoke({**self.input(), "messages": [old, AIMessage("prior"), current]})
                result = next(m for m in out["messages"] if isinstance(m, ToolMessage))
                self.assertEqual(result.status, "success" if pruned else "error")
                self.assertEqual(json.loads(result.content)["incomplete"], not pruned)
                self.assertIn(self.rows[2].source_url, result.content)
                text = "\n".join(m.text for m in self.calls[0]["messages"])
                self.assertIn("MIDDLE 14:00 ~ 15:00", text)
                self.assertEqual('"status": "busy"' in text, not pruned)
                self.rows[2].refresh_from_db()
                self.assertTrue(self.rows[2].extracted_text)

    def test_failed_read_persists_across_turns_but_new_identity_can_read(self):
        row = self.rows[0]
        stale = ChatAttachment.objects.get(pk=row.pk)
        with patch.object(browser_research, "browse", AsyncMock(side_effect=TimeoutError())) as external:
            first = self.graph().invoke(self.input([row]))
            external.assert_awaited_once()
            row.refresh_from_db()
            self.assertEqual(row.url_read_result["status"], "timeout")
            for selected in (row, stale):
                self.assertRaises(attachments.URLBodyUnavailable, attachments.source_text, selected)
            followup = {**self.input([row]), "messages": [*first["messages"], HumanMessage("retry", id="next")]}
            self.graph().invoke(followup)
            external.assert_awaited_once()
            prompt = "\n".join(m.text for m in self.calls[0]["messages"])
            self.assertIn("개인정보·인증 정보를 제외", prompt)
            fresh = ChatAttachment.objects.create(session=self.session, kind="url", name="fresh", source_url=row.source_url)
            self.graph().invoke(self.input([fresh]))
            self.assertEqual(external.await_count, 2)

    def test_operational_errors_request_pasted_body_on_first_turn_without_retry(self):
        import httpx
        from mcp.shared.exceptions import McpError
        from mcp.types import ErrorData
        request = httpx.Request("GET", self.rows[0].source_url)
        secret = "PRIVATE_READER_ERROR"
        errors = [httpx.HTTPStatusError(secret, request=request, response=httpx.Response(code, request=request))
                  for code in (401, 403)]
        errors += [PermissionError(secret), RuntimeError(secret), McpError(ErrorData(code=403, message=secret))]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                self.clear_read_result()
                with patch.object(browser_research, "browse", AsyncMock(side_effect=error)) as reader:
                    for _ in range(2):
                        out = self.graph().invoke(self.input(self.rows[:1]))
                        self.assertTrue(out["attachment_source_incomplete"])
                        self.assertEqual(out["attachment_sources"], [])
                        prompt = "\n".join(m.text for m in self.calls[0]["messages"])
                        self.assertIn('"status": "error"', prompt)
                        self.assertIn("붙여 넣도록 요청", prompt)
                        self.assertIn("개인정보·인증 정보를 제외", prompt)
                        self.assertNotIn(secret, prompt)
                        self.assertNotIn(secret, json.dumps([m.model_dump() for m in out["messages"]], default=str))
                    reader.assert_awaited_once()
                self.rows[0].refresh_from_db()
                self.assertEqual(self.rows[0].url_read_result, {"status": "error"})
                self.assertEqual(self.rows[0].extracted_text, "")

    def test_upload_failed_url_creates_fresh_identity_and_preserves_dedup_and_history(self):
        from rest_framework.test import APIRequestFactory
        from llm.views.attachments import ChatAttachmentView
        row = self.rows[0]
        factory = APIRequestFactory()
        view = ChatAttachmentView.as_view(throttle_classes=[], authentication_classes=[])
        def upload(guest=None):
            request = factory.post("/", {"url": row.source_url}, format="json",
                                   HTTP_COOKIE=f"guest_id={guest or self.session.guest}")
            return view(request, session_id=str(self.session.pk))
        for status in ("busy", "timeout", "blocked", "error", "partial", "overflow"):
            with self.subTest(status=status):
                self.session.attachments.exclude(pk=row.pk).delete()
                failure = {"status": status}
                ChatAttachment.objects.filter(pk=row.pk).update(url_read_result=failure, extracted_text="")
                self.assertEqual(upload(uuid.uuid4()).status_code, 404)
                response = upload()
                self.assertEqual(response.status_code, 201)
                fresh = ChatAttachment.objects.get(pk=response.data["id"])
                self.assertNotEqual(fresh.pk, row.pk)
                self.assertEqual(upload().data["id"], str(fresh.pk))  # Unattempted dedup.
                with patch.object(browser_research, "web_body", side_effect=self.body) as reader:
                    out = self.graph().invoke(self.input([fresh]))
                    self.assertFalse(out["attachment_source_incomplete"])
                    self.assertEqual(upload().data["id"], str(fresh.pk))  # Successful dedup.
                    if status == "overflow":
                        self.assertRaises(attachments.AttachmentProcessingLimit, attachments.source_text, row)
                    else:
                        self.assertRaises(attachments.URLBodyUnavailable, attachments.source_text, row)
                    reader.assert_called_once_with(row.source_url, retry=False)
                row.refresh_from_db()
                self.assertEqual(row.url_read_result, failure)
                self.assertEqual(row.extracted_text, "")
                self.assertEqual(self.session.attachments.filter(source_url=row.source_url).count(), 2)

    def test_partial_observation_is_reused_without_full_success_claim(self):
        row = self.rows[0]
        result = self.body(row.source_url) | {"status": "partial", "limitations": ["article_missing"]}
        with patch.object(browser_research, "web_body", return_value=result) as reader:
            for _ in range(2):
                out = self.graph().invoke(self.input([row]))
                self.assertTrue(out["attachment_source_incomplete"])
                self.assertEqual(out["attachment_sources"], [])
                self.assertIn('"status": "partial"', next(m for m in self.calls[0]["messages"] if m.id == "q").text)
            reader.assert_called_once_with(row.source_url, retry=False)
        row.refresh_from_db()
        self.assertEqual(row.extracted_text, "")
        self.assertEqual(row.url_read_result["body"], result["body"])

    def test_cancelled_read_keeps_identity_and_does_not_poison_next_attempt(self):
        row = self.rows[0]
        for value in ({"status": "cancelled"}, Stopped()):
            with patch.object(browser_research, "web_body", **({"side_effect": value} if isinstance(value, Exception) else {"return_value": value})):
                self.assertRaises(Stopped, attachments.source_text, row)
            row.refresh_from_db()
            self.assertEqual(row.url_read_result, {})
        with patch.object(browser_research, "web_body", side_effect=self.body) as reader:
            self.assertIn("BEGIN", attachments.source_text(row))
            reader.assert_called_once()

    def test_course_fixed_role_tools_keep_scope_without_classifier_or_model(self):
        executed, calls = [], []
        graph = course_sub_agent.build(ScriptedModel(script=[], calls=calls), fake_tools(executed))
        with patch("llm.v2.middleware.dynamic_tools.MIGRATED_TOOLS", frozenset()):
            for allowed in (True, False):
                out = graph.invoke({"messages": [HumanMessage("잠실 코스")],
                                    "decision": {"allowed": allowed, "capabilities": ["schedule"]}})
                result = next(m for m in out["messages"] if isinstance(m, ToolMessage))
                self.assertEqual(result.status, "success" if allowed else "error")
        self.assertEqual(executed, ["plan_course"])
        self.assertEqual(calls, [])
        self.classify.assert_not_called()

    def test_partial_body_malformed_cancellation_and_limits(self):
        for value in ({"status": "ok", "body": "forged"}, "not JSON", self.body(self.rows[0].source_url) | {"status": "partial"}):
            with self.subTest(value=value), patch.object(browser_research, "browse", AsyncMock(return_value=value)):
                self.clear_read_result()
                self.graph().invoke(self.input(self.rows[:1]))
                text = next(m for m in self.calls[0]["messages"] if m.id == "q").text
                self.assertIn('"status": "partial"' if isinstance(value, dict) and value.get("status") == "partial" else '"status": "error"', text)
                self.rows[0].refresh_from_db()
                self.assertEqual(self.rows[0].extracted_text, "")
        for result, error in (({"status": "cancelled"}, Stopped),
                              (observed("word " * 21000, "partial"), DirectContextLimit),
                              ({"status": "overflow"}, attachments.AttachmentProcessingLimit)):
            self.clear_read_result()
            with patch.object(browser_research, "web_body", return_value=result), self.assertRaises(error):
                self.graph().invoke(self.input(self.rows[:1]))
            self.assertEqual(self.calls, [])
        self.clear_read_result()
        with patch.object(browser_research, "web_body", side_effect=Stopped()), self.assertRaises(Stopped):
            self.graph().invoke(self.input(self.rows[:1]))

    def test_role_tools_fixed_and_classifier_once_with_real_specialist(self):
        for module in (baseball_sub_agent, travel_sub_agent, place_sub_agent):
            calls = []
            model = ScriptedModel(script=[AIMessage("ROLE")], calls=calls)
            module.build(model, fake_tools([])).invoke({"messages": [HumanMessage("task")],
                                                       "decision": {"allowed": True, "capabilities": ["schedule"]}})
            self.assertEqual(set(calls[0]["tools"]), set(module.TOOLS))
        self.classify.assert_not_called()
        self.classify.return_value = {"allowed": True, "capabilities": ["day_plan"]}
        with patch.object(browser_research, "web_body", side_effect=self.body):
            out = self.graph([call("ask_baseball", {"task": "잠실 일정"}, "baseball"), AIMessage("SPECIALIST"),
                              AIMessage("MAIN")]).invoke(self.input(self.rows[:1]))
        self.classify.assert_called_once()
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(set(self.calls[1]["tools"]), set(baseball_sub_agent.TOOLS))
        self.assertEqual(out["messages"][-1].text, "MAIN")

    def test_frames_citations_and_public_history_no_nested_summary(self):
        run = {"answer": "", "messages": []}
        with patch.object(browser_research, "web_body", side_effect=self.body), patch(
                "llm.v2.agent.chain.get_graph", return_value=self.graph()):
            frames = list(chat_v2._frames(self.input(self.rows[:1]), run))
        self.assertEqual([d["status"] for k, d in frames if k == "tool"], ["running", "completed"])
        self.assertFalse(any(k == "delta" and d.get("parent_id") for k, d in frames))
        self.assertIn("](https://example.com/page0)", run["answer"])
        public = project_history([self.input()["messages"][0], *run["messages"], AIMessage(run["answer"], id="answer")],
                                 {"q": {"status": "completed", "answer_id": "answer"}})
        self.assertNotIn("BEGIN", json.dumps(public, ensure_ascii=False))

    def test_enabled_web_specialist_stays_available_for_separate_user_request(self):
        self.classify.return_value = {"allowed": True, "capabilities": ["web_research"]}
        with patch.dict(os.environ, {"WEB_RESEARCH_ENABLED": "true"}), patch(
                "llm.v2.agent.web_sub_agent.research_model", return_value=ScriptedModel(script=[AIMessage("SEARCH")], calls=[])), patch(
                "llm.v2.agent.web_sub_agent.direct_tools", return_value=[]), patch.object(browser_research, "web_body", side_effect=self.body):
            out = self.graph([call("ask_web_research", {"task": "별도 메뉴 검색"}, "search"), AIMessage("MAIN")]).invoke(self.input(self.rows[:1]))
        self.classify.assert_called_once()
        self.assertEqual([m.name for m in out["messages"] if isinstance(m, ToolMessage)], ["jev_read_body", "ask_web_research"])

    def test_tool_replay_and_arguments_are_denied_before_handler(self):
        from types import SimpleNamespace
        from llm.v2.middleware.dynamic_tools import DynamicToolMiddleware
        middleware = DynamicToolMiddleware(["jev_read_body"], {})
        state = {"decision": {"allowed": True}, "attachment_web_call_id": "forced", "attachment_web_done": False}
        for identifier, args in (("forged", {}), ("forced", {"url": self.rows[0].source_url}),
                                 ("forced", {"attachment_ids": [str(self.rows[1].id)]})):
            request = SimpleNamespace(state=state, tool_call={"name": "jev_read_body", "id": identifier, "args": args})
            result = middleware.wrap_tool_call(request, lambda _: self.fail("unauthorized handler"))
            self.assertEqual(result.status, "error")

    def test_window_eviction_bytes_and_concurrent_context_isolation(self):
        from concurrent.futures import ThreadPoolExecutor
        row = self.rows[0]
        data = self.input([row])
        data["messages"] += [HumanMessage(str(i), id=f"later{i}") for i in range(4)]
        with patch.object(browser_research, "web_body") as reader:
            self.graph().invoke(data)
            reader.assert_not_called()
        row.extracted_text = "x" * (attachments.MAX_TEXT + 1)
        row.save(update_fields=["extracted_text"])
        with self.assertRaises(attachments.AttachmentProcessingLimit), patch.object(browser_research, "web_body") as reader:
            self.graph().invoke(self.input([row]))
        reader.assert_not_called()
        row.extracted_text = "cached first original"
        row.save(update_fields=["extracted_text"])
        self.rows[1].extracted_text = "cached second original"
        self.rows[1].save(update_fields=["extracted_text"])
        def execute(selected):
            calls = []
            graph = chain.build_graph(ScriptedModel(script=[AIMessage("MAIN")], calls=calls), fake_tools([]))
            out = graph.invoke(self.input([selected]))
            return out["attachment_web_call_id"], next(m for m in calls[0]["messages"] if m.id == "q").text
        with patch.object(browser_research, "web_body") as reader, ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(execute, self.rows[:2]))
        reader.assert_not_called()
        self.assertNotEqual(results[0][0], results[1][0])
        for i, (_, text) in enumerate(results):
            self.assertIn(self.rows[i].source_url, text)
            self.assertNotIn(self.rows[1-i].source_url, text)

    def test_text_only_and_no_attachment_do_not_read(self):
        row = ChatAttachment.objects.create(session=self.session, kind="text", name="notes")
        for rows in ([], [row]):
            with patch.object(attachments, "source_text", return_value="plain text"), patch.object(browser_research, "web_body") as reader:
                out = self.graph().invoke(self.input(rows))
                self.assertEqual(len(self.calls), 1)
                self.assertFalse(any(isinstance(m, ToolMessage) for m in out["messages"]))
                reader.assert_not_called()
