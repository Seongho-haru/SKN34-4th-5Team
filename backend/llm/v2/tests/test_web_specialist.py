"""Offline web ownership and genuine body admission regressions."""
import json
import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from django.test import SimpleTestCase, TransactionTestCase
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llm.models import ChatAttachment, ChatSession
from llm.service import attachments
from llm.v2.agent import browser_research, chain, travel_sub_agent
from llm.v2.tests.test_chain import ScriptedModel, fake_tools, call


def observed(body, status="ok", **extra):
    return {"schema_version": 1, "extractor_version": "jev-dom-v1", "status": status,
            "requested_url": "https://example.com/article", "final_url": "https://example.com/article",
            "source_url": "https://example.com/article", "title": "Fixture", "body": body,
            "source_kind": "rendered_dom_snapshot", "collected_at": "2026-10-08T00:00:00Z",
            "frames": [], "limitations": [], **extra}


class WebOwnershipTests(SimpleTestCase):
    def test_keyword_delegation_is_available_without_web_capability(self):
        for capabilities in (["web_research"], ["schedule"], []):
            calls = []
            script = [call("ask_web_research", {"task": "잠실 메뉴 검색"}, "web")]
            script.append(AIMessage("검색 근거 https://example.com/menu; 시간 미확인"))
            script.append(AIMessage("메인 안내"))
            with patch.dict(os.environ, {"WEB_RESEARCH_ENABLED": "false"}), patch(
                    "llm.v2.middleware.jev_guidelines.classify",
                    return_value={"allowed": True, "capabilities": capabilities}):
                graph = chain.build_graph(ScriptedModel(script=script, calls=calls), fake_tools([]))
                out = graph.invoke({"messages": [HumanMessage("잠실 검색")]})
            self.assertNotIn("web_search", calls[0]["tools"])
            self.assertNotIn("research_public_web", calls[0]["tools"])
            result = next(m for m in out["messages"] if isinstance(m, ToolMessage))
            self.assertEqual(set(calls[0]["tools"]),
                             set(chain.sub_agents.SPECIALISTS) | {"present_planning_questions"} | ({"get_games"} if "schedule" in capabilities else set()))
            self.assertEqual(result.status, "success")
            self.assertIn("https://example.com/menu", result.content)
            self.assertEqual(calls[1]["tools"], ())
            self.assertEqual(out["messages"][-1].text, "메인 안내")
        self.assertNotIn("research_public_web", travel_sub_agent.TOOLS)

    def test_page_analysis_accepts_only_completed_supported_answer(self):
        good = {"source_url": "https://example.com/", "result": {
            "status": "done", "answer": "generated analysis", "final_text": "observed page"}}
        with patch.object(browser_research, "browse", AsyncMock(return_value=good)) as browse:
            result = browser_research.web_page_analysis("https://example.com/", "question")
        self.assertEqual(result, {"status": "ok", "analysis": "generated analysis"})
        self.assertFalse(browse.call_args.kwargs.get("body", False))
        self.assertIn("question", browse.call_args.args[1])
        self.assertIn("iframe", browse.call_args.args[1])
        with patch.object(browser_research, "browse", AsyncMock(return_value=[{"type": "text", "text": json.dumps(good)}])):
            self.assertEqual(browser_research.web_page_analysis("https://example.com/")["analysis"], "generated analysis")
        for value in ([{"type": "text", "text": "not JSON"}], {}, {"result": []},
                      {"result": {"status": "done", "answer": "fake"}},
                      {"result": {"status": "done", "answer": "", "final_text": "page"}},
                      {"result": {"status": "done", "answer": "fake", "final_text": "page", "error": "failed"}},
                      {"status": "partial", "result": good["result"]},
                      {"result": {"status": "blocked", "answer": "fake", "final_text": "page"}},
                      {"result": {"status": "done", "answer": "x" * (attachments.MAX_TEXT + 1), "final_text": "page"}}):
            with self.subTest(value_type=type(value).__name__), patch.object(browser_research, "browse", AsyncMock(return_value=value)):
                self.assertNotEqual(browser_research.web_page_analysis("https://example.com/")["status"], "ok")

    def test_page_analysis_rejects_malformed_statuses(self):
        good = {"status": "done", "answer": "analysis", "final_text": "page"}
        for malformed in ([], {}):
            cases = (({"status": malformed, "result": good}, "error"),
                     ({"result": {**good, "status": malformed}}, "error"),
                     ({"result": {**good, "goal_assessment": {"status": malformed}}}, "partial"))
            for value, expected in cases:
                for payload in (value, [{"type": "text", "text": json.dumps(value)}]):
                    with self.subTest(value=value, wrapped=isinstance(payload, list)), patch.object(
                            browser_research, "browse", AsyncMock(return_value=payload)):
                        self.assertEqual(browser_research.web_page_analysis("https://example.com/"),
                                         {"status": expected})
                    self.assertFalse(browser_research._admission.locked())

    def test_shared_status_boundary_rejects_malformed_statuses(self):
        for malformed in ([], {}):
            for value in ({"status": malformed}, {"status": "ok", "result": {"status": malformed}}):
                for payload in (value, [{"type": "text", "text": json.dumps(value)}]):
                    with self.subTest(value=value, wrapped=isinstance(payload, list)):
                        self.assertEqual(browser_research.result_status(payload), "error")
            with patch.object(browser_research, "browse", AsyncMock(return_value={"status": malformed})):
                self.assertEqual(browser_research.web_body("https://example.com/"),
                                 {"status": "error", "source_url": "https://example.com/"})
            self.assertFalse(browser_research._admission.locked())

    def test_page_analysis_reuses_admission_validation_and_cancellation(self):
        from llm.service.chat_runs import Stopped
        with patch.object(browser_research, "browse", AsyncMock()) as browse:
            with self.assertRaises(Exception):
                browser_research.web_page_analysis("http://127.0.0.1/")
            browser_research._admission.acquire()
            try:
                self.assertEqual(browser_research.web_page_analysis("https://example.com/")["status"], "busy")
            finally:
                browser_research._admission.release()
            browse.assert_not_called()
        for error, status in ((TimeoutError(), "timeout"), (RuntimeError(), "error")):
            with patch.object(browser_research, "browse", AsyncMock(side_effect=error)):
                self.assertEqual(browser_research.web_page_analysis("https://example.com/")["status"], status)
            self.assertFalse(browser_research._admission.locked())
        with patch.object(browser_research, "browse", AsyncMock(side_effect=Stopped())), self.assertRaises(Stopped):
            browser_research.web_page_analysis("https://example.com/")
        self.assertFalse(browser_research._admission.locked())

    def test_body_rejects_generated_partial_private_overflow_and_busy(self):
        for result in ({"final_text": "made up", "status": "ok"},
                       {"status": "partial", "body": "short"}, {"status": "blocked"},
                       {"status": "ok", "body": "x" * (attachments.MAX_TEXT + 1), "source_kind": "rendered_dom_snapshot"}):
            with patch.object(browser_research, "browse", AsyncMock(return_value=result)):
                self.assertNotEqual(browser_research.web_body("https://example.com/")["status"], "ok")
        with patch.object(browser_research, "browse", AsyncMock()) as browse:
            with self.assertRaises(Exception):
                browser_research.web_body("http://127.0.0.1/")
            browse.assert_not_called()
            browser_research._admission.acquire()
            try:
                self.assertEqual(browser_research.web_body("https://example.com/")["status"], "busy")
            finally:
                browser_research._admission.release()
            browse.assert_not_called()


class URLContextTests(TransactionTestCase):
    def setUp(self):
        self.session = ChatSession.objects.create(guest=uuid.uuid4())
        self.row = ChatAttachment.objects.create(session=self.session, kind="url", name="article",
                                                 source_url="https://example.com/article")

    def invoke(self, older=False):
        calls = []
        messages = [HumanMessage("잠실 첨부", id="q", additional_kwargs={"attachment_ids": [str(self.row.id)]})]
        if older:
            messages.extend([AIMessage("previous"), HumanMessage("잠실 첨부", id="followup")])
        with patch("llm.v2.middleware.jev_guidelines.classify", return_value={"allowed": True, "capabilities": []}):
            graph = chain.build_graph(ScriptedModel(script=[AIMessage("findings"), AIMessage("main")], calls=calls), fake_tools([]))
            graph.invoke({"messages": messages,
                          "attachment_session_id": str(self.session.id)})
        return next(m for m in calls[-1]["messages"] if isinstance(m, HumanMessage)).text

    def test_generated_analysis_reaches_main_without_original_or_cache_claims(self):
        from llm.v2.middleware.attachment_context import direct_context
        analysis = attachments.URLPageAnalysis("generated legacy analysis")
        context = direct_context([self.row], {str(self.row.id): analysis})
        self.assertIn("generated legacy analysis", context)
        self.assertIn('"original_body": false', context)
        self.assertNotIn('"chars"', context)
        with patch.object(browser_research, "web_body", return_value={"status": "ok", "analysis": str(analysis)}):
            self.assertRaises(attachments.URLBodyUnavailable, attachments.source_text, self.row)
        self.row.refresh_from_db()
        self.assertEqual(self.row.extracted_text, "")

    def test_observed_body_reaches_main_and_success_is_reused(self):
        body = "BEGIN " + "baseball " * 1500 + " MIDDLE 14:00 ~ 15:00 " + "seats " * 1500 + " END"
        with patch.object(browser_research, "web_body", return_value=observed(body)) as fetch:
            for older in (False, True):
                context = self.invoke(older)
                self.assertIn(body, context)
                self.assertIn('"original_body": true', context)
                self.assertIn('"chars"', context)
                self.assertNotIn("14:15:00", context)
            self.assertEqual(fetch.call_count, 1)
        self.row.refresh_from_db()
        self.assertEqual(self.row.extracted_text, body)

    def test_current_success_survives_prior_transport_timeout_in_model_request(self):
        old = ChatAttachment.objects.create(session=self.session, kind="url", name="old",
                                             source_url="https://example.com/old")
        calls = []
        messages = [HumanMessage("고척", id="old", additional_kwargs={"attachment_ids": [str(old.id)]}),
                    AIMessage("previous"), HumanMessage("현재 잠실", id="current",
                    additional_kwargs={"attachment_ids": [str(self.row.id)]})]
        async def browse(url, *args, **kwargs):
            if url == old.source_url:
                raise TimeoutError()
            return observed("CURRENT_ORIGINAL 14:00 ~ 15:00")
        with patch.object(browser_research, "browse", side_effect=browse), patch(
                "llm.v2.middleware.jev_guidelines.classify", return_value={"allowed": True, "capabilities": []}):
            graph = chain.build_graph(ScriptedModel(script=[AIMessage("findings"), AIMessage("main")], calls=calls), fake_tools([]))
            graph.invoke({"messages": messages, "attachment_session_id": str(self.session.id)})
        self.assertIn("CURRENT_ORIGINAL", next(m for m in calls[-1]["messages"] if m.id == "current").text)
        self.assertIn('"status": "timeout"', next(m for m in calls[-1]["messages"] if m.id == "old").text)

    def test_malformed_metadata_never_cache_or_abort_model(self):
        malformed = {"source_url": 123, "frames": {}, "limitations": {}, "requested_url": {},
                     "final_url": [], "title": {}, "extractor_version": [], "collected_at": [], "schema_version": []}
        for key, value in malformed.items():
            with self.subTest(key=key), patch.object(browser_research, "browse", AsyncMock(return_value=observed("DO_NOT_CACHE", **{key: value}))):
                context = self.invoke()
                self.assertIn('"status": "error"', context)
                self.assertNotIn("DO_NOT_CACHE", context)
            self.row.refresh_from_db()
            self.assertEqual(self.row.extracted_text, "")

    def test_transport_recovers_but_programming_security_and_stop_propagate(self):
        import httpx
        import anyio
        from llm.service.chat_runs import Stopped
        for error, status in ((TimeoutError(), "timeout"), (httpx.ConnectError("fixture"), "error"),
                              (ExceptionGroup("nested", [anyio.EndOfStream()]), "error")):
            with patch.object(browser_research, "browse", AsyncMock(side_effect=error)):
                self.assertIn(f'"status": "{status}"', self.invoke())
        for error in (RuntimeError("programmer"), Stopped(), ExceptionGroup("mixed", [TimeoutError(), ValueError("bug")])):
            with patch.object(browser_research, "browse", AsyncMock(side_effect=error)):
                self.assertRaises(type(error), self.invoke)
        with patch.object(browser_research, "browse", AsyncMock(return_value=observed("private", source_url="http://127.0.0.1/"))):
            from rest_framework.exceptions import ValidationError
            self.assertRaises(ValidationError, self.invoke)

    def test_success_prompt_keeps_exact_url_destination_and_no_paste_request(self):
        with patch.object(browser_research, "web_body", return_value=observed("CURRENT_BODY")):
            context = self.invoke()
        self.assertIn("링크 destination에는 정확한 URL만", context)
        self.assertIn("링크 밖 일반 텍스트", context)
        self.assertNotIn("본문을 붙여", context)

    def test_current_observed_partial_survives_previous_busy_source(self):
        old = ChatAttachment.objects.create(session=self.session, kind="url", name="old Gocheok",
                                             source_url="https://example.com/old")
        calls = []
        messages = [HumanMessage("고척", id="old", additional_kwargs={"attachment_ids": [str(old.id)]}),
                    AIMessage("previous failure"), HumanMessage("현재 잠실 파오파오 휴게시간?", id="current",
                    additional_kwargs={"attachment_ids": [str(self.row.id)]})]
        def read(url):
            if url == old.source_url:
                return {"status": "busy"}
            return observed("파오파오 휴게시간 14:00 ~ 15:00", "partial", limitations=["material_article_frame_missing"])
        with patch.object(browser_research, "web_body", side_effect=read), patch(
                "llm.v2.middleware.jev_guidelines.classify", return_value={"allowed": True, "capabilities": []}):
            graph = chain.build_graph(ScriptedModel(script=[AIMessage("findings"), AIMessage("main")], calls=calls), fake_tools([]))
            graph.invoke({"messages": messages, "attachment_session_id": str(self.session.id)})
        current = next(m for m in calls[-1]["messages"] if m.id == "current").text
        previous = next(m for m in calls[-1]["messages"] if m.id == "old").text
        self.assertIn("파오파오 휴게시간 14:00 ~ 15:00", current)
        self.assertIn('"status": "partial"', current)
        self.assertIn('"status": "busy"', previous)
        self.assertNotIn("고척", current)
        self.row.refresh_from_db()
        self.assertEqual(self.row.extracted_text, "")

    def test_failed_cancelled_and_overflow_never_cache_body(self):
        from llm.v2.middleware.attachment_context import DirectContextLimit
        for status in ("blocked", "partial", "busy", "timeout", "error"):
            with patch.object(browser_research, "web_body", return_value={"status": status}) as fetch:
                for _ in range(2):
                    context = self.invoke()
                    self.assertIn('"available_body": false', context)
                    self.assertIn(f'"status": "{status}"', context)
                    self.assertNotIn("fabricated", context)
                self.assertEqual(fetch.call_count, 2)
            self.row.refresh_from_db()
            self.assertEqual(self.row.extracted_text, "")
        for result, error in (({"status": "overflow"}, attachments.AttachmentProcessingLimit),
                              (observed("word " * 21000, "partial"), DirectContextLimit)):
            with patch.object(browser_research, "web_body", return_value=result), self.assertRaises(error):
                self.invoke()
            self.row.refresh_from_db()
            self.assertEqual(self.row.extracted_text, "")
        with patch.object(browser_research, "web_body", side_effect=RuntimeError("cancelled")), self.assertRaises(RuntimeError):
            self.invoke()
        self.row.refresh_from_db()
        self.assertEqual(self.row.extracted_text, "")
