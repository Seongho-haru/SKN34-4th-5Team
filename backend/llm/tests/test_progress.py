import json
import queue
import threading
import uuid
from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core import signing
from django.db import DatabaseError, connection
from django.test import TransactionTestCase, override_settings
from django.utils import timezone
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from rest_framework.test import APIClient

from ..v1.chat_service import ChatService
from ..models import ChatMessage, ChatProgressEvent, ChatSession, ChatTurn
from ..v1.progress import (
    MAX_ARGUMENT_BYTES,
    MAX_RESULT_BYTES,
    ProgressCallback,
    ProgressCancelled,
    ProgressCollector,
    ProgressStorageError,
    collect,
    current,
    operation,
    project_event,
    sanitize,
)
from ..v1.rag import dispatcher, domain_tools, pipeline as rag_pipeline
from ..v1.rag.assistant import pipeline as assistant
from ..v1.rag.assistant import tools as assistant_tools
from ..v1.rag.pipeline import last_detail
from ..v1.rag.venue import agent as venue
from ..v1.serializers import ChatTurnSerializer
from ..v1.views import RECEIPT_SALT, threaded_stream


def parse_frame(frame):
    lines = frame.decode().strip().splitlines()
    return lines[0][7:], json.loads(lines[1][6:])


def queued_events(collector):
    events = []
    while True:
        try:
            kind, payload = collector.output.get_nowait()
        except queue.Empty:
            return events
        if kind == "progress":
            events.append(payload)


@override_settings(CHAT_CHECKPOINT_SIGNING_KEY="progress-test-signing-key")
class ProgressApiTest(TransactionTestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = get_user_model().objects.create_user(username=f"progress-{uuid.uuid4()}")
        self.client.force_authenticate(self.user)
        self.session = ChatSession.objects.create(user=self.user)
        model = RunnableLambda(lambda _prompt: AIMessage(content="기본 답변"))
        patcher = patch("llm.v1.chat_service.ChatOpenAI", return_value=model)
        patcher.start()
        self.addCleanup(patcher.stop)

    def stream(self, stream_method, *, guest=False):
        target = "/api/v1/chat/guest/" if guest else f"/api/v1/chat/sessions/{self.session.pk}/messages/"
        data = {"messages": [{"role": "user", "content": "질문"}]} if guest else {"content": "질문"}
        patcher = patch.object(ChatService, "stream_with_history", side_effect=stream_method)
        patcher.start()
        self.addCleanup(patcher.stop)
        return self.client.post(
            target, data, format="json", HTTP_ACCEPT="text/event-stream",
            REMOTE_ADDR="203.0.113.41",
        )

    def test_started_reaches_http_client_before_tool_release_and_finalize_does_not_duplicate(self):
        release = threading.Event()

        def slow_tool(*_args):
            with operation("tool", "search_places", arguments={"query": "잠실 맛집"}):
                self.assertTrue(release.wait(3))
            yield "최종 답변"

        response = self.stream(slow_tool)
        frames = iter(response.streaming_content)
        name, checkpoint = parse_frame(next(frames))
        self.assertEqual(name, "checkpoint")
        name, started = parse_frame(next(frames))
        self.assertEqual((name, started["status"]), ("progress", "started"))
        self.assertFalse(release.is_set())
        self.assertEqual(ChatProgressEvent.objects.filter(turn_id=started["turn_id"]).count(), 1)

        release.set()
        remaining = [parse_frame(frame) for frame in frames]
        self.assertEqual([name for name, _ in remaining], ["progress", "delta", "done"])
        done = remaining[-1][1]
        turn = ChatTurn.objects.get(pk=started["turn_id"])
        rows_before = turn.progress_events.count()
        payload = {"receipt": done["receipt"], "prefix": "최종 답변", "status": "completed"}
        url = f"/api/v1/chat/turns/{turn.pk}/finalize/"
        first = self.client.post(url, payload, format="json")
        second = self.client.post(url, payload, format="json")
        self.assertEqual((first.status_code, second.status_code), (200, 200))
        self.assertEqual(turn.progress_events.count(), rows_before)

        receipt = signing.loads(
            done["receipt"], key="progress-test-signing-key", salt=RECEIPT_SALT
        )
        self.assertEqual(receipt["length"], len("최종 답변"))
        self.assertNotIn("잠실 맛집", json.dumps(receipt, ensure_ascii=False))

    def test_guest_progress_reuses_ephemeral_turn_and_writes_no_chat_rows(self):
        before = tuple(model.objects.count() for model in (
            ChatSession, ChatTurn, ChatMessage, ChatProgressEvent,
        ))

        def guest_tool(*_args):
            with operation("tool", "get_games", arguments={"team_code": "LG"}):
                pass
            yield "답"

        response = self.stream(guest_tool, guest=True)
        events = [parse_frame(frame) for frame in response.streaming_content]
        progress = [data for name, data in events if name == "progress"]
        self.assertEqual([event["status"] for event in progress], ["started", "completed"])
        self.assertEqual(len({event["turn_id"] for event in progress}), 1)
        self.assertEqual(before, tuple(model.objects.count() for model in (
            ChatSession, ChatTurn, ChatMessage, ChatProgressEvent,
        )))

    def test_admin_gets_sanitized_details_but_guest_forged_flag_does_not(self):
        def detailed_stream(*_args):
            collector = current()
            operation_id = uuid.uuid4()
            collector.emit(
                kind="tool", status="started", label="경기 일정 조회 중",
                operation_id=operation_id, tool_name="get_games", tool_call_id="call-admin",
                arguments={"team": "LG", "sql": "SELECT secret"},
            )
            collector.emit(
                kind="tool", status="completed", label="경기 일정 조회 완료",
                operation_id=operation_id, tool_name="get_games", tool_call_id="call-admin",
                result={"count": 1, "password": "secret"},
            )
            yield "답변"

        self.user.is_staff = True
        self.user.save(update_fields=("is_staff",))
        admin_events = [parse_frame(frame) for frame in self.stream(detailed_stream).streaming_content]
        admin_progress = [data for name, data in admin_events if name == "progress"]
        self.assertEqual(admin_progress[0]["tool_call_id"], "call-admin")
        self.assertEqual(admin_progress[0]["arguments"], {"team": "LG"})
        self.assertEqual(admin_progress[1]["result"], {"count": 1})

        with patch.object(ChatService, "stream_with_history", side_effect=detailed_stream):
            guest = self.client.post(
                "/api/v1/chat/guest/",
                {"messages": [{"role": "user", "content": "질문"}], "debug": True},
                format="json", HTTP_ACCEPT="text/event-stream", REMOTE_ADDR="203.0.113.42",
            )
        guest_events = [parse_frame(frame) for frame in guest.streaming_content]
        guest_progress = [data for name, data in guest_events if name == "progress"]
        self.assertTrue(guest_progress)
        self.assertFalse(
            {"tool_call_id", "arguments", "result", "truncated"} & guest_progress[0].keys()
        )

    def test_saved_history_details_require_staff_and_keep_owner_scope(self):
        turn = ChatTurn.objects.create(session=self.session, question="질문")
        ProgressCollector(turn.pk, persistent=True).emit(
            kind="tool", status="completed", label="경기 일정 조회 완료",
            tool_name="get_games", tool_call_id="call-history",
            arguments={"team": "LG"}, result={"count": 1}, direct=True,
        )
        ordinary = self.client.get(f"/api/v1/chat/sessions/{self.session.pk}/turns/").json()["results"][0]["progress"][0]
        self.assertFalse({"tool_call_id", "arguments", "result", "truncated"} & ordinary.keys())

        self.user.is_staff = True
        self.user.save(update_fields=("is_staff",))
        admin = self.client.get(f"/api/v1/chat/sessions/{self.session.pk}/turns/").json()["results"][0]["progress"][0]
        self.assertEqual(admin["tool_call_id"], "call-history")
        self.assertEqual(admin["arguments"], {"team": "LG"})
        self.assertEqual(admin["result"], {"count": 1})

    def test_guest_and_member_followups_normalize_langchain_history(self):
        seen = []

        def fake_dispatch(question, history=None, **_kwargs):
            seen.append((question, history))
            self.assertTrue(all(isinstance(item, dict) for item in history))
            yield "연속 답변"
            return {"answer": "연속 답변", "places": [], "coursePayload": None, "route": "test"}

        with (
            patch.object(rag_pipeline, "chat_chain", return_value=rag_pipeline.rag_chain),
            patch.object(rag_pipeline.dispatcher, "stream", side_effect=fake_dispatch),
        ):
            guest = self.client.post(
                "/api/v1/chat/guest/",
                {"messages": [
                    {"role": "user", "content": "잠실 알려줘"},
                    {"role": "assistant", "content": "잠실 답변"},
                    {"role": "user", "content": "가는 법은?"},
                ]},
                format="json", HTTP_ACCEPT="text/event-stream", REMOTE_ADDR="203.0.113.43",
            )
            self.assertIn("연속 답변", b"".join(guest.streaming_content).decode())

            ChatMessage.objects.create(session=self.session, sequence_no=1, role="human", message="잠실 알려줘")
            ChatMessage.objects.create(session=self.session, sequence_no=2, role="ai", message="잠실 답변")
            member = self.client.post(
                f"/api/v1/chat/sessions/{self.session.pk}/messages/",
                {"content": "가는 법은?"}, format="json", HTTP_ACCEPT="text/event-stream",
            )
            self.assertIn("연속 답변", b"".join(member.streaming_content).decode())

        self.assertEqual(seen[0][1], [
            {"role": "user", "content": "잠실 알려줘"},
            {"role": "assistant", "content": "잠실 답변"},
        ])
        self.assertEqual(seen[1][1], seen[0][1])

    def test_turn_history_is_owner_only_and_exposes_stale_start_as_unknown(self):
        turn = ChatTurn.objects.create(session=self.session, question="질문")
        collector = ProgressCollector(turn.pk, persistent=True)
        operation_id = collector.emit(kind="tool", status="started", label="조회 중")
        turn.progress_events.update(created_at=timezone.now() - timedelta(minutes=6))

        own = self.client.get(f"/api/v1/chat/sessions/{self.session.pk}/turns/")
        self.assertEqual(own.status_code, 200)
        self.assertEqual(own.json()["results"][0]["progress"][0]["status"], "unknown")
        self.assertEqual(
            ChatProgressEvent.objects.get(operation_id=operation_id).status, "started"
        )

        other = get_user_model().objects.create_user(username=f"other-{uuid.uuid4()}")
        self.client.force_authenticate(other)
        self.assertEqual(
            self.client.get(f"/api/v1/chat/sessions/{self.session.pk}/turns/").status_code, 404
        )

    def test_turn_history_next_link_keeps_public_api_prefix(self):
        ChatTurn.objects.bulk_create([
            ChatTurn(session=self.session, question=f"질문 {index}")
            for index in range(21)
        ])

        first = self.client.get(f"/api/v1/chat/sessions/{self.session.pk}/turns/?page=1")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["count"], 21)
        self.assertEqual(
            first.json()["next"],
            f"/api/v1/chat/sessions/{self.session.pk}/turns/?page=2",
        )
        second = self.client.get(f"/api/v1/chat/sessions/{self.session.pk}/turns/?page=2")
        self.assertEqual((second.status_code, len(second.json()["results"])), (200, 1))

    def test_nonstream_has_persistent_turn_progress_and_model_work_is_outside_atomic(self):
        def run(_service, _values):
            self.assertFalse(connection.in_atomic_block)
            with operation("phase", "assistant"):
                return "비스트림 답변"

        with patch.object(ChatService, "_run", autospec=True, side_effect=run):
            response = self.client.post(
                f"/api/v1/chat/sessions/{self.session.pk}/messages/",
                {"content": "질문"}, format="json",
            )
        self.assertEqual(response.status_code, 201)
        data = response.json()
        turn = ChatTurn.objects.get(pk=data["turn_id"])
        self.assertEqual((turn.status, turn.progress_events.count()), ("completed", 2))
        self.assertEqual([event["status"] for event in data["progress"]], ["started", "completed"])
        self.assertEqual(self.session.messages.count(), 2)

    def test_nonstream_stale_history_is_409_and_failed_state_survives_rollback(self):
        def race(_service, _values):
            ChatMessage.objects.create(
                session=self.session, sequence_no=1, role="human", message="경쟁 요청"
            )
            return "늦은 답변"

        with patch.object(ChatService, "_run", autospec=True, side_effect=race):
            response = self.client.post(
                f"/api/v1/chat/sessions/{self.session.pk}/messages/",
                {"content": "질문"}, format="json",
            )
        self.assertEqual(response.status_code, 409)
        turn = ChatTurn.objects.get(question="질문")
        self.assertEqual(turn.status, "failed")
        self.assertEqual(list(self.session.messages.values_list("message", flat=True)), ["경쟁 요청"])

        with (
            patch.object(ChatService, "_run", return_value="답변"),
            patch(
                "llm.v1.chat_message_histories.DjangoChatMessageHistory.add_messages",
                side_effect=DatabaseError("message write failed"),
            ),
        ):
            with self.assertRaises(DatabaseError):
                self.client.post(
                    f"/api/v1/chat/sessions/{self.session.pk}/messages/",
                    {"content": "저장 실패"}, format="json",
                )
        self.assertEqual(ChatTurn.objects.get(question="저장 실패").status, "failed")


class ProgressCoreTest(TransactionTestCase):
    def collector(self, turn=None, *, persistent=False):
        return ProgressCollector(turn or uuid.uuid4(), persistent=persistent, output=queue.Queue())

    def test_sanitization_bounds_and_secret_like_values_never_reach_rows_or_events(self):
        user = get_user_model().objects.create_user(username=f"sanitize-{uuid.uuid4()}")
        session = ChatSession.objects.create(user=user)
        turn = ChatTurn.objects.create(session=session, question="질문")
        collector = ProgressCollector(turn.pk, persistent=True)
        marker = "sk-SECRET123456789"
        huge = [{key: "x" * 500 for key in (
            "team", "team_code", "stadium", "stadium_code", "date_from",
            "date_to", "start_date", "end_date", "status", "home_away",
            "actual_date",
        )} for _ in range(20)]
        collector.emit(
            kind="tool", status="completed", label=f"{marker} SELECT * FROM users",
            tool_name="get_games", tool_call_id=f"token={marker}",
            arguments={"query": "SELECT password FROM users", "url": "https://internal/secret"},
            result=huge,
        )
        row = turn.progress_events.get()
        serialized = json.dumps({
            "label": row.label, "tool_name": row.tool_name, "tool_call_id": row.tool_call_id,
            "arguments": row.arguments, "result": row.result,
        }, ensure_ascii=False)
        for forbidden in (marker, "SELECT password", "https://internal", "users"):
            self.assertNotIn(forbidden, serialized)
        self.assertTrue(row.truncated)
        self.assertLessEqual(len(json.dumps(row.arguments, ensure_ascii=False).encode()), MAX_ARGUMENT_BYTES)
        self.assertLessEqual(len(json.dumps(row.result, ensure_ascii=False).encode()), MAX_RESULT_BYTES)

        collector.emit(
            kind="tool", status="completed", label="알 수 없는 도구",
            tool_name="unknown_tool", arguments={"query": "public"},
            result={"name": "should-not-leak"},
        )
        unknown = turn.progress_events.get(sequence_no=2)
        self.assertIsNone(unknown.arguments)
        self.assertIsNone(unknown.result)

        cleaned, truncated = sanitize({"query": marker, "sql": "SELECT 1"}, MAX_ARGUMENT_BYTES)
        self.assertEqual(cleaned, {"query": "[비공개]"})
        self.assertFalse(truncated)

    def test_public_tool_events_expose_only_tool_name_and_status(self):
        collector = self.collector()
        operation_id = uuid.uuid4()
        with collect(collector):
            collector.emit(
                kind="tool", status="started", label="경기 일정 조회 중",
                operation_id=operation_id, tool_name="get_games", tool_call_id="call-1",
                arguments={"team": "LG", "date_from": "2026-09-16", "sql": "SELECT secret"},
            )
            collector.emit(
                kind="tool", status="completed", label="경기 일정 조회 완료",
                operation_id=operation_id, tool_name="get_games", tool_call_id="call-1",
                result={"count": 1, "actual_date": "2026-09-16", "password": "secret"},
            )
        started, completed = [project_event(event) for event in queued_events(collector)]
        self.assertEqual(started["tool_name"], "get_games")
        self.assertEqual(completed["tool_name"], "get_games")
        for event in (started, completed):
            self.assertIsNone(event["summary"])
            self.assertFalse({"tool_call_id", "arguments", "result", "truncated"} & event.keys())

    def test_stream_tool_callbacks_keep_real_ids_arguments_and_domain_results(self):
        class ToolModel:
            def __init__(self, call):
                self.call = call
                self.plans = 0

            def bind_tools(self, _tools):
                return self

            def invoke(self, _messages, config=None):
                self.plans += 1
                return AIMessage(content="", tool_calls=[self.call]) if self.plans == 1 else AIMessage(content="READY")

            def stream(self, _messages, config=None):
                yield AIMessage(content="실제 최종 답변")

        game_results = {
            "LG": {"count": 1, "games": [{"game_date": "2026-09-16", "home_team": "LG 트윈스"}]},
            "두산": {"count": 2, "games": [{"game_date": "2026-09-17", "home_team": "두산 베어스"}]},
        }

        def get_games(team: str):
            """Test games."""
            return game_results[team]

        def get_standings(as_of: str):
            """Test standings."""
            return {"standings": [{"rank": 1, "team_name_ko": "LG 트윈스", "wins": 80}]}

        tool_map = {
            "get_games": StructuredTool.from_function(get_games),
            "get_standings": StructuredTool.from_function(get_standings),
        }

        def run(name, arguments):
            tool = tool_map[name]
            user = get_user_model().objects.create_user(username=f"tool-audit-{uuid.uuid4()}")
            session = ChatSession.objects.create(user=user)
            turn = ChatTurn.objects.create(session=session, question="질문")
            collector = ProgressCollector(turn.pk, persistent=True, output=queue.Queue())
            call = {"name": name, "args": arguments, "id": f"{name}-{arguments}", "type": "tool_call"}
            with collect(collector):
                self.assertEqual(
                    "".join(assistant.stream_answer(
                        "질문", model=ToolModel(call), tool_list=[tool],
                        retriever=lambda values: {**values, "context": "검색 결과 없음"},
                    )),
                    "실제 최종 답변",
                )
            events = [event for event in queued_events(collector) if event["kind"] == "tool"]
            return events, list(turn.progress_events.filter(kind="tool").order_by("sequence_no"))

        lg_events, lg_rows = run("get_games", {"team": "LG"})
        doosan_events, doosan_rows = run("get_games", {"team": "두산"})
        standings_events, standings_rows = run("get_standings", {"as_of": "2026-09-16"})
        self.assertEqual([event["status"] for event in lg_events], ["started", "completed"])
        self.assertEqual({event["tool_name"] for event in standings_events}, {"get_standings"})
        self.assertEqual(lg_rows[0].arguments, {"team": "LG"})
        self.assertEqual(doosan_rows[0].arguments, {"team": "두산"})
        self.assertNotEqual(lg_rows[0].tool_call_id, doosan_rows[0].tool_call_id)
        self.assertEqual(lg_rows[1].result["games"][0]["home_team"], "LG 트윈스")
        self.assertEqual(standings_rows[1].result["standings"][0]["rank"], 1)

    def test_storage_failure_propagates_without_dispatcher_fallback(self):
        user = get_user_model().objects.create_user(username=f"storage-{uuid.uuid4()}")
        session = ChatSession.objects.create(user=user)
        turn = ChatTurn.objects.create(session=session, question="질문")
        collector = ProgressCollector(turn.pk, persistent=True)
        with (
            collect(collector),
            patch.object(ChatProgressEvent.objects, "create", side_effect=DatabaseError("private")),
            patch.object(dispatcher, "_domain_answer") as fallback,
        ):
            with self.assertRaises(ProgressStorageError):
                dispatcher.answer("LG 선수 알려줘")
        fallback.assert_not_called()

        original = ChatProgressEvent.objects.create
        calls = 0

        def fail_second(**row):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise DatabaseError("terminal write failed")
            return original(**row)

        with collect(ProgressCollector(turn.pk, persistent=True)):
            with patch.object(ChatProgressEvent.objects, "create", side_effect=fail_second):
                with self.assertRaises(ProgressStorageError):
                    with operation("tool", "get_games"):
                        pass
        self.assertEqual(turn.progress_events.count(), 1)

    def test_concurrent_collectors_and_course_metadata_do_not_mix(self):
        barrier = threading.Barrier(2)
        seen = {}

        def run(name):
            collector = self.collector()
            with collect(collector):
                with operation("tool", "get_games", arguments={"team_code": name}):
                    barrier.wait()
            seen[name] = queued_events(collector)

        threads = [threading.Thread(target=run, args=(name,)) for name in ("LG", "DOOSAN")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertNotEqual(seen["LG"][0]["turn_id"], seen["DOOSAN"][0]["turn_id"])
        self.assertEqual([[e["sequence_no"] for e in seen[name]] for name in seen], [[1, 2], [1, 2]])
        self.assertIsNone(seen["LG"][0]["summary"])
        self.assertIsNone(seen["DOOSAN"][0]["summary"])

        details = {}

        def detail(name):
            rag_pipeline._LAST.set({"places": [{"name": name}]})
            barrier.wait()
            details[name] = last_detail()["places"][0]["name"]

        barrier.reset()
        threads = [threading.Thread(target=detail, args=(name,)) for name in ("A", "B")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3)
        self.assertEqual(details, {"A": "A", "B": "B"})

    def test_disconnect_persists_queued_terminal_once_and_starts_no_new_work(self):
        user = get_user_model().objects.create_user(username=f"disconnect-{uuid.uuid4()}")
        session = ChatSession.objects.create(user=user)
        turn = ChatTurn.objects.create(session=session, question="질문")
        collector = ProgressCollector(turn.pk, persistent=True, output=queue.Queue(maxsize=8))
        release = threading.Event()
        terminal_queued = threading.Event()
        continue_after_disconnect = threading.Event()
        producer_done = threading.Event()
        later_calls = []

        def stream():
            try:
                with operation("tool", "get_games", arguments={"team_code": "LG"}):
                    release.wait(3)
                terminal_queued.set()
                continue_after_disconnect.wait(3)
                with operation("tool", "get_standings"):
                    later_calls.append("called")
                yield "답변"
            finally:
                producer_done.set()

        response = threaded_stream(stream, collector, detail_getter=lambda: {})
        kind, event = next(response)
        self.assertEqual((kind, event["status"]), ("progress", "started"))
        release.set()
        self.assertTrue(terminal_queued.wait(3))
        response.close()  # terminal progress는 큐에 있지만 HTTP 소비자는 읽지 않은 시점
        continue_after_disconnect.set()
        self.assertTrue(producer_done.wait(3))
        self.assertEqual(
            list(turn.progress_events.values_list("status", flat=True)),
            ["started", "completed", "interrupted"],
        )
        self.assertEqual(turn.progress_events.filter(status="completed").count(), 1)
        self.assertEqual(later_calls, [])

    def test_cancel_stops_operation_body_and_real_structured_tool_invocation(self):
        collector = self.collector()
        collector.cancel(persist=False)
        manual_calls = []
        with collect(collector):
            with self.assertRaises(ProgressCancelled):
                with operation("tool", "get_games"):
                    manual_calls.append("called")
        self.assertEqual(manual_calls, [])

        structured_calls = []

        def real_tool():
            """테스트 도구."""
            structured_calls.append("called")
            return {"count": 1}

        tool = StructuredTool.from_function(real_tool, name="get_games")
        with collect(collector):
            with self.assertRaises(ProgressCancelled):
                tool.invoke({}, config={"callbacks": [ProgressCallback()]})
        self.assertEqual(structured_calls, [])

        assistant_call = patch.object(dispatcher.assistant, "answer")
        fallback_call = patch.object(dispatcher, "_domain_answer")
        with assistant_call as assistant_mock, fallback_call as fallback_mock, collect(collector):
            with self.assertRaises(ProgressCancelled):
                dispatcher.answer("LG 선수 알려줘")
        assistant_mock.assert_not_called()
        fallback_mock.assert_not_called()

    def test_json_string_error_from_real_structured_tool_is_failed(self):
        def error_tool():
            """JSON 오류를 문자열로 반환하는 테스트 도구."""
            return '{"error":"safe failure"}'

        collector = self.collector()
        tool = StructuredTool.from_function(error_tool, name="get_games")
        with collect(collector):
            self.assertEqual(
                tool.invoke({}, config={"callbacks": [ProgressCallback()]}),
                '{"error":"safe failure"}',
            )
        events = queued_events(collector)
        self.assertEqual([event["status"] for event in events], ["started", "failed"])

    def test_active_paths_emit_paired_events_without_tool_double_counting(self):
        collector = self.collector()
        with collect(collector), patch.object(
            dispatcher.assistant, "answer",
            return_value={"answer": "ok", "sources": [], "route": "agent"},
        ):
            dispatcher.answer("LG 선수 알려줘")
        events = queued_events(collector)
        self.assertEqual([event["status"] for event in events], ["started", "completed"])

        collector = self.collector()
        with (
            collect(collector),
            patch.object(dispatcher.assistant, "answer", side_effect=RuntimeError("down")),
            patch.object(
                dispatcher.club, "answer",
                return_value={"answer": "fallback", "sources": [], "route": "club"},
            ),
        ):
            dispatcher.answer("LG 선수 알려줘")
        events = queued_events(collector)
        self.assertEqual([event["status"] for event in events], [
            "started", "failed", "started", "completed",
        ])

        collector = self.collector()
        assistant_tools.new_state()
        with collect(collector):
            assistant.retrieve(
                {"question": "잠실 주차", "history": [], "hint": "JAMSIL"},
                _search=lambda *_: [], _embed=lambda _q: [0.0],
            )
        events = queued_events(collector)
        self.assertEqual(
            [(event["kind"], event["status"]) for event in events],
            [("retrieval", "started"), ("retrieval", "completed")],
        )

        class CallbackTool:
            name = "get_games"

            def invoke(self, arguments, config=None):
                handler = config["callbacks"][0]
                run_id = uuid.uuid4()
                handler.on_tool_start({"name": self.name}, json.dumps(arguments), run_id=run_id, inputs=arguments)
                result = {"count": 1}
                handler.on_tool_end(result, run_id=run_id)
                return result

        class CallbackModel:
            calls = 0

            def bind_tools(self, _tools, **_kwargs):
                return self

            def invoke(self, _messages, config=None):
                self.calls += 1
                if self.calls == 1:
                    return AIMessage(content="", tool_calls=[{
                        "name": "get_games", "args": {"team_code": "LG"}, "id": "call-1",
                    }])
                return AIMessage(content="done")

        collector = self.collector()
        with collect(collector), patch.object(domain_tools, "tools_for", return_value=(CallbackTool(),)):
            domain_tools.run_model(CallbackModel(), [], "club")
        tool_events = [event for event in queued_events(collector) if event["kind"] == "tool"]
        self.assertEqual([event["status"] for event in tool_events], ["started", "completed"])

        collector = self.collector()
        callback = ProgressCallback()
        parent_run, child_run = uuid.uuid4(), uuid.uuid4()
        with collect(collector):
            callback.on_chat_model_start({}, [], run_id=parent_run)
            callback.on_tool_start(
                {"name": "plan_course"}, "{}", run_id=child_run,
                parent_run_id=parent_run, inputs={"stadium_code": "JAMSIL"},
            )
            callback.on_tool_end({}, run_id=child_run, parent_run_id=parent_run)
            callback.on_llm_end({}, run_id=parent_run)
        nested = queued_events(collector)
        parent_id = nested[0]["operation_id"]
        child = [event for event in nested if event["kind"] == "tool"]
        self.assertEqual(len(child), 2)
        self.assertEqual({event["parent_operation_id"] for event in child}, {parent_id})

        class VenueAgent:
            def invoke(self, _input, config=None):
                handler = config["callbacks"][0]
                run_id = uuid.uuid4()
                handler.on_tool_start(
                    {"name": "get_stadium"}, "{}", run_id=run_id,
                    inputs={"stadium_code": "JAMSIL"},
                )
                handler.on_tool_end({"name": "잠실"}, run_id=run_id)
                return {"messages": [
                    ToolMessage(content='{"name":"잠실"}', tool_call_id="v1", name="get_stadium"),
                    AIMessage(content="잠실 안내"),
                ]}

        collector = self.collector()
        with collect(collector), patch.object(venue, "agent", return_value=VenueAgent()):
            venue.answer("잠실 주소 알려줘")
        tool_events = [event for event in queued_events(collector) if event["kind"] == "tool"]
        self.assertEqual([event["status"] for event in tool_events], ["started", "completed"])

    def test_error_return_is_failed_and_cancel_vs_stale_unknown_are_distinct(self):
        collector = self.collector()
        callback = ProgressCallback()
        run_id = uuid.uuid4()
        with collect(collector):
            callback.on_tool_start({"name": "get_games"}, "{}", run_id=run_id, inputs={})
            callback.on_tool_end({"error": "raw exception text"}, run_id=run_id)
        events = queued_events(collector)
        self.assertEqual([event["status"] for event in events], ["started", "failed"])
        self.assertNotIn("raw exception", json.dumps(events, ensure_ascii=False))

        user = get_user_model().objects.create_user(username=f"cancel-{uuid.uuid4()}")
        session = ChatSession.objects.create(user=user)
        turn = ChatTurn.objects.create(session=session, question="질문")
        collector = ProgressCollector(turn.pk, persistent=True)
        collector.cancel()
        interrupted = turn.progress_events.get()
        self.assertEqual(interrupted.status, "interrupted")

        stale_turn = ChatTurn.objects.create(session=session, question="다른 질문")
        stale_collector = ProgressCollector(stale_turn.pk, persistent=True)
        stale = stale_collector.emit(
            kind="tool", status="started", label="조회 중", direct=True
        )
        stale_turn.progress_events.filter(operation_id=stale).update(
            created_at=timezone.now() - timedelta(minutes=6)
        )
        serialized = ChatTurnSerializer(stale_turn).data
        self.assertIn("unknown", [event["status"] for event in serialized["progress"]])
        self.assertEqual(
            stale_turn.progress_events.get(operation_id=stale).status, "started"
        )
