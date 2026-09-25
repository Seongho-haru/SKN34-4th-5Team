from unittest.mock import Mock, patch

import json

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.db import DatabaseError
from django.test import override_settings
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from openai import OpenAIError
from rest_framework.test import APIClient, APITestCase
from drf_spectacular.generators import SchemaGenerator

from ..service.chat_service import ChatService
from ..models import ChatMessage, ChatSession, ChatTurn, Document, DocumentChunk
from ..v1.tools import create_baseball_tools


class ChatToolCallingTest(APITestCase):
    class BaseballService:
        def get_baseball_schema(self):
            return {"tables": [{"quoted_name": '"STADIUM"', "columns": ["name"]}]}

        def execute_baseball_select(self, sql, params=None, max_rows=100):
            assert sql == 'SELECT "name" FROM "STADIUM"'
            return {"columns": ["name"], "rows": [["잠실"]], "truncated": False}

    @staticmethod
    def scripted_model():
        seen = []

        def respond(prompt):
            messages = prompt.to_messages()
            seen.append(messages)
            tools = [message for message in messages if isinstance(message, ToolMessage)]
            if not tools:
                return AIMessage(content="", tool_calls=[{
                    "name": "get_baseball_schema", "args": {}, "id": "schema-1"
                }])
            if len(tools) == 1:
                return AIMessage(content="작성 중인 SQL은 노출하지 않습니다.", tool_calls=[{
                    "name": "execute_baseball_select",
                    "args": {"sql": 'SELECT "name" FROM "STADIUM"'},
                    "id": "select-1",
                }])
            return AIMessage(content="조회 결과는 잠실입니다.")

        model = RunnableLambda(respond)
        model.bind_tools = Mock(return_value=model)
        return model, seen

    def service(self):
        model, seen = self.scripted_model()
        return ChatService(llm=model, tools=create_baseball_tools(self.BaseballService())), seen

    def test_member_and_guest_use_schema_select_answer_loop(self):
        user = get_user_model().objects.create_user(username="tool-user")
        session = ChatSession.objects.create(user=user)
        service, member_calls = self.service()
        answer = service.invoke(user.pk, session.pk, "DB에서 야구장 조회해줘")
        self.assertEqual(answer, "조회 결과는 잠실입니다.")
        self.assertEqual([m.name for m in member_calls[-1] if isinstance(m, ToolMessage)],
                         ["get_baseball_schema", "execute_baseball_select"])
        self.assertEqual(list(session.messages.values_list("role", "message")), [
            ("human", "DB에서 야구장 조회해줘"), ("ai", "조회 결과는 잠실입니다.")
        ])

        guest_service, guest_calls = self.service()
        self.assertEqual(
            list(guest_service.stream_with_history([], "DB에서 야구장 조회해줘")),
            ["조회 결과는 잠실입니다."],
        )
        self.assertEqual([m.name for m in guest_calls[-1] if isinstance(m, ToolMessage)],
                         ["get_baseball_schema", "execute_baseball_select"])

    def test_unknown_tool_is_not_dispatched(self):
        chain = Mock()
        chain.invoke.side_effect = [
            AIMessage(content="", tool_calls=[{"name": "drop_database", "args": {}, "id": "bad"}]),
            AIMessage(content="그 도구는 사용할 수 없습니다."),
        ]
        service = ChatService.__new__(ChatService)
        service.chain = chain
        service.tools = ()
        service.tool_map = {}
        self.assertEqual(
            service._run({"question": "삭제", "chat_history": []}),
            "그 도구는 사용할 수 없습니다.",
        )
        tool_message = chain.invoke.call_args_list[1].args[0]["tool_messages"][1]
        self.assertIn("허용되지", tool_message.content)

        chain.invoke.reset_mock()
        chain.invoke.side_effect = None
        chain.invoke.return_value = AIMessage(
            content="", tool_calls=[{"name": "drop_database", "args": {}, "id": ""}]
        )
        self.assertIn("한도를 초과", service._run({"question": "삭제", "chat_history": []}))
        self.assertEqual(chain.invoke.call_count, 1)

    def test_stream_yields_immediately_and_close_stops_provider(self):
        class ProviderStream:
            def __init__(self, chunks):
                self.chunks, self.consumed, self.closed = chunks, 0, False

            def __iter__(self):
                for chunk in self.chunks:
                    self.consumed += 1
                    yield chunk

            def close(self):
                self.closed = True

        service, _ = self.service()
        service.chain = Mock()
        service.chain.invoke.return_value = AIMessage(content="계획 완료")
        provider = ProviderStream([AIMessage(content="첫 청크"), AIMessage(content="두 번째")])
        service.final_chain = Mock()
        service.final_chain.stream.return_value = provider
        output = service.stream_with_history([], "조회")
        self.assertEqual(next(output), "첫 청크")
        self.assertEqual(provider.consumed, 1)
        output.close()
        self.assertTrue(provider.closed)
        self.assertEqual(provider.consumed, 1)

        oversized = ProviderStream([
            AIMessage(content="x" * service.MAX_ANSWER_LENGTH),
            AIMessage(content="y"),
            AIMessage(content="unconsumed"),
        ])
        service.final_chain.stream.return_value = oversized
        with self.assertRaisesRegex(ValueError, "too long"):
            list(service.stream_with_history([], "조회"))
        self.assertTrue(oversized.closed)
        self.assertEqual(oversized.consumed, 2)


class DocumentChatCoexistenceTest(APITestCase):
    def test_chat_deletion_preserves_document_and_embedding(self):
        from django.urls import resolve

        self.assertEqual(resolve("/admin/").url_name, "index")
        self.client.force_authenticate(
            user=get_user_model().objects.create_user(username="coexist")
        )
        document = Document.objects.create(title="문서", source="test")
        chunk = DocumentChunk.objects.create(
            document=document, content="원문", chunk_index=0, embedding=[1.0] * 1536
        )
        session = self.client.post("/api/v1/chat/sessions/", {"title": "채팅"}, format="json")
        self.assertEqual(session.status_code, 201)
        self.assertEqual(
            self.client.delete(f"/api/v1/chat/sessions/{session.json()['id']}/").status_code, 204
        )
        chunk.refresh_from_db()
        self.assertEqual(chunk.content, "원문")
        self.assertEqual(len(chunk.embedding), 1536)
        self.assertTrue(Document.objects.filter(pk=document.pk).exists())


class ChatApiTest(APITestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username="tester")
        self.client.force_authenticate(user=self.user)
        self.prompts = []

        def respond(prompt):
            self.prompts.append(prompt.to_messages())
            return AIMessage(content="테스트 답변")

        # 모델만 대체하고 View, ChatService, 히스토리 저장은 실제 실행합니다.
        patcher = patch("llm.v1.chat_service.ChatOpenAI", return_value=RunnableLambda(respond))
        self.model = patcher.start()
        self.addCleanup(patcher.stop)

    def test_session_and_message_crud(self):
        room = self.client.post("/api/v1/chat/sessions/", {"title": "test"}, format="json")
        self.assertEqual(room.status_code, 201)
        session_id = room.json()["id"]

        updated = self.client.patch(
            f"/api/v1/chat/sessions/{session_id}/",
            {"title": "updated"},
            format="json",
        )
        self.assertEqual(updated.json()["title"], "updated")

        message = self.client.post(
            f"/api/v1/chat/sessions/{session_id}/messages/",
            {"content": "hello"},
            format="json",
        )
        self.assertEqual(message.status_code, 201)
        self.assertEqual(message.json()["assistant_message"], "테스트 답변")
        self.assertEqual(
            self.client.get(f"/api/v1/chat/sessions/{session_id}/messages/").json()[0]["content"],
            "hello",
        )

        second = self.client.post(
            f"/api/v1/chat/sessions/{session_id}/messages/", {"content": "두 번째 질문"}, format="json"
        )
        self.assertEqual(second.status_code, 201)
        self.assertEqual(
            [(m.type, m.content) for m in self.prompts[1][1:]],
            [("human", "hello"), ("ai", "테스트 답변"), ("human", "두 번째 질문")],
        )
        messages = self.client.get(f"/api/v1/chat/sessions/{session_id}/messages/").json()
        self.assertEqual([m["role"] for m in messages], ["human", "ai", "human", "ai"])
        self.assertEqual([m["sequence_no"] for m in messages], [1, 2, 3, 4])

        self.assertEqual(self.client.delete(f"/api/v1/chat/sessions/{session_id}/").status_code, 204)
        self.assertFalse(ChatMessage.objects.exists())

    def test_openapi_keeps_json_and_sse_contracts_distinct(self):
        schema = SchemaGenerator().get_schema(request=None, public=True)
        schemas = schema["components"]["schemas"]
        self.assertTrue({
            "ChatSession", "ChatMessage", "ChatFinalize", "ChatFinalizeResponse",
            "ChatNonStreamResponse", "MemberChatEventPayload", "GuestChatEventPayload",
            "ChatDoneEvent", "GuestChatDoneEvent", "GuestChatMessage",
        }.issubset(schemas))
        self.assertEqual(
            "#/components/schemas/GuestChatMessage",
            schemas["GuestChat"]["properties"]["messages"]["items"]["$ref"],
        )
        member = schema["paths"]["/api/v1/chat/sessions/{session_id}/messages/"]["post"]["responses"]
        self.assertEqual(
            "#/components/schemas/MemberChatEventPayload",
            member["200"]["content"]["text/event-stream"]["schema"]["$ref"],
        )
        self.assertEqual(
            "#/components/schemas/ChatNonStreamResponse",
            member["201"]["content"]["application/json"]["schema"]["$ref"],
        )
        self.assertTrue({"places", "coursePayload", "route"}.issubset(schemas["ChatDoneEvent"]["properties"]))

    def test_invalid_input_and_other_users_session_do_not_call_llm(self):
        session = ChatSession.objects.create(user=self.user)
        url = f"/api/v1/chat/sessions/{session.pk}/messages/"
        self.assertEqual(self.client.post(url, {"content": " "}, format="json").status_code, 400)
        self.client.force_authenticate(user=get_user_model().objects.create_user(username="other"))
        self.assertEqual(self.client.get(url).status_code, 404)
        self.assertEqual(self.client.post(url, {"content": "hello"}, format="json").status_code, 404)
        self.client.force_authenticate(user=None)
        self.assertIn(self.client.post(url, {"content": "hello"}).status_code, (401, 403))
        self.model.assert_not_called()

    def test_llm_failure_returns_502_without_saving_messages(self):
        def fail(prompt):
            raise OpenAIError("upstream private error")

        self.model.return_value = RunnableLambda(fail)
        session = ChatSession.objects.create(user=self.user)
        response = self.client.post(
            f"/api/v1/chat/sessions/{session.pk}/messages/", {"content": "hello"}, format="json"
        )
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("upstream private error", response.content.decode())
        self.assertFalse(session.messages.exists())

    def test_missing_llm_credentials_returns_sanitized_502_without_saving_messages(self):
        self.model.side_effect = OpenAIError("private missing API key detail")
        session = ChatSession.objects.create(user=self.user)
        response = self.client.post(
            f"/api/v1/chat/sessions/{session.pk}/messages/", {"content": "hello"}, format="json"
        )
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("private missing API key detail", response.content.decode())
        self.assertFalse(session.messages.exists())

    def test_storage_failure_is_not_reported_as_success(self):
        session = ChatSession.objects.create(user=self.user)
        with patch("llm.v1.chat_message_histories.ChatMessage.objects.bulk_create", side_effect=DatabaseError):
            with self.assertRaises(DatabaseError):
                self.client.post(
                    f"/api/v1/chat/sessions/{session.pk}/messages/", {"content": "hello"}, format="json"
                )
        self.assertFalse(session.messages.exists())

    @staticmethod
    def events(response):
        body = b"".join(response.streaming_content).decode()
        return [
            (frame.splitlines()[0][7:], json.loads(frame.splitlines()[1][6:]))
            for frame in body.strip().split("\n\n")
        ]

    def stream(self, session, question="hello", chunks=("첫 ", "답변")):
        with patch.object(ChatService, "stream_with_history", side_effect=lambda *_: iter(chunks)):
            response = self.client.post(
                f"/api/v1/chat/sessions/{session.pk}/messages/",
                {"content": question}, format="json", HTTP_ACCEPT="text/event-stream",
            )
            return self.events(response)

    def finalize(self, event, prefix, final_status):
        return self.client.post(
            f"/api/v1/chat/turns/{event['turn_id']}/finalize/",
            {"receipt": event["receipt"], "prefix": prefix, "status": final_status},
            format="json",
        )

    @override_settings(CHAT_CHECKPOINT_SIGNING_KEY="test-only-signing-key")
    def test_full_stream_persists_only_after_explicit_finalize_with_ids(self):
        session = ChatSession.objects.create(user=self.user)
        events = self.stream(session)
        self.assertEqual([name for name, _ in events], ["checkpoint", "delta", "delta", "done"])
        self.assertFalse(session.messages.exists())
        result = self.finalize(events[-1][1], "첫 답변", "completed")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["status"], "completed")
        self.assertIsInstance(result.json()["user_message_id"], int)
        self.assertIsInstance(result.json()["assistant_message_id"], int)
        self.assertEqual(
            list(session.messages.values_list("role", "message", "status")),
            [("human", "hello", ""), ("ai", "첫 답변", "completed")],
        )

    @override_settings(CHAT_CHECKPOINT_SIGNING_KEY="test-only-signing-key")
    def test_course_metadata_survives_stream_and_non_stream_responses(self):
        metadata = {
            "places": [{
                "phase": "BEFORE", "name": "식당", "lat": 37.5, "lng": 127.0,
                "category": "식당", "placeId": "place-1", "address": "서울",
                "placeUrl": "https://example.com/place", "distance": 120,
                "reason": "가까움", "time": "15:00", "stayMin": 60,
            }],
            "coursePayload": {
                "title": "직관 코스", "stadium": "잠실", "content": "일정",
                "contentFormat": "", "duration": "약 3시간", "tags": ["직관코스"],
                "startLat": 37.5, "startLng": 127.0,
                "stops": [{
                    "position": 0, "name": "식당", "lat": 37.5, "lng": 127.0,
                    "category": "식당", "placeId": "place-1", "address": "서울",
                    "isMapPoint": True,
                }],
            },
            "route": "course:DOOSAN",
        }
        session = ChatSession.objects.create(user=self.user)
        with patch("llm.v1.views.last_detail", return_value=metadata):
            done = self.stream(session, chunks=("답",))[-1][1]
            non_stream = self.client.post(
                f"/api/v1/chat/sessions/{session.pk}/messages/", {"content": "다른 질문"}, format="json"
            ).json()
        for key in ("places", "coursePayload", "route"):
            self.assertEqual(metadata[key], done[key])
            self.assertEqual(metadata[key], non_stream[key])

    @override_settings(CHAT_CHECKPOINT_SIGNING_KEY="test-only-signing-key")
    def test_stop_exact_prefix_wins_completion_race_and_is_idempotent(self):
        session = ChatSession.objects.create(user=self.user)
        events = self.stream(session, chunks=("받은", "뒷부분"))
        partial, completed = events[1][1], events[-1][1]
        first = self.finalize(completed, "받은뒷부분", "completed").json()
        stopped = self.finalize(partial, "받은", "stopped").json()
        repeated = self.finalize(completed, "받은뒷부분", "completed").json()
        self.assertEqual(stopped["status"], "stopped")
        self.assertEqual(stopped["assistant_message"], "받은")
        self.assertEqual(stopped["user_message_id"], first["user_message_id"])
        self.assertEqual(stopped["assistant_message_id"], first["assistant_message_id"])
        self.assertEqual(repeated, stopped)
        self.assertEqual(session.messages.count(), 2)

    @override_settings(CHAT_CHECKPOINT_SIGNING_KEY="test-only-signing-key")
    def test_stop_before_first_token_saves_human_without_empty_assistant(self):
        session = ChatSession.objects.create(user=self.user)
        checkpoint = self.stream(session, chunks=())[0][1]
        result = self.finalize(checkpoint, "", "stopped").json()
        self.assertEqual(result["status"], "stopped")
        self.assertIsNone(result["assistant_message_id"])
        self.assertEqual(list(session.messages.values_list("role", "message")), [("human", "hello")])

    @override_settings(CHAT_CHECKPOINT_SIGNING_KEY="test-only-signing-key")
    def test_abandoned_pending_turn_does_not_block_new_turn_but_cannot_append_stale_history(self):
        session = ChatSession.objects.create(user=self.user)
        abandoned = self.stream(session, question="abandoned", chunks=("old",))[-1][1]
        current = self.stream(session, question="current", chunks=("new",))[-1][1]
        self.assertEqual(self.finalize(current, "new", "completed").status_code, 200)
        self.assertEqual(self.finalize(abandoned, "old", "completed").status_code, 409)
        self.assertEqual(
            list(session.messages.values_list("role", "message")),
            [("human", "current"), ("ai", "new")],
        )

    @override_settings(CHAT_CHECKPOINT_SIGNING_KEY="test-only-signing-key")
    def test_partial_stop_is_used_by_the_next_question(self):
        session = ChatSession.objects.create(user=self.user)
        partial = self.stream(session, question="첫 질문", chunks=("부분", " 나머지"))[1][1]
        self.finalize(partial, "부분", "stopped")
        response = self.client.post(
            f"/api/v1/chat/sessions/{session.pk}/messages/", {"content": "후속 질문"}, format="json"
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(
            [(m.type, m.content) for m in self.prompts[-1][1:]],
            [("human", "첫 질문"), ("ai", "부분"), ("human", "후속 질문")],
        )

    @override_settings(CHAT_CHECKPOINT_SIGNING_KEY="test-only-signing-key")
    def test_finalize_rejects_foreign_owner_and_forged_prefix(self):
        session = ChatSession.objects.create(user=self.user)
        checkpoint = self.stream(session, chunks=("서버",))[1][1]
        forged = self.finalize(checkpoint, "클라이언트 위조", "stopped")
        self.assertEqual(forged.status_code, 400)
        invalid = self.client.post(
            f"/api/v1/chat/turns/{checkpoint['turn_id']}/finalize/",
            {"receipt": "not-a-server-receipt", "prefix": "서버", "status": "stopped"},
            format="json",
        )
        self.assertEqual(invalid.status_code, 400)
        self.client.force_authenticate(user=get_user_model().objects.create_user(username="other"))
        foreign = self.finalize(checkpoint, "서버", "stopped")
        self.assertEqual(foreign.status_code, 404)
        self.assertFalse(session.messages.exists())

    @override_settings(CHAT_CHECKPOINT_SIGNING_KEY="test-only-signing-key")
    def test_completed_turn_cannot_be_rewritten_after_a_later_message(self):
        session = ChatSession.objects.create(user=self.user)
        events = self.stream(session, chunks=("앞", "뒤"))
        self.finalize(events[-1][1], "앞뒤", "completed")
        self.client.post(
            f"/api/v1/chat/sessions/{session.pk}/messages/", {"content": "later"}, format="json"
        )
        result = self.finalize(events[1][1], "앞", "stopped").json()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["assistant_message"], "앞뒤")

    @override_settings(CHAT_CHECKPOINT_SIGNING_KEY="test-only-signing-key")
    def test_completed_turn_cannot_be_rewritten_after_a_newer_turn_starts(self):
        session = ChatSession.objects.create(user=self.user)
        first = self.stream(session, question="first", chunks=("kept", " tail"))
        self.finalize(first[-1][1], "kept tail", "completed")
        second = self.stream(session, question="second", chunks=("next",))
        late_stop = self.finalize(first[1][1], "kept", "stopped").json()
        self.assertEqual(late_stop["status"], "completed")
        self.assertEqual(late_stop["assistant_message"], "kept tail")
        self.assertEqual(self.finalize(second[-1][1], "next", "completed").status_code, 200)
        self.assertEqual(
            list(session.messages.values_list("role", "message")),
            [("human", "first"), ("ai", "kept tail"), ("human", "second"), ("ai", "next")],
        )

    @override_settings(CHAT_CHECKPOINT_SIGNING_KEY="test-only-signing-key")
    def test_stream_failure_is_sanitized_and_saves_no_messages(self):
        def fail(*_):
            yield "부분"
            raise OpenAIError("private provider detail")

        session = ChatSession.objects.create(user=self.user)
        with patch.object(ChatService, "stream_with_history", side_effect=fail):
            response = self.client.post(
                f"/api/v1/chat/sessions/{session.pk}/messages/",
                {"content": "hello"}, format="json", HTTP_ACCEPT="text/event-stream",
            )
            body = b"".join(response.streaming_content).decode()
        self.assertIn("event: error", body)
        self.assertNotIn("private provider detail", body)
        self.assertFalse(session.messages.exists())

    def test_guest_questions_are_read_only_and_never_reach_the_model(self):
        cache.clear()
        before = (ChatSession.objects.count(), ChatMessage.objects.count(), ChatTurn.objects.count())
        guest = APIClient()
        with patch.object(ChatService, "stream_with_history", return_value=iter(("답",))) as model:
            response = guest.post(
                "/api/v1/chat/guest/",
                {"messages": [
                    {"role": "user", "content": "첫 질문"},
                    {"role": "assistant", "content": "부분 답"},
                    {"role": "user", "content": "후속 질문"},
                ]},
                format="json", HTTP_ACCEPT="text/event-stream", REMOTE_ADDR="203.0.113.9",
            )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(model.call_count, 0)
        self.assertEqual(before, (ChatSession.objects.count(), ChatMessage.objects.count(), ChatTurn.objects.count()))
