import json
from unittest.mock import Mock, patch

from django.test import SimpleTestCase
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from ..v1.chat_service import ChatService
from ..v1.rag import dispatcher
from ..v1.rag import domain_tools
from ..v1.rag.assistant import pipeline as assistant_pipeline
from ..v1.rag.club import agent as club
from ..v1.rag.course import agent as course
from ..v1.rag.nearby import agent as nearby
from ..v1.rag.assistant import tools as assistant_tools
from ..v1.rag.venue import agent as venue
from ..v1.tools import DOMAIN_TOOL_NAMES


EXPECTED = set(DOMAIN_TOOL_NAMES) | {
    "get_baseball_schema", "execute_baseball_select", "get_ticket_policy", "search_kbo_documents",
    "search_nearby_places", "plan_course", "search_documents_tool",
}


class FakeTool:
    def __init__(self, name="search_players", result=None):
        self.name = name
        self.result = result or {"items": [{"name": "도구 최신 선수"}]}

    def invoke(self, arguments):
        return {**self.result, "arguments": arguments}


class ToolCallingModel:
    def __init__(self, tool_name="search_players", arguments=None):
        self.bound_names = ()
        self.tool_name = tool_name
        self.arguments = arguments or {"team_code": "LG"}

    def bind_tools(self, tools, **_kwargs):
        self.bound_names = tuple(tool.name for tool in tools)
        return self

    def invoke(self, messages):
        result = next((message for message in messages if isinstance(message, ToolMessage)), None)
        if result is None:
            return AIMessage(content="", tool_calls=[{
                "name": self.tool_name, "args": self.arguments, "id": "tool-1",
            }])
        return AIMessage(content=f"도구 답변: {result.content}")


class DomainAllowlistTest(SimpleTestCase):
    def test_every_answer_agent_gets_the_same_complete_inventory(self):
        inventories = {
            domain: tuple(tool.name for tool in domain_tools.tools_for(domain))
            for domain in ("assistant", "club", "course", "venue", "nearby")
        }
        model = RunnableLambda(lambda _: AIMessage(content="ok"))
        model.bind_tools = Mock(return_value=model)
        inventories["chat"] = tuple(tool.name for tool in ChatService(llm=model).tools)
        self.assertEqual(set(inventories.values()), {inventories["assistant"]})
        self.assertEqual(set(inventories["assistant"]), EXPECTED)
        self.assertEqual(len(inventories["assistant"]), len(EXPECTED))
        self.assertIs(assistant_tools.build_tools()[0].func, assistant_tools.get_games)
        with self.assertRaisesRegex(ValueError, "not allowed"):
            domain_tools.invoke("venue", "not_registered", {})

    def test_answer_entrypoints_use_fresh_state_and_restore_parent(self):
        parent = assistant_tools.new_state("STALE", "old", [{"role": "user", "content": "old"}])
        parent["schema_seen"] = True
        expected_history = [{"role": "user", "content": "fresh history"}]

        def inspect(*_args, **_kwargs):
            current = assistant_tools.state()
            self.assertEqual((current["hint"], current["question"], current["history"]),
                             ("JAMSIL", "fresh", expected_history))
            self.assertFalse(current["schema_seen"])
            return {"answer": "ok"}

        with patch.object(assistant_pipeline, "_answer", side_effect=inspect):
            self.assertEqual(assistant_pipeline.answer("fresh", expected_history, "JAMSIL")["answer"], "ok")
        for module in (club, course, venue, nearby):
            with self.subTest(module=module.__name__), patch.object(module, "_answer", side_effect=inspect):
                self.assertEqual(module.answer("fresh", expected_history, "JAMSIL")["answer"], "ok")

        def inspect_chat():
            current = assistant_tools.state()
            self.assertEqual((current["hint"], current["question"], current["history"]),
                             (None, "fresh", expected_history))
            self.assertFalse(current["schema_seen"])
            return {"answer": "ok"}

        service = ChatService.__new__(ChatService)
        with patch.object(service, "_run_scoped", side_effect=lambda _values: inspect_chat()):
            self.assertEqual(service._run({"question": "fresh", "chat_history": [HumanMessage(content="fresh history")]}),
                             {"answer": "ok"})
        self.assertIs(assistant_tools.state(), parent)

    def test_streaming_uses_full_inventory_visible_text_and_restores_state(self):
        parent = assistant_tools.new_state("STALE", "old", [])
        history = [{"role": "user", "content": "fresh history"}]

        class StreamingModel:
            bound_names = ()

            def bind_tools(self, tools):
                self.bound_names = tuple(tool.name for tool in tools)
                return self

            def invoke(self, _messages, config=None):
                current = assistant_tools.state()
                self.assert_state(current)
                return AIMessage(content="READY")

            def stream(self, _messages, config=None):
                current = assistant_tools.state()
                self.assert_state(current)
                yield AIMessage(content=[
                    {"type": "reasoning", "text": "비공개 추론"},
                    {"type": "output_text", "text": "보이는 답변"},
                ])

            @staticmethod
            def assert_state(current):
                assert (current["hint"], current["question"], current["history"]) == (
                    "JAMSIL", "fresh", history,
                )

        model = StreamingModel()
        stream = assistant_pipeline.stream_answer(
            "fresh", history, "JAMSIL", model=model,
            retriever=lambda values: {**values, "context": "", "doc_count": 0, "stadium": "JAMSIL"},
        )
        self.assertEqual("".join(stream), "보이는 답변")
        self.assertEqual(set(model.bound_names), EXPECTED)
        self.assertIs(assistant_tools.state(), parent)

        interrupted = assistant_pipeline.stream_answer(
            "fresh", history, "JAMSIL", model=StreamingModel(),
            retriever=lambda values: {**values, "context": "", "doc_count": 0, "stadium": "JAMSIL"},
        )
        self.assertEqual(next(interrupted), "보이는 답변")
        interrupted.close()
        self.assertIs(assistant_tools.state(), parent)

    def test_weather_routes_only_with_explicit_course_context(self):
        self.assertEqual(dispatcher.route("날씨 알려줘"), "scope")
        self.assertEqual(dispatcher.route("잠실 직관 코스와 날씨를 알려줘", intent="route"), "course")

    def test_dispatcher_club_executes_bounded_tool_loop_and_consumes_result(self):
        model = ToolCallingModel()
        with (
            # 에이전트 파이프라인 도입(#30) 후 도메인 라우팅은 fallback 경로다 — 실패시켜서 그 경로를 검증한다
            patch.object(dispatcher.assistant, "answer", side_effect=RuntimeError("agent down")),
            patch.object(domain_tools, "tools_for", return_value=(FakeTool(),)),
            patch.object(club, "llm", return_value=model),
            patch.object(club, "embed", return_value=[0.0]),
            patch.object(club, "search", return_value=([], 0.0)),
            patch.object(club, "keyword_rerank", side_effect=lambda _q, rows, k: rows[:k]),
        ):
            result = dispatcher.answer("LG 선수 알려줘")
        self.assertEqual(model.bound_names, ("search_players",))
        self.assertIn("도구 최신 선수", result["answer"])
        self.assertTrue(result["route"].startswith("agent:error>club>"))

    def test_club_structured_answer_prefers_canonical_tool_result_over_old_rag(self):
        current = {"actual_date": "2026-09-16", "items": [{
            "team__team_code": "LG", "rank": 1, "wins": 80, "losses": 40, "draws": 2, "games_behind": "0",
        }]}
        old = [{"team": "LG", "rank": 10, "games": 122, "win": 40, "lose": 80, "draw": 2,
                "rate": "0.333", "gb": "40", "as_of": "2026-09-01"}]
        with patch.object(club, "invoke_domain_tool", return_value=current), patch.object(club.structured, "standings", return_value=old):
            result = club.answer("LG 순위 알려줘")
        self.assertIn("1위", result["answer"])
        self.assertNotIn("10위", result["answer"])

    def test_venue_agent_keeps_rag_tool_with_all_registered_tools(self):
        seen = {}
        fake_agent = object()

        def create_agent(*, model, tools, system_prompt):
            seen["names"] = {tool.name for tool in tools}
            return fake_agent

        previous = venue._agent
        venue._agent = None
        try:
            with patch.object(venue, "create_agent", side_effect=create_agent), patch.object(venue, "llm", return_value=object()):
                self.assertIs(venue.agent(), fake_agent)
        finally:
            venue._agent = previous
        self.assertEqual(seen["names"], EXPECTED)

    def test_venue_answer_consumes_new_tool_result(self):
        class FakeAgent:
            def __init__(self, tools):
                self.tool = next(tool for tool in tools if tool.name == "get_stadium")

            def invoke(self, _input, config=None):
                result = self.tool.invoke({"stadium_code": "JAMSIL"})
                return {"messages": [
                    ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id="stadium-1", name="get_stadium"),
                    AIMessage(content=f"도구 구장: {result['item']['stadium_name_ko']}"),
                ]}

        tool = FakeTool("get_stadium", {"item": {"stadium_name_ko": "도구 잠실"}})
        previous = venue._agent
        venue._agent = None
        try:
            with (
                patch.object(venue, "tools_for", return_value=(tool,)),
                patch.object(venue, "llm", return_value=object()),
                patch.object(venue, "create_agent", side_effect=lambda **kwargs: FakeAgent(kwargs["tools"])),
            ):
                result = venue.answer("잠실 주소 알려줘")
        finally:
            venue._agent = previous
        self.assertIn("도구 잠실", result["answer"])
        self.assertIn("agent:tool", result["route"])

    def test_course_dispatch_uses_canonical_schedule_and_keeps_response_contract(self):
        calls = []

        def invoke(_domain, name, arguments):
            calls.append((name, arguments))
            if name == "get_games":
                return {"items": [{
                    "game_date": "2026-09-20", "game_time": "18:30", "stadium__stadium_name_ko": "잠실야구장",
                    "home_team__team_name_ko": "LG 트윈스", "away_team__team_name_ko": "두산 베어스", "status_code": "scheduled",
                }]}
            if name == "search_places":
                category = arguments["category"]
                return {"places": [{
                    "id": category, "place_name": "최신 맛집" if category == "FD6" else "최신 카페",
                    "x": "127.076", "y": "37.517", "category_name": "음식점" if category == "FD6" else "카페",
                    "road_address_name": "서울", "address_name": "서울",
                }]}
            if name == "search_tourism":
                return {"status": "ok", "places": [], "truncated": False}
            if name == "get_directions":
                return {"distance": 500, "seconds": 420, "legs": [{"seconds": 420}]}
            raise AssertionError(name)

        anchor = {"key": "STADIUM", "phase": "GAME", "name": "잠실야구장", "lat": 37.516, "lng": 127.075,
                  "category": "STADIUM", "detail": "", "placeId": None, "address": "서울", "placeUrl": "",
                  "distance": 0, "doc_id": "stadium:JAMSIL"}
        selection = json.dumps({"intro": "최신 도구 자료 기준", "course": [
            {"place_key": "P1", "phase": "BEFORE", "reason": "식사"},
            {"place_key": "STADIUM", "phase": "GAME", "reason": "관람"},
        ]}, ensure_ascii=False)
        select = Mock(return_value=(selection, 1.0))
        with (
            patch.object(course, "invoke_domain_tool", side_effect=invoke),
            patch.object(course, "stadium_anchor", return_value=anchor),
            patch.object(course, "embed_many", return_value=([0.0], [0.0])),
            patch.object(course, "search_places", return_value=[]),
            patch.object(course.structured, "games", side_effect=AssertionError("old RAG schedule used")),
            patch.object(course, "call_llm", select),
        ):
            result = dispatcher.answer("9월 20일 잠실 코스 짜줘", intent="route")
        self.assertIn("LG 트윈스 홈 vs 두산 베어스 원정", select.call_args.args[1])
        self.assertEqual(result["places"][0]["name"], "최신 맛집")
        self.assertIsNotNone(result["coursePayload"])
        self.assertEqual({name for name, _ in calls}, {"get_games", "search_places", "search_tourism", "get_directions"})

    def test_course_model_binding_is_bounded_and_consumes_tool_result(self):
        model = ToolCallingModel("get_weather", {"stadium_code": "JAMSIL"})
        tool = FakeTool("get_weather", {"label": "맑음"})
        with patch.object(domain_tools, "tools_for", return_value=(tool,)):
            response = domain_tools.run_model(model, [], "course", max_tool_rounds=1)
        self.assertEqual(model.bound_names, ("get_weather",))
        self.assertIn("맑음", response.content)

    def test_cross_domain_tool_executes_and_unknown_call_is_rejected(self):
        cross_domain = ToolCallingModel("get_weather", {"stadium_code": "JAMSIL"})
        weather = FakeTool("get_weather", {"label": "맑음"})
        with patch.object(domain_tools, "tools_for", return_value=(weather,)):
            response = domain_tools.run_model(cross_domain, [], "club", max_tool_rounds=1)
        self.assertIn("맑음", response.content)

        unknown = ToolCallingModel("not_registered", {})
        with patch.object(domain_tools, "tools_for", return_value=(weather,)):
            response = domain_tools.run_model(unknown, [], "course", max_tool_rounds=1)
        self.assertIn("허용되지 않은 도구", response.content)

    def test_nearby_executes_cross_domain_tool_and_consumes_result(self):
        model = ToolCallingModel("get_weather", {"stadium_code": "JAMSIL"})
        weather = FakeTool("get_weather", {"label": "주변 분기에서 확인한 맑음"})
        places = [{
            "kind": "stay", "kindLabel": "숙박", "name": "테스트 호텔", "detail": "여행 > 숙박 > 호텔",
            "distance": 500, "lat": 37.5, "lng": 127.0, "address": "서울", "placeId": "1",
            "placeUrl": "", "phone": "",
        }]
        with (
            patch.object(domain_tools, "tools_for", return_value=(weather,)),
            patch.object(nearby, "llm", return_value=model),
            patch.object(nearby.kakao, "enabled", return_value=True),
            patch.object(nearby.kakao, "nearby", return_value=places),
        ):
            result = nearby.answer("잠실 숙박 추천해줘")
        self.assertEqual(model.bound_names, ("get_weather",))
        self.assertIn("주변 분기에서 확인한 맑음", result["answer"])
        self.assertEqual(result["sources"][0]["doc_id"], "kakao:1")

    def test_recursive_course_tool_is_bounded_and_context_is_reset(self):
        model = ToolCallingModel("plan_course", {"request": "잠실 코스"})
        response = domain_tools.run_model(model, [], "course", max_tool_rounds=1)
        self.assertIn("plan_course를 다시 호출할 수 없습니다", response.content)
        self.assertIsNone(domain_tools.active_domain())

    def test_document_search_rewriter_stays_a_helper_call(self):
        class Transformer:
            calls = 0

            def invoke(self, values):
                self.calls += 1
                return values["query"]

        transformer = Transformer()
        previous = venue._transformer
        venue._transformer = transformer
        try:
            model = ToolCallingModel("search_documents_tool", {"query": "잠실 포토존"})
            with (
                patch.object(venue, "vector_search", return_value=[]),
                patch.object(venue, "keyword_fallback_search", return_value=[]),
            ):
                response = domain_tools.run_model(model, [], "club", max_tool_rounds=1)
        finally:
            venue._transformer = previous
        self.assertEqual(transformer.calls, 1)
        self.assertIn("관련 문서를 찾을 수 없습니다", response.content)
        self.assertIsNone(domain_tools.active_domain())

    def test_followup_club_and_course_tool_results_are_consumed(self):
        cases = (
            ("club", "search_community_posts"), ("club", "get_prediction_games"),
            ("course", "get_stadium"), ("course", "get_stadium_contents"),
            ("course", "search_courses"), ("course", "get_course"),
        )
        for domain, name in cases:
            with self.subTest(domain=domain, tool=name):
                model = ToolCallingModel(name, {"test": True})
                tool = FakeTool(name, {"marker": f"{domain}:{name}"})
                with patch.object(domain_tools, "tools_for", return_value=(tool,)):
                    response = domain_tools.run_model(model, [], domain, max_tool_rounds=1)
                self.assertIn(f"{domain}:{name}", response.content)

    def test_current_dispatcher_routes_remain_unchanged(self):
        self.assertEqual(dispatcher.route("잠실 포토존 어디야"), "venue")
        self.assertEqual(dispatcher.route("잠실 직관 코스 짜줘", intent="route"), "course")
        self.assertEqual(dispatcher.route("LG 팬 투표 승부예측 알려줘"), "club")

    def test_saved_public_course_lookup_bypasses_itinerary_slot_guard(self):
        model = ToolCallingModel("search_courses", {"query": "잠실"})
        tool = FakeTool("search_courses", {"items": [{"title": "공개 저장 코스"}]})
        with (
            # 도메인 fallback 경로 검증 — 에이전트 파이프라인(#30)을 실패시킨다
            patch.object(dispatcher.assistant, "answer", side_effect=RuntimeError("agent down")),
            patch.object(domain_tools, "tools_for", return_value=(tool,)),
            patch.object(course, "llm", return_value=model),
        ):
            result = dispatcher.answer("저장된 공개 코스 찾아줘")
        self.assertIn("공개 저장 코스", result["answer"])
        self.assertEqual(result["route"], "agent:error>course>course:public_lookup")
        self.assertEqual(result["places"], [])
        self.assertIsNone(result["coursePayload"])

        course_id = "12345678-1234-4123-8123-123456789abc"
        detail_model = ToolCallingModel("get_course", {"course_id": course_id})
        detail_tool = FakeTool("get_course", {"item": {"id": course_id, "title": "공개 상세 코스"}})
        with (
            patch.object(dispatcher.assistant, "answer", side_effect=RuntimeError("agent down")),
            patch.object(domain_tools, "tools_for", return_value=(detail_tool,)),
            patch.object(course, "llm", return_value=detail_model),
        ):
            detail = dispatcher.answer(f"{course_id} 코스 상세 알려줘")
        self.assertIn("공개 상세 코스", detail["answer"])
        self.assertEqual(detail["route"], "agent:error>course>course:public_lookup")

    def test_prediction_question_bypasses_schedule_shortcut_and_consumes_fan_vote_tool(self):
        model = ToolCallingModel("get_prediction_games", {"game_date": "2026-09-16", "team_code": "LG"})
        tool = FakeTool("get_prediction_games", {
            "items": [{"fan_votes": {"home": 7, "away": 3, "total": 10},
                       "fan_vote_notice": "실제 승리 확률이 아닌 팬 투표"}],
        })
        with (
            # 도메인 fallback 경로 검증 — 에이전트 파이프라인(#30)을 실패시킨다
            patch.object(dispatcher.assistant, "answer", side_effect=RuntimeError("agent down")),
            patch.object(domain_tools, "tools_for", return_value=(tool,)),
            patch.object(club, "llm", return_value=model),
            patch.object(club.structured, "answer", side_effect=AssertionError("schedule shortcut used")),
            patch.object(club, "embed", return_value=[0.0]),
            patch.object(club, "search", return_value=([], 0.0)),
            patch.object(club, "keyword_rerank", side_effect=lambda _q, rows, k: rows[:k]),
        ):
            result = dispatcher.answer("오늘 LG 경기 팬 투표 현황 알려줘")
        self.assertIn("실제 승리 확률이 아닌 팬 투표", result["answer"])
        self.assertTrue(result["route"].startswith("agent:error>club>"))
