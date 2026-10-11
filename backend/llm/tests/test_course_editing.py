from copy import deepcopy
from unittest.mock import patch

from django.test import SimpleTestCase
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rest_framework.exceptions import ValidationError

from llm.serializer.message import _validate_context
from llm.tools import assistant
from llm.v1.rag.course import agent, editing
from llm.v2.agent.course_output import public_course
from llm.v2.middleware.dynamic_tools import DynamicToolMiddleware


PLACES = [
    {"visitId": "food", "label": "출발", "name": "기존 식당", "lat": 37.511, "lng": 127.08, "placeId": "1", "category": "FOOD", "phase": "BEFORE"},
    {"visitId": "cafe", "label": "1", "name": "기존 카페", "lat": 37.512, "lng": 127.079, "placeId": "2", "category": "CAFE", "phase": "BEFORE"},
    {"visitId": "game", "label": "2", "name": "잠실야구장", "lat": 37.512, "lng": 127.071, "placeId": "3", "category": "STADIUM", "phase": "GAME"},
    {"visitId": "park", "label": "3", "name": "아시아공원", "lat": 37.509, "lng": 127.075, "placeId": "4", "category": "WALK", "phase": "AFTER"},
]
CURRENT = {"places": PLACES, "stadiumCode": "JAMSIL", "travelMode": "walk", "legModes": {}, "game": {"date": "2026-10-06", "time": "18:30"}}
GAME = {"date": "2026-10-06", "time": "18:30", "home": "LG", "away": "두산"}
REPLACEMENT = {"name": "메가MGC커피 새 지점", "lat": 37.513, "lng": 127.079, "placeId": "99", "category": "CAFE", "reason": "메가MGC커피 브랜드 확인"}


def plan(op, targets, **kw):
    return {"operation": op, "targets": targets, "reference": "", "position": "before", "query": "", "conditions": [], "clarification": "", **kw}


class CourseEditingTest(SimpleTestCase):
    def test_replacement_keeps_all_other_places_and_original_input(self):
        before = deepcopy(PLACES)
        result = editing.mutate(PLACES, plan("replace", ["cafe"]), REPLACEMENT)
        self.assertEqual(PLACES, before)
        self.assertEqual([result[i] for i in (0, 2, 3)], [before[i] for i in (0, 2, 3)])
        self.assertEqual(result[1]["placeId"], "99")
        self.assertEqual(result[1]["visitId"], "cafe")

    def test_swap_and_move_never_replace_or_add_places(self):
        for operation, expected in ((plan("swap", ["food", "cafe"]), ["cafe", "food", "game", "park"]),
                                    (plan("move", ["cafe"], reference="park", position="after"), ["food", "game", "park", "cafe"]),
                                    (plan("move", ["park"], position="first"), ["park", "food", "cafe", "game"]),
                                    (plan("reorder", ["park", "cafe", "food"]), ["park", "cafe", "game", "food"])):
            result = editing.mutate(PLACES, operation)
            self.assertEqual([p["visitId"] for p in result], expected)
            self.assertEqual(sorted(result, key=lambda p: p["visitId"]), sorted(PLACES, key=lambda p: p["visitId"]))

    def test_invalid_or_stadium_replacement_is_atomic(self):
        for operation in (plan("replace", ["game"]), plan("swap", ["missing", "cafe"]),
                          plan("swap", ["cafe", "cafe"]), plan("move", ["cafe"], reference="cafe")):
            before = deepcopy(PLACES)
            with self.assertRaises(ValueError):
                editing.mutate(PLACES, operation, REPLACEMENT)
            self.assertEqual(PLACES, before)

    def test_ambiguous_and_unmatched_edits_keep_course_without_regenerating(self):
        for operation in (plan("clarify", [], clarification="경기 전 카페와 경기 후 카페 중 어디를 바꿀까요?"),
                          plan("replace", ["cafe"], conditions=["조용함"])):
            with patch.object(editing, "interpret", return_value=operation), patch.object(editing, "candidates", return_value=[]), \
                    patch.object(agent, "stadium_anchor", return_value=PLACES[2]), patch.object(agent, "_answer") as generate:
                result = agent.answer("카페 바꿔줘", hint_stadium="JAMSIL", current_course=CURRENT)
            self.assertEqual(result["places"], [])
            self.assertIn("기존 코스는 그대로", result["answer"])
            generate.assert_not_called()

    def test_only_explicit_new_course_or_new_stadium_can_use_generation(self):
        with patch.object(editing, "interpret", return_value=plan("new", [])), patch.object(agent, "_answer", return_value={"new": True}) as generate:
            self.assertTrue(agent.answer("처음부터 새로 짜줘", hint_stadium="JAMSIL", current_course=CURRENT)["new"])
            generate.assert_called_once()
        with patch.object(editing, "interpret", return_value=plan("new", [])), patch.object(editing, "answer") as edit, patch.object(agent, "_answer", return_value={}) as generate:
            agent.answer("사직으로 바꿔줘", hint_stadium="SAJIK", current_course=CURRENT)
            edit.assert_not_called()
            generate.assert_called_once()

    def test_existing_candidates_with_unverified_menu_are_not_reported_as_missing_shops(self):
        operation = plan("add", [], category="FOOD", reference="cafe", conditions=["자장면"], query="중식집")
        with patch.object(editing, "interpret", return_value=operation), patch.object(agent, "stadium_anchor", return_value=PLACES[2]), \
                patch.object(editing, "candidates", return_value=[{**PLACES[0], "placeId": "new"}]), patch.object(editing, "choose", return_value=None):
            result = agent.answer("카페 전에 자장면 먹을 중식집 추가", hint_stadium="JAMSIL", current_course=CURRENT)
        self.assertIn("검증 후 남은 1곳의 후보", result["answer"])
        self.assertIn("자장면", result["answer"])
        self.assertNotIn("브랜드나 업종을 알려", result["answer"])
        self.assertEqual(result["places"], [])

    def test_rebuild_keeps_game_day_updates_phases_and_preserves_unchanged_leg_mode(self):
        current = deepcopy(CURRENT)
        key = editing.leg_key(PLACES[2], PLACES[3])
        current["legModes"] = {key: "transit", editing.leg_key(PLACES[0], PLACES[1]): "car"}
        places = editing.mutate(PLACES, plan("swap", ["food", "cafe"]))
        actual = {"legs": [{"status": "ok", "distance": 400, "seconds": 301}]}
        with patch.object(agent, "invoke_domain_tool", return_value=actual) as directions, \
                patch.object(agent, "load_schedule", return_value=({}, 1)) as schedule, \
                patch.object(agent, "find_game", return_value=(GAME, False, [])):
            result = editing.rebuild(places, current, "JAMSIL", None, "순서 바꿔줘", [])
        self.assertEqual(schedule.call_args.args[1], "2026-10-06")
        self.assertEqual([c.args[2]["mode"] for c in directions.call_args_list], ["walk", "walk", "transit"])
        self.assertEqual(result["legModes"], {key: "transit"})
        self.assertEqual([p["time"] for p in result["places"]], ["16:28", "17:14", "18:10", "21:46"])
        self.assertEqual(result["game"], CURRENT["game"])
        self.assertIn("16분", result["travel"]["summary"])  # ceil(301 * 3 / 60), 지도 전체 합계와 같은 반올림
        self.assertEqual([p["placeId"] for p in result["places"]], ["2", "1", "3", "4"])
        self.assertEqual([p["visitId"] for p in public_course(result)["places"]], ["cafe", "food", "game", "park"])

    def test_crossing_game_boundary_updates_phase_without_inserting_any_stop(self):
        places = editing.mutate(PLACES, plan("move", ["cafe"], position="last"))
        with patch.object(editing, "route_legs", return_value=[{"minutes": 5, "meters": 400, "by": "walk"}] * 3), \
                patch.object(agent, "load_schedule", return_value=({}, 1)), patch.object(agent, "find_game", return_value=(GAME, False, [])):
            result = editing.rebuild(places, CURRENT, "JAMSIL", None, "카페 맨 뒤로", [])
        self.assertEqual([p["phase"] for p in result["places"]], ["BEFORE", "GAME", "AFTER", "AFTER"])
        self.assertEqual(len(result["places"]), 4)

    def test_explicit_date_edit_overrides_saved_game_and_recalculates_times(self):
        changed_game = {**GAME, "date": "2026-10-15", "time": "17:00"}
        before = deepcopy(CURRENT)
        with patch.object(editing, "route_legs", return_value=[{"minutes": 5, "meters": 400, "by": "walk"}] * 3), \
                patch.object(agent, "load_schedule", return_value=({}, 1)) as schedule, \
                patch.object(agent, "find_game", return_value=(changed_game, False, [])):
            result = editing.answer("10월 15일로 바꿔줘", [], CURRENT, "JAMSIL", None, parsed=plan("date", []))
        self.assertEqual(schedule.call_args.args[1], "2026-10-15")
        self.assertEqual(result["game"], {"date": "2026-10-15", "time": "17:00"})
        self.assertEqual(next(p for p in result["places"] if p["category"] == "STADIUM")["time"], "16:40")
        self.assertEqual([p["placeId"] for p in result["places"]], [p["placeId"] for p in CURRENT["places"]])
        self.assertEqual(CURRENT, before)

    def test_requested_date_with_no_game_preserves_course_without_old_game_fallback(self):
        with patch.object(editing, "route_legs", return_value=[{"minutes": 5, "meters": 400, "by": "walk"}] * 3), \
                patch.object(agent, "load_schedule", return_value=({"items": []}, 1)), \
                patch.object(agent, "find_game", return_value=(None, False, [])):
            result = editing.answer("10월 15일로 맞춰줘", [], CURRENT, "JAMSIL", None, parsed=plan("date", []))
        self.assertEqual(result["places"], [])
        self.assertNotIn("game", result)
        self.assertIn("다른 날짜로 바꾸지 않고", result["answer"])

    def test_combined_date_and_place_edit_searches_availability_on_requested_day(self):
        dates = []
        def candidates(*args):
            dates.append(editing.availability.visit_date())
            return []
        with editing.availability.session(), patch.object(agent, "stadium_anchor", return_value=PLACES[2]), patch.object(editing, "candidates", side_effect=candidates):
            result = editing.answer("2026-10-15로 바꾸고 카페 교체", [], CURRENT, "JAMSIL", None,
                                    parsed=plan("replace", ["cafe"], category="CAFE"))
        self.assertEqual(dates, ["2026-10-15"])
        self.assertEqual(result["places"], [])

    def test_manual_course_without_stadium_has_no_invented_game_or_absolute_time(self):
        with patch.object(editing, "route_legs", return_value=[{"minutes": 8, "meters": 600, "by": "walk"}]), patch.object(agent, "load_schedule") as games:
            result = editing.rebuild(deepcopy(PLACES[:2]), CURRENT, "JAMSIL", None, "순서", [])
        games.assert_not_called()
        self.assertNotIn("game", result)
        self.assertTrue(all("time" not in p for p in result["places"]))
        self.assertEqual(len(result["places"]), 2)
        self.assertIn("약 8분", result["answer"])

    def test_candidate_search_excludes_existing_places_and_outside_radius(self):
        def raw(identifier, lat=37.512, name="다른 카페", cat="CE7"):
            return {"id": identifier, "place_name": name, "x": "127.079", "y": str(lat), "category_group_code": cat}
        payload = {"places": [raw("2"), raw("10", 37.8), raw("11", cat="FD6"), raw("12"), raw("13", float("nan"))]}
        with patch.object(agent, "invoke_domain_tool", return_value=payload):
            result = editing.candidates(PLACES[1], PLACES[2], plan("replace", ["cafe"]), PLACES)
        self.assertEqual([p["placeId"] for p in result], ["12"])

    def test_frontend_context_is_validated_and_bounded(self):
        value = _validate_context({"currentCourse": CURRENT})["currentCourse"]
        self.assertEqual(value, CURRENT)
        for bad in ({**CURRENT, "places": PLACES * 4}, {**CURRENT, "places": [PLACES[0]] * 2},
                    {**CURRENT, "places": [{**PLACES[0], "lat": True}]}, {**CURRENT, "travelMode": []},
                    {**CURRENT, "game": {"date": "2026-02-31", "time": "18:30"}}, {**CURRENT, "legModes": {"oops": "walk"}}):
            with self.assertRaises(ValidationError):
                _validate_context({"currentCourse": bad})

    def test_tool_receives_authoritative_snapshot(self):
        class FakeCourse:
            @staticmethod
            def answer(question, **kwargs):
                self.assertEqual(kwargs["current_course"], CURRENT)
                return {"answer": "변경", "places": PLACES, "edit": True}
        with assistant.request_state("JAMSIL", "카페만 바꿔", [], current_course=CURRENT):
            self.assertEqual(assistant.plan_course("엉뚱한 요약", _course=FakeCourse), "변경")

    def test_edit_destination_uses_confirmed_course_not_screen_default(self):
        from types import SimpleNamespace
        for current, saved in ((CURRENT, {}), (None, {"current": CURRENT, "stadiumCode": "JAMSIL"})):
            for question, mode, expected in (("카페만 바꿔줘", "EDIT", "JAMSIL"),
                                             ("고척으로 바꿔줘", "EDIT", "GOCHEOK"),
                                             ("새로 짜줘", "NEW", "SAJIK")):
                with self.subTest(question=question, current=bool(current)), assistant.request_state(
                        "SAJIK", question, [], current_course=current) as state, patch.object(
                        editing, "interpret", return_value=plan("clarify", [], clarification="확인")) as interpret, patch.object(
                        editing, "answer", return_value={"answer": "확인", "places": []}), patch.object(
                        agent, "_answer", return_value={"answer": "새 코스", "places": []}) as generate:
                    state.update(course_request=mode, course_memory=deepcopy(saved))
                    assistant.plan_course("사직 카페 변경", _course=agent)
                    self.assertEqual(interpret.call_args.args[1]["stadiumCode"], expected)
                    if mode == "EDIT" and expected == "JAMSIL":
                        self.assertEqual(interpret.call_args.args[1]["places"], PLACES)
                        generate.assert_not_called()
                    if mode == "NEW":
                        self.assertEqual(interpret.call_args.args[1]["places"], [])
                        self.assertEqual(interpret.call_args.args[2], [])
                        generate.assert_called_once()
        # The V2 middleware must carry EDIT into the same shared caller.
        request = SimpleNamespace(tool_call={"name": "plan_course", "id": "edit", "args": {}},
                                  state={"messages": [HumanMessage("카페만 바꿔줘")],
                                         "decision": {"course_request": "EDIT"},
                                         "context": {"stadium": "SAJIK", "currentCourse": CURRENT}})
        with patch.object(agent, "answer", return_value={"answer": "확인"}) as answer:
            DynamicToolMiddleware(["plan_course"]).wrap_tool_call(request, lambda _: assistant.plan_course("사직 카페"))
        self.assertEqual(answer.call_args.kwargs["hint_stadium"], "JAMSIL")
        self.assertEqual(answer.call_args.kwargs["current_course"], CURRENT)

    def test_direct_edit_destination_uses_saved_course_before_hint(self):
        for kwargs in ({"current_course": CURRENT}, {"course_memory": {"current": CURRENT}}):
            with self.subTest(kwargs=kwargs), patch.object(editing, "interpret", return_value=plan(
                    "clarify", [], clarification="확인")) as interpret, patch.object(
                    editing, "answer", return_value={"answer": "확인"}), patch.object(agent, "_answer") as generate:
                agent.answer("카페만 바꿔줘", hint_stadium="SAJIK", course_request="EDIT", **kwargs)
            self.assertEqual(interpret.call_args.args[1]["stadiumCode"], "JAMSIL")
            self.assertEqual(interpret.call_args.args[1]["places"], PLACES)
            generate.assert_not_called()

    def test_failed_edit_answer_is_not_rewritten_as_success(self):
        from types import SimpleNamespace
        call = AIMessage("", tool_calls=[{"name": "plan_course", "id": "c", "args": {"request": "교체"}}])
        result = ToolMessage("조건 미확인, 기존 코스 유지", name="plan_course", tool_call_id="c", artifact={"course_edit_handled": True})
        middleware = DynamicToolMiddleware(["plan_course"])
        request = SimpleNamespace(state={"messages": [HumanMessage("교체"), call, result]}, tools=[])
        request.override = lambda **kwargs: request
        final = AIMessage("조건 미확인, 기존 코스 유지")
        response = middleware.wrap_model_call(request, lambda _: final)
        self.assertIs(response, final)

    def test_v2_partial_request_routes_to_edit_tool_and_keeps_map_artifact(self):
        from llm.v2.agent import chain
        from llm.v2.middleware import jev_guidelines
        from llm.v2.tests.test_chain import ScriptedModel, call, decision, fake_tools
        registered = fake_tools([])
        registered.update({t.name: t for t in assistant.build_specialized_tools() if t.name == "plan_course"})
        for outcome in ({"answer": "카페만 변경", "places": PLACES, "edit": True, "stadiumCode": "JAMSIL"},
                        editing.unchanged("어느 카페인가요?")):
            graph = chain.build_graph(ScriptedModel(script=[call("ask_course", {"task": "카페만 교체"}, "e"), AIMessage(outcome["answer"])], calls=[]), registered)
            with patch.object(jev_guidelines, "classify", return_value=decision(capabilities=["nearby_places"])), \
                    patch.object(agent, "answer", return_value=outcome) as edit:
                result = graph.invoke({"messages": [HumanMessage("카페만 바꿔줘")], "context": {"stadium": "JAMSIL", "currentCourse": CURRENT}})
            self.assertEqual(edit.call_args.kwargs["current_course"], CURRENT)
            self.assertEqual(result["messages"][-1].content, outcome["answer"])
            tool = next(m for m in result["messages"] if isinstance(m, ToolMessage))
            from llm.v2.agent.course_output import course_artifacts
            self.assertEqual(any(a.get("course") for a in course_artifacts([tool])), bool(outcome["places"]))

    def test_original_brand_request_is_available_even_when_summary_misses_it(self):
        from unittest.mock import Mock
        model = Mock()
        model.with_structured_output.return_value.invoke.return_value = editing.CandidateChoice(place_id="", evidence="스타벅스 후보 없음")
        with patch.object(agent, "llm", return_value=model), patch("llm.v1.rag.course.evidence_memory.enrich", side_effect=lambda candidates, _: candidates):
            selected = editing.choose([REPLACEMENT], plan("replace", ["cafe"], conditions=["식당은 유지"]), "스타벅스로 바꿔줘")
        self.assertIsNone(selected)
        supplied = model.with_structured_output.return_value.invoke.call_args.args[0][-1].content
        self.assertIn("스타벅스로 바꿔줘", supplied)

    def test_routing_failure_uses_labeled_estimate_and_never_drops_stop(self):
        with patch.object(agent, "invoke_domain_tool", side_effect=RuntimeError("provider unavailable")):
            legs = editing.route_legs(PLACES, "walk", {})
        self.assertEqual(len(legs), 3)
        self.assertTrue(all(leg.get("estimated") for leg in legs))
