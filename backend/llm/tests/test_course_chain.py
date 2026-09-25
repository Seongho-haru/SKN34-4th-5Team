"""출발지 기준 이어 짜기(geo.chain_course) 단위 테스트 — DB·LLM 없이 좌표 계산만 본다."""
from django.test import SimpleTestCase

from ..v1.rag.course import geo

# 위도 0.009 ≈ 1km
ANCHOR = {"key": "STADIUM", "lat": 37.500, "lng": 127.000}


def place(key, lat, lng, category="FOOD_OUT", dist=0.3):
    return {"key": key, "lat": lat, "lng": lng, "category": category, "dist": dist, "detail": ""}


def relevance(p):
    return 1 - p["dist"]


def same_kind(a, b):
    return a["category"] == b["category"]


class ChainCourseTests(SimpleTestCase):
    def run_chain(self, course, cands, origin):
        lookup = {p["key"]: p for p in cands} | {"STADIUM": ANCHOR}
        return geo.chain_course(course, lookup, cands, origin, ANCHOR, relevance, same_kind)

    def test_first_stop_moves_next_to_the_origin_and_after_stop_next_to_the_stadium(self):
        origin = {"lat": 37.5135, "lng": 127.000}                   # 구장 북쪽 1.5km
        north = place("P1", 37.5125, 127.0005)                       # 출발지 바로 옆
        south = place("P2", 37.4950, 127.000)                        # 구장 남쪽 (LLM 이 고른 곳)
        cafe_near = place("P3", 37.5005, 127.0005, "CAFE")
        cafe_far = place("P4", 37.5180, 127.000, "CAFE")            # LLM 이 고른 곳, 구장에서 2km
        course = [{"key": "P2", "phase": "BEFORE", "reason": "llm"},
                  {"key": "STADIUM", "phase": "GAME", "reason": "경기 관람"},
                  {"key": "P4", "phase": "AFTER", "reason": "llm"}]
        result, changed = self.run_chain(course, [north, south, cafe_near, cafe_far], origin)
        self.assertTrue(changed)
        self.assertEqual([c["key"] for c in result], ["P1", "STADIUM", "P3"])
        self.assertTrue(result[0]["reason"].startswith("출발지에서 도보"))
        self.assertTrue(result[2]["reason"].startswith("구장에서 도보"))
        self.assertEqual(course[0]["key"], "P2")                     # 원본은 그대로

    def test_each_before_stop_is_chosen_around_the_previous_one(self):
        origin = {"lat": 37.5180, "lng": 127.000}                   # 구장 북쪽 2km
        a = place("A", 37.5170, 127.000)                             # 출발지 옆
        b_near_a = place("B", 37.5140, 127.000, "SPOT")              # A 에서 구장 쪽으로 330m
        b_near_origin_but_away = place("C", 37.5190, 127.004, "SPOT")  # 구장에서 더 멀어짐
        course = [{"key": "A", "phase": "BEFORE", "reason": "llm"},
                  {"key": "C", "phase": "BEFORE", "reason": "llm"},
                  {"key": "STADIUM", "phase": "GAME", "reason": "경기 관람"}]
        result, _ = self.run_chain(course, [a, b_near_a, b_near_origin_but_away], origin)
        self.assertEqual([c["key"] for c in result], ["A", "B", "STADIUM"])
        self.assertEqual(result[0]["reason"], "llm")                 # 이미 최선이면 LLM 이유 유지
        self.assertTrue(result[1]["reason"].startswith("앞 장소에서 도보"))

    def test_places_moving_far_from_the_stadium_are_excluded_when_possible(self):
        origin = {"lat": 37.5000, "lng": 127.001}                   # 구장 바로 옆
        far = place("F", 37.5000, 127.015, dist=0.0)                 # 관련도 최고지만 구장에서 1.2km 멀어짐
        near = place("N", 37.5040, 127.001, dist=0.5)
        course = [{"key": "F", "phase": "BEFORE", "reason": "llm"},
                  {"key": "STADIUM", "phase": "GAME", "reason": "경기 관람"}]
        result, _ = self.run_chain(course, [far, near], origin)
        self.assertEqual(result[0]["key"], "N")

    def test_without_origin_nothing_changes(self):
        cands = [place("P1", 37.51, 127.0)]
        course = [{"key": "P1", "phase": "BEFORE", "reason": "llm"}]
        result, changed = self.run_chain(course, cands, None)
        self.assertFalse(changed)
        self.assertEqual(result, course)


class StepScoreTests(SimpleTestCase):
    """앞 지점 주변 후보 중 '가깝고, 구장 쪽으로 다가가는' 곳을 고르는 규칙."""

    def test_toward_stadium_beats_same_distance_sideways_and_backwards(self):
        prev = {"lat": 37.5180, "lng": 127.000}                     # 구장 북쪽 2km
        toward = place("T", 37.5135, 127.000)                        # 구장 쪽으로 500m
        sideways = place("S", 37.5180, 127.0057)                     # 옆으로 500m
        backwards = place("B", 37.5225, 127.000)                     # 반대쪽으로 500m
        score = lambda p: geo.step_score(p, prev, ANCHOR, relevance)
        self.assertGreater(score(toward), score(sideways))
        self.assertGreater(score(sideways), score(backwards))

    def test_near_step_beats_far_step_in_the_same_direction_when_both_are_allowed(self):
        prev = {"lat": 37.5180, "lng": 127.000}
        near = place("N", 37.5150, 127.0020)                         # 앞 지점 옆
        far_lateral = place("F", 37.5100, 127.0200)                  # 멀고 옆으로 크게 빠짐
        score = lambda p: geo.step_score(p, prev, ANCHOR, relevance)
        self.assertGreater(score(near), score(far_lateral))

    def test_before_steps_cannot_move_away_from_the_stadium_and_after_steps_stay_close(self):
        prev = {"lat": 37.5180, "lng": 127.000}
        self.assertFalse(geo.step_allowed(place("B", 37.5225, 127.0), prev, ANCHOR, "BEFORE", 1500))
        self.assertTrue(geo.step_allowed(place("T", 37.5135, 127.0), prev, ANCHOR, "BEFORE", 1500))
        self.assertTrue(geo.step_allowed(place("A", 37.5090, 127.0), ANCHOR, ANCHOR, "AFTER", 1500))
        self.assertFalse(geo.step_allowed(place("X", 37.5270, 127.0), ANCHOR, ANCHOR, "AFTER", 1500))


class OriginStepSearchTests(SimpleTestCase):
    """출발지 → 1번(출발지 주변 검색) → 2번(1번 주변 검색) → 구장 → 경기 후(구장 주변 검색)."""

    def test_each_step_searches_around_the_previous_point(self):
        from unittest.mock import patch

        from ..v1.rag.course import agent

        centers = []

        def invoke(domain, name, args):
            self.assertEqual((domain, name), ("course", "search_places"))
            lat, lng = args["latitude"], args["longitude"]
            centers.append((args["category"], round(lat, 4), round(lng, 4)))
            # 검색 중심에서 구장 쪽(남쪽)으로 300m 떨어진 곳 하나, 반대쪽 300m 하나를 돌려준다
            label = {"FD6": "식당", "CE7": "카페", "AT4": "명소"}[args["category"]]
            if args.get("query") == "술집":
                label = "술집"
            return {"places": [
                {"id": f"{label}-S-{lat}", "place_name": f"{label} 남 {lat:.4f}", "x": str(lng), "y": str(lat - 0.0027),
                 "category_name": f"음식점 > {label}"},
                {"id": f"{label}-N-{lat}", "place_name": f"{label} 북 {lat:.4f}", "x": str(lng), "y": str(lat + 0.0027),
                 "category_name": f"음식점 > {label}"},
            ]}

        sl = {"spare": "normal", "prefs": [], "ban": [], "boost": [], "exclude": set()}
        origin = {"lat": 37.5270, "lng": 127.000}                   # 구장 북쪽 3km
        with patch.object(agent, "invoke_domain_tool", side_effect=invoke):
            steps = agent.build_origin_course(origin, ANCHOR, [], sl, evening=True)

        self.assertEqual([s["phase"] for s in steps], ["BEFORE", "BEFORE", "GAME", "AFTER"])
        food, cafe, _, bar = steps
        # 1번은 출발지 주변 검색, 2번은 1번 주변 검색, 경기 후는 구장 주변 검색
        self.assertEqual(centers[0], ("FD6", 37.527, 127.0))
        self.assertEqual(centers[1], ("CE7", round(food["place"]["lat"], 4), 127.0))
        self.assertEqual(centers[2], ("FD6", 37.5, 127.0))
        # 매 단계 구장 쪽(남쪽)을 고른다
        self.assertLess(food["place"]["lat"], origin["lat"])
        self.assertLess(cafe["place"]["lat"], food["place"]["lat"])
        self.assertTrue(food["reason"].startswith("출발지에서 도보"))
        self.assertTrue(cafe["reason"].startswith("앞 장소에서 도보"))
        self.assertTrue(bar["reason"].startswith("구장에서 도보"))
        self.assertEqual(agent.timeline.kind_of(bar["place"]), "BAR")

    def test_no_search_results_means_no_origin_course(self):
        from unittest.mock import patch

        from ..v1.rag.course import agent

        sl = {"spare": "tight", "prefs": [], "ban": [], "boost": [], "exclude": set()}
        with patch.object(agent, "invoke_domain_tool", return_value={"places": []}):
            self.assertIsNone(agent.build_origin_course({"lat": 37.51, "lng": 127.0}, ANCHOR, [], sl, evening=False))


class OriginExtraStepTests(SimpleTestCase):
    def sl(self, **extra):
        return {"spare": "normal", "prefs": [], "ban": [], "boost": [], "exclude": set(), **extra}

    def test_walk_request_adds_a_walk_step_before_the_game(self):
        from ..v1.rag.course import agent

        self.assertEqual(agent.plan_steps(self.sl(extras=["walk"]), evening=False), (["FOOD", "CAFE", "WALK"], ["CAFE"]))
        self.assertEqual(agent.plan_steps(self.sl(extras=["walk"], scope="after"), evening=True), ([], ["BAR", "WALK"]))
        self.assertEqual(agent.plan_steps(self.sl(extras=["indoor", "stay"]), evening=True), (["FOOD", "CAFE", "INDOOR"], ["BAR", "STAY"]))
        self.assertEqual(agent.plan_steps(self.sl(scope="before"), evening=True), (["FOOD", "CAFE"], []))

    def test_origin_course_searches_parks_for_the_walk_step(self):
        from unittest.mock import patch

        from ..v1.rag.course import agent

        calls = []

        def invoke(domain, name, args):
            calls.append(args)
            lat, lng = args["latitude"], args["longitude"]
            if args.get("query") == "공원":
                return {"places": [
                    {"id": "p1", "place_name": "주차타워", "x": str(lng), "y": str(lat - 0.001), "category_name": "교통 > 주차장"},
                    {"id": "p2", "place_name": "강변공원", "x": str(lng), "y": str(lat - 0.002), "category_name": "여행 > 공원"},
                ]}
            label = {"FD6": "식당", "CE7": "카페"}[args["category"]]
            return {"places": [{"id": f"{label}-{lat}", "place_name": f"{label} {lat:.4f}", "x": str(lng), "y": str(lat - 0.0027),
                                "category_name": f"음식점 > {label}"}]}

        origin = {"lat": 37.5270, "lng": 127.000}
        with patch.object(agent, "invoke_domain_tool", side_effect=invoke):
            steps = agent.build_origin_course(origin, ANCHOR, [], self.sl(extras=["walk"]), evening=False)

        before = [s for s in steps if s["phase"] == "BEFORE"]
        self.assertEqual([s["place"]["category"] for s in before], ["FOOD_OUT", "CAFE", "WALK"])
        self.assertEqual(before[-1]["place"]["name"], "강변공원")          # 주차장은 산책 후보가 아니다
        walk_call = next(c for c in calls if c.get("query") == "공원")
        self.assertEqual(walk_call["method"], "keyword")
        self.assertNotIn("category", walk_call)


class ContextPrefixTests(SimpleTestCase):
    def test_stadium_and_origin_prefixes_are_split_in_any_order(self):
        from ..v1.rag.pipeline import split_context_prefix, split_stadium_prefix

        self.assertEqual(
            split_context_prefix("[선택한 구장: 잠실야구장]\n[출발지: 37.51234,127.07123]\n코스 짜줘"),
            ("코스 짜줘", "잠실야구장", {"lat": 37.51234, "lng": 127.07123}),
        )
        self.assertEqual(
            split_context_prefix("[출발지: 37.5, 127.0][선택한 구장: 창원 NC 파크]\n코스"),
            ("코스", "창원 NC 파크", {"lat": 37.5, "lng": 127.0}),
        )
        self.assertEqual(split_context_prefix("그냥 질문"), ("그냥 질문", None, None))
        self.assertEqual(split_stadium_prefix("[선택한 구장: 잠실야구장]\n[출발지: 37.5,127.0]\n질문"), ("질문", "잠실야구장"))

    def test_origin_outside_korea_is_ignored(self):
        from ..v1.rag.course.agent import valid_origin

        self.assertEqual(valid_origin({"lat": "37.5", "lng": 127}), {"lat": 37.5, "lng": 127.0})
        self.assertIsNone(valid_origin({"lat": 0, "lng": 0}))
        self.assertIsNone(valid_origin({"lat": 37.5}))
        self.assertIsNone(valid_origin(None))
