"""course 이동수단·범위 슬롯 단위 테스트 — DB·LLM 없이 돈다.

    cd backend && python -m unittest llm.v1.rag.course.test_transport
"""
import unittest

from . import slots, transport

JAMSIL_ROWS = [
    {"access_code": "SUBWAY", "mode": "SUBWAY", "title": "종합운동장역", "details": "2호선 6·7번 출구",
     "reservation_required": "N", "evidence_type": "OFFICIAL", "status": "CONFIRMED"},
    {"access_code": "PARKING", "mode": "CAR", "title": "종합운동장 주차", "details": "명목 876면. 소형 5분 200원",
     "reservation_required": "N", "evidence_type": "OFFICIAL", "status": "CONFIRMED"},
]
SUWON_ROWS = [
    {"access_code": "PARKING_CAPACITY_NEWS", "mode": "PARKING", "title": "합산 주차 수용면", "details": "1,402면",
     "evidence_type": "UNOFFICIAL", "status": "PARTIAL"},
    {"access_code": "PARKING_GAME_RESERVATION", "mode": "PARKING", "title": "야구경기 예약차량 주차",
     "details": "경기 7일 전 17:00부터 예약", "reservation_required": "Y", "evidence_type": "OFFICIAL", "status": "CONFIRMED"},
]
LEGACY_ROWS = [
    {"data_type": "주차_수용면", "field_name": "parking_total", "value": "1679", "unit": "면",
     "detail": "지하 1,220면 + 지상 459면.", "status": "CONFIRMED", "source_grade": "A-"},
    {"data_type": "좌석도", "field_name": "seat_capacity", "value": "20500", "unit": "석", "detail": "", "status": "CONFIRMED"},
]


class ModeTest(unittest.TestCase):
    def test_detects_modes(self):
        self.assertEqual(transport.mode_of("차타고 갈건데 루트 추천해줘"), "car")
        self.assertEqual(transport.mode_of("주차 편한 코스로"), "car")
        self.assertEqual(transport.mode_of("지하철 타고 가요"), "transit")
        self.assertEqual(transport.mode_of("걸어서 다닐 거예요"), "walk")
        self.assertIsNone(transport.mode_of("녹차 좋아하는데 잠실 코스"))
        self.assertIsNone(transport.mode_of("잠실 코스 짜줘"))

    def test_taxi_is_not_driving(self):
        self.assertTrue(transport.is_taxi("택시 타고 갈게요"))
        self.assertEqual(transport.ban_words("car", taxi=True), [])
        self.assertIn("술집", transport.ban_words("car"))


class LegTest(unittest.TestCase):
    def test_short_leg_walks_even_by_car(self):
        self.assertEqual(transport.leg(700, "car")["by"], "walk")

    def test_long_leg_drives_with_parking_time(self):
        x = transport.leg(2400, "car")
        self.assertEqual(x["by"], "car")
        self.assertGreater(x["minutes"], transport.PARK_MIN)
        self.assertLess(x["minutes"], transport.leg(2400, "walk")["minutes"])

    def test_missing_coords_use_default(self):
        self.assertEqual(transport.legs([{"lat": None, "lng": None}, {"lat": 37.5, "lng": 127.0}], "car")[0]["minutes"],
                         transport.DEFAULT_LEG_MIN)

    def test_summary(self):
        walk_only = [transport.leg(500, "car"), transport.leg(300, "car")]
        self.assertIn("한 번 주차", transport.summary(walk_only, "car"))
        mixed = [transport.leg(2400, "car"), transport.leg(300, "car")]
        self.assertIn("차량", transport.summary(mixed, "car"))
        self.assertTrue(transport.summary([transport.leg(800, None)], None).startswith("총 도보"))
        self.assertEqual(transport.summary([], "car"), "")
        self.assertIn("택시에서 내린 뒤", transport.summary(walk_only, "car", taxi=True))


class AccessLinesTest(unittest.TestCase):
    def test_car_gets_parking_transit_gets_subway(self):
        self.assertTrue(transport.access_lines(JAMSIL_ROWS, "car")[0].startswith("주차: 종합운동장 주차"))
        self.assertTrue(transport.access_lines(JAMSIL_ROWS, "transit")[0].startswith("오는 길: 종합운동장역"))
        self.assertEqual(transport.access_lines(JAMSIL_ROWS, "walk"), [])
        self.assertEqual(transport.access_lines(JAMSIL_ROWS, "car", taxi=True), [])

    def test_reservation_first_and_weak_rows_marked(self):
        lines = transport.access_lines(SUWON_ROWS, "car")
        self.assertIn("예약 필요", lines[0])
        self.assertIn("참고용", lines[1])

    def test_legacy_csv_rows(self):
        lines = transport.access_lines(LEGACY_ROWS, "car")
        self.assertEqual(len(lines), 1)
        self.assertIn("1679면", lines[0])


class ScopeSlotTest(unittest.TestCase):
    def test_before_only(self):
        q = "여자친구랑 둘이 일식집,카페 먹고 경기장갈거야 차타고 갈건데 루트 추천해줘"
        s = slots.parse(q)
        self.assertEqual((s["scope"], s["mode"], s["companion"]), ("before", "car", "couple"))
        self.assertIn("술집", s["ban"])
        self.assertIn("경기 전만", slots.prompt_line(s))

    def test_after_only_and_both(self):
        self.assertEqual(slots.scope_of("경기 끝나고 갈 데 추천해줘"), "after")
        self.assertEqual(slots.scope_of("경기 전후 코스 짜줘"), "both")
        self.assertEqual(slots.scope_of("잠실 직관 코스 짜줘"), "both")
        self.assertEqual(slots.scope_of("직관 후기 보고 코스 짜줘"), "both")
        self.assertEqual(slots.scope_of("카페 들렀다가 잠실 경기장 갈 거야"), "before")
        self.assertEqual(slots.scope_of("밥 먹고 경기 보러 갈래"), "before")
        # 사이에 말이 끼거나 조사가 붙어도 경기 전만 (요청하지 않은 경기 후 장소를 넣지 않는다)
        self.assertEqual(slots.scope_of("여자친구랑 분식집 갔다가 카페 방문 후 산책 좀 하고 구장에 가고 싶어"), "before")
        self.assertEqual(slots.scope_of("카페 갔다가 야구장으로 갈게"), "before")
        self.assertEqual(slots.scope_of("친구랑 경기 보고 나서 술 한잔할 곳"), "after")
        self.assertEqual(slots.scope_of("치킨 먹고 싶어 잠실 코스 알려줘"), "both")

    def test_mode_carries_from_history(self):
        history = [{"role": "user", "content": "차 끌고 잠실 갈 거예요"}, {"role": "assistant", "content": "..."}]
        self.assertEqual(slots.parse("다른 데 없어?", history)["mode"], "car")


if __name__ == "__main__":
    unittest.main()
