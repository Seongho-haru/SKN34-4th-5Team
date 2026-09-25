"""nearby(카카오 실시간 조회) 단위 테스트 — 네트워크·DB·LLM 없이 돈다.

    cd backend && python -m unittest llm.v1.rag.nearby.test_nearby
"""
import json
import unittest
import urllib.parse
from unittest import mock

from . import kakao
from . import agent as nearby
from ..club.router import detect_stadium

STADIUM_DOC = {"id": "1", "place_name": "잠실야구장", "category_name": "스포츠,레저 > 야구장", "x": "127.0719", "y": "37.5122",
               "road_address_name": "서울 송파구 올림픽로 25", "address_name": ""}


def doc(i, name, cat, x, y, road="서울 송파구 다른로 1"):
    return {"id": str(i), "place_name": name, "category_name": cat, "x": str(x), "y": str(y),
            "road_address_name": road, "address_name": "", "place_url": "", "phone": ""}


def fake_fetch(pages):
    """URL 의 query/category 로 응답을 고른다."""
    def fetch(url):
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        if q.get("query") == "잠실야구장":
            return {"documents": [STADIUM_DOC], "meta": {"is_end": True}}
        key = q.get("query") or q.get("category_group_code")
        return {"documents": pages.get(key, []), "meta": {"is_end": True}}
    return fetch


class KindTest(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(nearby.kinds_of("챔피언스 필드 주변 숙박시설 추천해줘"), ["stay"])
        self.assertEqual(nearby.kinds_of("산책할곳 추천해줘"), ["walk"])
        self.assertEqual(nearby.kinds_of("비 오면 갈만한 실내 놀거리"), ["indoor"])
        self.assertEqual(nearby.kinds_of("구장 근처 편의점 있어?"), ["store"])
        self.assertEqual(nearby.kinds_of("비 오면 경기 취소돼?"), [])
        self.assertEqual(nearby.kinds_of("잠실 맛집 추천해줘"), [])

    def test_stadium_alias_ignores_spaces(self):
        self.assertEqual(detect_stadium("챔피언스 필드 주변 숙박시설"), "GWANGJU")
        self.assertEqual(nearby.stadium_of("산책할곳 추천해줘", [{"role": "user", "content": "광주경기 보러 가요"}], None), ("GWANGJU", "carry"))
        self.assertEqual(nearby.stadium_of("산책할곳 추천해줘", [], "SAJIK"), ("SAJIK", "hint"))


class KakaoTest(unittest.TestCase):
    def test_same_conditions_as_map(self):
        pages = {
            "AD5": [doc(10, "롯데호텔 월드", "여행 > 숙박 > 호텔", 127.098, 37.511),
                    doc(11, "구장 안 숙소?", "여행 > 숙박", 127.072, 37.512, road="서울 송파구 올림픽로 25"),   # 구장 건물
                    doc(12, "먼 호텔", "여행 > 숙박 > 호텔", 127.20, 37.60)],                                        # 2.5km 밖
            "공원": [doc(20, "아시아공원", "여행 > 공원 > 도시근린공원", 127.0766, 37.5104),
                     doc(21, "공원약국", "의료,건강 > 약국", 127.073, 37.513)],                                        # 병원·약국 제외
        }
        with mock.patch.object(kakao, "_cache", return_value=None):
            stay = kakao.nearby("JAMSIL", "stay", fetch=fake_fetch(pages))
            walk = kakao.nearby("JAMSIL", "walk", fetch=fake_fetch(pages))
        self.assertEqual([p["name"] for p in stay], ["롯데호텔 월드"])
        self.assertEqual([p["name"] for p in walk], ["아시아공원"])
        self.assertLess(walk[0]["distance"], 800)                  # 실제 야구장 좌표 기준 거리
        self.assertEqual(stay[0]["placeId"], "10")

    def test_no_key_returns_empty(self):
        with mock.patch.dict("os.environ", {"KAKAO_REST_API_KEY": ""}):
            self.assertEqual(kakao.nearby("JAMSIL", "stay"), [])

    def test_failure_returns_empty(self):
        def broken(url):
            raise OSError("down")
        with mock.patch.object(kakao, "_cache", return_value=None):
            self.assertEqual(kakao.nearby("JAMSIL", "stay", fetch=broken), [])


class AnswerTest(unittest.TestCase):
    PLACES = [
        {"kind": "stay", "kindLabel": "숙박", "name": "잠실 게스트하우스", "detail": "여행 > 숙박 > 게스트하우스", "distance": 800,
         "lat": 37.51, "lng": 127.08, "address": "x", "placeId": "1", "placeUrl": "", "phone": ""},
        {"kind": "stay", "kindLabel": "숙박", "name": "롯데호텔 월드", "detail": "여행 > 숙박 > 호텔", "distance": 2100,
         "lat": 37.51, "lng": 127.09, "address": "y", "placeId": "2", "placeUrl": "", "phone": ""},
        {"kind": "stay", "kindLabel": "숙박", "name": "롯데호텔 서울", "detail": "여행 > 숙박 > 호텔", "distance": 2200,
         "lat": 37.51, "lng": 127.09, "address": "z", "placeId": "3", "placeUrl": "", "phone": ""},
    ]

    def test_narrow_subtype_and_brand(self):
        self.assertEqual([p["name"] for p in nearby.narrow(self.PLACES, "stay", "잠실 근처 호텔")], ["롯데호텔 월드"])
        self.assertEqual(len(nearby.narrow(self.PLACES, "stay", "잠실 근처 숙소")), 2)

    def test_answer_falls_back_to_template_without_llm(self):
        with mock.patch.object(kakao, "enabled", return_value=True), \
             mock.patch.object(kakao, "nearby", return_value=self.PLACES), \
             mock.patch.object(nearby, "llm", side_effect=RuntimeError("no llm")):
            r = nearby.answer("챔피언스 필드 주변 숙박 추천해줘")
        self.assertIn("게스트하우스", r["answer"])
        self.assertIn("template", r["route"])
        self.assertNotIn("places", r)                 # 코스 카드로 오해받지 않게

    def test_clarify_and_no_key(self):
        self.assertEqual(nearby.answer("숙소 추천해줘")["route"], "nearby:clarify")
        with mock.patch.object(kakao, "enabled", return_value=False):
            self.assertIn("옆 지도", nearby.answer("잠실 숙소 추천해줘")["answer"])


if __name__ == "__main__":
    unittest.main()
