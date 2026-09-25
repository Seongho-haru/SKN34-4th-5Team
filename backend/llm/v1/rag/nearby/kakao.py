"""[nearby] 카카오 장소 실시간 조회 — 루트 작성 화면의 지도와 **같은 조건**으로 백엔드에서 찾는다. LLM 호출 0회.

왜 필요한가
    지도(프론트)는 화면을 열 때마다 카카오에서 먹거리·카페·산책·명소·실내놀거리·편의점·숙박을 새로 찾는다.
    챗봇(RAG)은 9월 초에 모아 임베딩한 음식점·카페·관광명소 3종류만 알아서,
    "숙박 추천해줘"·"산책할 곳"에 "정보가 없다"고 답했다. 챗봇이 지도와 같은 장소를 보게 하려고
    백엔드가 카카오를 직접 조회하는 통로를 만든다. (RAG 에 없는 종류를 채우는 용도)

지도와 맞춘 조건 (frontend/lib/nearby-places.ts · nearby-search.ts)
    기준점   구장 주소 좌표가 아니라 카카오에서 찾은 "야구장" 장소 좌표 (주소 좌표는 잠실에서 약 500m 어긋난다)
    반경     2.5km (먹거리_플레이스_반경 정책 2026-09-08)
    종류     walk(공원·산책로·둘레길) · sight(AT4) · indoor(CT1 + 볼링장·보드게임카페·방탈출·영화관)
             · store(CS2) · stay(AD5) · food(FD6) · cafe(CE7)
    제외     구장과 같은 도로명주소(구장 안 매장), 병원·약국

비용·속도
    종류마다 검색 1~4개 × 최대 3페이지. 결과는 Django 캐시에 6시간 보관한다 (구장·종류 단위).
    키가 없거나 카카오가 실패하면 빈 목록 — 챗봇은 "지금은 찾지 못했다"고 답하고 죽지 않는다.

키: 환경변수 KAKAO_REST_API_KEY (프론트 지도와 같은 REST 키, docker-compose backend 에도 넘겨야 한다)
"""
import json
import logging
import math
import os
import re
import urllib.parse
import urllib.request

log = logging.getLogger(__name__)

RADIUS_M = 2500
PAGES = 3
PAGE_SIZE = 15
CACHE_SECONDS = 6 * 60 * 60
TIMEOUT = 6
BASE = "https://dapi.kakao.com/v2/local/search"

KIND_LABEL = {"walk": "산책", "sight": "관광 명소", "indoor": "실내 놀거리", "store": "편의점",
              "stay": "숙박", "food": "먹거리", "cafe": "카페·디저트"}

# (방식, 검색어 또는 카테고리 코드)
SPECS = {
    "walk": [("keyword", "공원"), ("keyword", "산책로"), ("keyword", "둘레길")],
    "sight": [("category", "AT4")],
    "indoor": [("category", "CT1"), ("keyword", "볼링장"), ("keyword", "보드게임카페"), ("keyword", "방탈출"), ("keyword", "영화관")],
    "store": [("category", "CS2")],
    "stay": [("category", "AD5")],
    "food": [("category", "FD6")],
    "cafe": [("category", "CE7")],
}
_WALK = re.compile(r"공원|산책로|둘레길|도보여행|수변|숲|호수")
_INDOOR = re.compile(r"보드게임|방탈출|볼링|영화관|박물관|미술관|전시관|과학관|공연장|극장|아트홀|아쿠아리움|오락실|실내")
_EXCLUDE = re.compile(r"병원|의원|약국|주차장")
_ROAD = re.compile(r"(\S+(?:로|길))\s*(\d+(?:-\d+)?)")

# 코드 → 카카오에서 "야구장"을 찾을 검색어 (stadium_coordinates.csv 한글명과 같다)
STADIUM_QUERY = {"JAMSIL": "잠실야구장", "GOCHEOK": "고척스카이돔", "MUNHAK": "인천 SSG 랜더스필드",
                 "SUWON": "수원 KT 위즈 파크", "DAEJEON": "대전 한화생명 볼파크", "DAEGU": "대구 삼성 라이온즈 파크",
                 "GWANGJU": "광주-KIA 챔피언스 필드", "SAJIK": "사직야구장", "CHANGWON": "창원 NC 파크"}


def _key():
    return (os.getenv("KAKAO_REST_API_KEY") or "").strip()


def enabled() -> bool:
    return bool(_key())


def _cache():
    try:
        from django.core.cache import cache
        return cache
    except Exception:                                  # 순수 함수 테스트 (Django 없음)
        return None


def _get(path, params, fetch=None):
    """카카오 로컬 API 한 번. fetch 는 테스트용 주입 (url -> dict)."""
    url = f"{BASE}/{path}.json?{urllib.parse.urlencode(params)}"
    if fetch:
        return fetch(url)
    req = urllib.request.Request(url, headers={"Authorization": f"KakaoAK {_key()}"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as res:   # noqa: S310 — 고정된 카카오 도메인
        return json.loads(res.read().decode("utf-8"))


def haversine_m(a_lat, a_lng, b_lat, b_lng):
    r = 6371000.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp, dl = math.radians(b_lat - a_lat), math.radians(b_lng - a_lng)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def road_key(address):
    m = _ROAD.search(address or "")
    return f"{m.group(1)} {m.group(2)}" if m else None


def resolve_stadium(code, fetch=None):
    """카카오에서 찾은 야구장 좌표 {"name","lat","lng","address"} (지도와 같은 기준점). 못 찾으면 None."""
    query = STADIUM_QUERY.get(code)
    if not query:
        return None
    cache, ck = _cache(), f"kakao:stadium:{code}"
    if cache and (hit := cache.get(ck)):
        return hit
    data = _get("keyword", {"query": query, "size": 15, "sort": "accuracy"}, fetch)
    compact = lambda v: re.sub(r"[\s-]", "", v or "").lower().replace("kia", "기아")   # noqa: E731
    want = compact(query).removeprefix("인천")
    for d in data.get("documents") or []:
        if "야구장" in (d.get("category_name") or "") and want in compact(d.get("place_name")):
            out = {"name": d["place_name"], "lat": float(d["y"]), "lng": float(d["x"]),
                   "address": d.get("road_address_name") or d.get("address_name") or ""}
            if cache:
                cache.set(ck, out, 24 * 60 * 60)
            return out
    return None


def _accept(kind, doc, stadium):
    detail, name = doc.get("category_name") or "", doc.get("place_name") or ""
    if _EXCLUDE.search(detail) or "야구장" in detail:
        return False
    if road_key(doc.get("road_address_name")) and road_key(doc.get("road_address_name")) == road_key(stadium["address"]):
        return False                                   # 구장과 같은 건물 = 구장 안 매장
    if kind == "walk":
        return bool(_WALK.search(detail))
    if kind == "indoor":
        return bool(_INDOOR.search(f"{detail} {name}"))
    return True


def _normalize(kind, doc, stadium):
    lat, lng = float(doc["y"]), float(doc["x"])
    return {
        "kind": kind, "kindLabel": KIND_LABEL[kind], "name": doc.get("place_name") or "",
        "detail": doc.get("category_name") or "", "lat": lat, "lng": lng,
        "distance": int(haversine_m(stadium["lat"], stadium["lng"], lat, lng)),
        "address": doc.get("road_address_name") or doc.get("address_name") or "",
        "placeId": str(doc.get("id") or "") or None, "placeUrl": doc.get("place_url") or "",
        "phone": doc.get("phone") or "",
    }


def nearby(code, kind, fetch=None) -> list[dict]:
    """구장 반경 2.5km 안의 한 종류 장소 (가까운 순). 키가 없거나 실패하면 []."""
    if kind not in SPECS or not (fetch or enabled()):
        return []
    cache, ck = _cache(), f"kakao:nearby:{code}:{kind}"
    if cache and (hit := cache.get(ck)) is not None:
        return hit
    try:
        stadium = resolve_stadium(code, fetch)
        if not stadium:
            return []
        found = {}
        for method, value in SPECS[kind]:
            for page in range(1, PAGES + 1):
                params = {"x": stadium["lng"], "y": stadium["lat"], "radius": RADIUS_M,
                          "sort": "distance", "page": page, "size": PAGE_SIZE}
                params.update({"query": value} if method == "keyword" else {"category_group_code": value})
                data = _get(method, params, fetch)
                for doc in data.get("documents") or []:
                    if doc.get("id") and doc["id"] not in found and _accept(kind, doc, stadium):
                        found[doc["id"]] = _normalize(kind, doc, stadium)
                if (data.get("meta") or {}).get("is_end", True):
                    break
        out = sorted((p for p in found.values() if p["distance"] <= RADIUS_M), key=lambda p: p["distance"])
    except Exception:
        log.exception("kakao nearby failed: %s %s", code, kind)
        return []
    if cache:
        cache.set(ck, out, CACHE_SECONDS)
    return out
