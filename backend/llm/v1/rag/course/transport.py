"""[course] 이동수단 — 도보 · 자동차 · 대중교통. 구간 시간 · 주차/대중교통 안내. LLM 호출 0회.

왜 필요한가: "차 타고 갈 건데 루트 짜줘" 에도 예전 코스는 전부 도보 기준(80m/분)으로 시간을 쟀고,
주차 이야기가 한 줄도 없었다. 경기 끝나고 운전하는 사람에게 술집을 추천하기도 했다.

이 파일이 하는 일
    mode_of(question)      질문 → "walk" | "car" | "transit" | None(말 안 함 → 도보로 계산)
    legs(points, mode)     구간마다 {meters, minutes, by}  — 차량이어도 가까운 구간(1km 이하)은 걸어서
    summary(legs, mode)    "이동 약 3.1km · 도보 12분 + 차량 9분(주차 포함)" 같은 한 줄
    access_lines(rows, mode)  DB(TRANSPORT 청크)에서 주차 또는 지하철·버스 안내 줄 1~2개
    ban_words(mode)        운전이면 술집 업종을 후보에서 뺀다
    info(code, mode)       위 안내를 DB 에서 읽어 한 번에 돌려준다 (agent 가 부르는 진입점)

숫자 기준 (전부 상수라 바꾸기 쉽다 · 이동시간은 '예상'으로만 말한다)
    도보           80m/분 (geo·timeline 과 같은 값)
    차량           도심 평균 20km/h(333m/분) × 도로 우회계수 1.4 + 주차·걷기 10분
    차량으로 갈지   구간 직선거리 1km 이하면 차를 빼느니 걷는다 (주차 두 번이 더 오래 걸린다)

좌표·요금·면수는 전부 구장교통정보.csv / 구장잔여정보_좌석주차버스.csv 에서 온 DB 값이다.
PARTIAL·RECHECK·비공식 행은 "(참고용)" 을 붙여 확정 정보처럼 말하지 않는다.
"""
import json
import math
import re

WALK_M_PER_MIN = 80
CAR_M_PER_MIN = 20000 / 60          # 도심 평균 20km/h
ROAD_FACTOR = 1.4                   # 직선거리 → 도로거리 대략 보정
PARK_MIN = 10                       # 주차장 찾고 세우고 걸어가는 시간
CAR_WALK_MAX_M = 1000               # 이 거리 이하 구간은 차 대신 걷는다
DEFAULT_LEG_MIN = 10

LABEL = {"walk": "도보", "car": "자동차", "transit": "대중교통"}

# ── 이동수단 판별 ─────────────────────────────────────────────────────────────
# "차" 한 글자는 녹차·차이 같은 말에도 걸리니 "차 타고 / 차로 가 / 차 끌고" 처럼 동사와 붙은 경우만 본다.
CAR = re.compile(r"자가용|자차|자동차|승용차|차량|렌터카|렌트카|쏘카|운전|드라이브|주차|"
                 r"차\s*(를|로)?\s*(타고|끌고|가지고|몰고|갖고)|차로\s*(가|갈|이동|움직)|"
                 r"차\s*타고|차\s*있어|택시")
TRANSIT = re.compile(r"대중교통|지하철|전철|버스|[0-9]호선|기차|KTX|SRT|ITX|셔틀|뚜벅")
WALK = re.compile(r"걸어서|도보로|도보\s*(이동|로만)|걸어\s*(다닐|갈|다니)")
# 운전하면 경기 후라도 술집은 뺀다 (timeline.BAR_WORDS 와 같은 목록 + 음주 표현)
BAR_WORDS = ["술집", "호프", "포장마차", "이자카야", "맥주", "바(BAR)", "요리주점", "실내포장마차", "와인", "칵테일"]


def mode_of(question: str):
    """'car' | 'transit' | 'walk' | None. 택시도 차량 동선으로 본다 (주차 안내는 빼고)."""
    q = question or ""
    if CAR.search(q):
        return "car"
    if TRANSIT.search(q):
        return "transit"
    if WALK.search(q):
        return "walk"
    return None


def is_taxi(question: str) -> bool:
    return bool(re.search(r"택시", question or "")) and not re.search(r"자가용|자차|자동차|운전|주차|차\s*끌고", question or "")


def ban_words(mode, taxi=False) -> list[str]:
    """운전할 때만 술집을 뺀다 (택시는 운전이 아니다)."""
    return list(BAR_WORDS) if mode == "car" and not taxi else []


def after_hint(mode, evening: bool, taxi=False) -> str:
    """경기 후 후보 종류 안내 (LLM 프롬프트 <after_hint> 용)."""
    if mode == "car" and not taxi:
        return "카페·야식(술집 제외 — 운전)" if evening else "카페·명소·산책"
    return "야식·술집·카페" if evening else "카페·명소·산책"


# ── 구간 계산 ────────────────────────────────────────────────────────────────
def _haversine(a, b):
    if None in (a.get("lat"), a.get("lng"), b.get("lat"), b.get("lng")):
        return None
    r = 6371000.0
    p1, p2 = math.radians(a["lat"]), math.radians(b["lat"])
    dp, dl = math.radians(b["lat"] - a["lat"]), math.radians(b["lng"] - a["lng"])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def leg(meters, mode) -> dict:
    """한 구간. by 는 실제로 어떻게 움직이는지 ('walk' | 'car')."""
    if not meters:
        return {"meters": None, "minutes": DEFAULT_LEG_MIN, "by": "walk"}
    if mode == "car" and meters > CAR_WALK_MAX_M:
        minutes = math.ceil(meters * ROAD_FACTOR / CAR_M_PER_MIN) + PARK_MIN
        return {"meters": int(meters), "minutes": int(minutes), "by": "car"}
    return {"meters": int(meters), "minutes": max(1, round(meters / WALK_M_PER_MIN)), "by": "walk"}


def legs(points, mode) -> list[dict]:
    """연속 구간 목록 (길이 len(points)-1). 대중교통 코스도 구장 주변 장소끼리는 걸어서 잇는다."""
    return [leg(_haversine(a, b), mode) for a, b in zip(points, points[1:])]


def leg_minutes(legs_) -> list[int]:
    return [x["minutes"] for x in legs_]


def _km(m):
    return f"{m / 1000:.1f}km" if m >= 1000 else f"{int(m)}m"


def summary(legs_, mode, taxi=False) -> str:
    """답변 꼬리의 이동 요약 한 줄. 빈 문자열이면 붙이지 않는다."""
    known = [x for x in legs_ if x["meters"]]
    if not known:
        return ""
    total = sum(x["meters"] for x in known)
    walk_min = sum(x["minutes"] for x in known if x["by"] == "walk")
    car_min = sum(x["minutes"] for x in known if x["by"] == "car")
    if mode == "car":
        if not car_min:
            first = "택시에서 내린 뒤엔 걸어 다니면 돼요" if taxi else "장소끼리 가까워서 한 번 주차하고 걸어 다니면 돼요"
            return f"{first} · 도보 약 {_km(total)} · {walk_min}분"
        parts = [f"{'택시' if taxi else '차량'} 약 {car_min}분{'' if taxi else '(주차 포함)'}"] + ([f"도보 {walk_min}분"] if walk_min else [])
        return f"이동 약 {_km(total)} · " + " + ".join(parts)
    return f"총 도보 약 {_km(total)} · {walk_min}분"


# ── 주차 · 대중교통 안내 (DB) ─────────────────────────────────────────────────
_WEAK_STATUS = {"PARTIAL", "RECHECK", "HISTORICAL_OFFICIAL", "OFFICIAL_APPROXIMATE"}
_TRANSIT_MODES = {"SUBWAY", "BUS", "PUBLIC_TRANSIT", "BUS_TAXI", "SUBWAY_BUS", "BUS_SUBWAY", "SHUTTLE"}


def _kind(m) -> str:
    """TRANSPORT 청크 한 줄이 주차인지 대중교통인지. 두 CSV 의 컬럼이 달라 둘 다 본다."""
    mode, access = str(m.get("mode") or ""), str(m.get("access_code") or "")
    data_type = str(m.get("data_type") or "")
    if mode in ("CAR", "PARKING") or access.startswith(("PARK", "DRIVE")) or data_type.startswith("주차"):
        return "parking"
    if mode in _TRANSIT_MODES or data_type == "버스정류장명":
        return "transit"
    return ""


def _weak(m) -> bool:
    return (str(m.get("status") or "") in _WEAK_STATUS or str(m.get("evidence_type") or "") == "UNOFFICIAL"
            or str(m.get("source_grade") or "") in ("C", "D"))


def _text(m) -> str:
    """'종합운동장 주차 — 명목 876면 …' 형태. 레거시 CSV 는 field_name/value/detail 로 되어 있다."""
    if m.get("title"):
        head, body = str(m["title"]), str(m.get("details") or "")
    else:
        unit = m.get("unit") or ""
        head = {"주차_수용면": "주차 수용", "주차장_요금": "주차 요금", "버스정류장명": "버스 정류장"}.get(
            str(m.get("data_type") or ""), str(m.get("data_type") or "안내"))
        body = f"{m.get('value') or ''}{unit} · {m.get('detail') or ''}".strip(" ·")
    body = re.sub(r"\s*\[20\d\d-\d\d-\d\d[^\]]*\].*$", "", body)      # 조사 메모 꼬리 제거
    if len(body) > 90:
        body = body[:88].rstrip() + "…"
    need = "예약 필요 · " if str(m.get("reservation_required") or "") == "Y" else ""
    weak = " (참고용·변동 가능)" if _weak(m) else ""
    return f"{head} — {need}{body}{weak}"


def _rank(m, kind) -> int:
    """낮을수록 먼저. 확정 > 참고용, 예약·요금·총량 > 개별 주차장 표시."""
    s = 10 if _weak(m) else 0
    access = str(m.get("access_code") or "") + str(m.get("field_name") or "")
    if kind == "parking":
        if "RESERVATION" in access:
            s -= 5                                   # 예약제는 모르면 입장 자체가 막힌다
        if access in ("PARKING", "parking_total", "parking_fee") or "GENERAL" in access:
            s -= 3
        if access.startswith("DRIVE"):
            s += 2
    else:
        if str(m.get("mode") or "") == "SUBWAY" or "SUBWAY" in access:
            s -= 3
        if str(m.get("mode") or "") == "SHUTTLE":
            s -= 2
    return s


def access_lines(rows, mode, taxi=False, limit=2) -> list[str]:
    """TRANSPORT metadata 목록 → 안내 줄. 도보·모름이면 빈 목록."""
    want = {"car": "parking", "transit": "transit"}.get(mode)
    if not want or (mode == "car" and taxi):
        return []
    picked = sorted((m for m in rows if _kind(m) == want), key=lambda m: _rank(m, want))
    prefix = "주차" if want == "parking" else "오는 길"
    seen, out = set(), []
    for m in picked:
        line = _text(m)
        if line in seen:
            continue
        seen.add(line)
        out.append(f"{prefix}: {line}")
        if len(out) >= limit:
            break
    return out


def notes(mode, taxi=False) -> list[str]:
    """이동수단별로 한 줄씩 덧붙이는 안내 (숫자 없는 일반 안내만)."""
    if mode == "car" and taxi:
        return ["택시는 경기 직후 구장 앞이 붐비니, 한두 블록 떨어진 곳에서 잡는 게 편해요."]
    if mode == "car":
        return ["운전하셔서 술집은 뺐어요. 경기 직후엔 주차장 출차가 몰리니 여유 있게 움직이세요."]
    if mode == "transit":
        return ["경기 끝나는 시각에 역이 붐비니, 돌아가는 길은 한 정거장 걸어서 타는 것도 방법이에요."]
    return []


def _meta(m):
    if isinstance(m, str):
        try:
            m = json.loads(m)
        except (json.JSONDecodeError, TypeError):
            return {}
    return m if isinstance(m, dict) else {}


def fetch_rows(code) -> list[dict]:
    """구장의 TRANSPORT 청크 metadata 전부 (많아야 10행 안팎)."""
    from django.db import connection                 # 순수 함수 테스트를 위해 지연 import
    with connection.cursor() as cur:
        cur.execute("""SELECT metadata FROM llm_documentchunk
                       WHERE metadata->>'category' = 'TRANSPORT' AND metadata->>'stadium_code' = %s""", [code])
        return [_meta(r[0]) for r in cur.fetchall()]


def info(code, mode, question="") -> dict:
    """agent 진입점. DB 가 없어도(테스트·장애) 안내만 빠지고 코스는 나간다."""
    taxi = is_taxi(question)
    lines = []
    if mode in ("car", "transit"):
        try:
            lines = access_lines(fetch_rows(code), mode, taxi=taxi)
        except Exception:                            # 안내 줄은 부가 정보 — 실패해도 코스는 준다
            lines = []
    return {"mode": mode or "walk", "label": LABEL.get(mode or "walk"), "taxi": taxi,
            "lines": lines, "notes": notes(mode, taxi)}
