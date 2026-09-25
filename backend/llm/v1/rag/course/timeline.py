"""[course] 코스 시간표 — 경기 시작 시각에서 역산해 각 장소의 도착·출발 시각을 계산한다. LLM 호출 0회.

우리 챗봇은 경기 시작 시각을 실제 데이터로 알고 있다(club.structured.games()).
그래서 "어디" 뿐 아니라 "몇 시에" 까지 말해 줄 수 있다 — 일반 맛집 추천과 갈리는 지점.

기준값 (전부 상수라 바꾸기 쉽다)
    경기 시간  KBO 9이닝 평균 3시간 10분(190분). 연장·우천은 알 수 없으니 답변에서 "예상"이라고만 쓴다
    입장       경기 시작 45분 전 (매점·좌석·응원 준비)
    도보       80m/분 — 반경 2.5km 정책과 같은 기준 (먹거리_플레이스_반경 정책 2026-09-08)
    체류       음식점 50분 · 카페 40분 · 명소 40분 · 술집 80분

계산 방향
    BEFORE 는 구장 입장 시각에서 거꾸로(체류·이동을 빼면서), AFTER 는 경기 종료 예상 시각에서 앞으로.
    그래서 "경기 시작 18:30" 하나만 알면 나머지 시각이 전부 따라 나온다.
"""

GAME_DURATION_MIN = 190      # KBO 9이닝 평균 경기 시간 (약 3시간 10분)
ENTER_BEFORE_MIN = 45        # 경기 시작 몇 분 전에 구장 도착을 권하는지
WALK_M_PER_MIN = 80          # 도보 평속 (반경 정책과 동일)
DEFAULT_LEG_MIN = 10         # 좌표가 없어 이동시간을 못 재는 구간의 기본값

STAY_MIN = {"FOOD": 50, "CAFE": 40, "SPOT": 40, "BAR": 80, "INDOOR": 60, "STAY": None, "STADIUM": None}
BAR_WORDS = ("술집", "호프", "포장마차", "이자카야", "맥주", "바(BAR)", "요리주점", "실내포장마차")


def walk_min(meters) -> int:
    """미터 → 도보 분 (최소 1분)"""
    if not meters:
        return 0
    return max(1, round(meters / WALK_M_PER_MIN))


def to_min(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def to_hhmm(minutes: int) -> str:
    """분 → 'HH:MM'. 자정을 넘기면 '익일 01:10' 으로 표시한다 (야간 경기 후 코스)."""
    minutes = int(round(minutes))
    day, rest = divmod(minutes % (60 * 24 * 2), 60 * 24)
    s = f"{rest // 60:02d}:{rest % 60:02d}"
    return f"익일 {s}" if day else s


def kind_of(place) -> str:
    """체류시간·표시용 종류. FOOD 중 술집 업종은 BAR 로 따로 본다 (경기 후 체류가 길다)."""
    cat = place.get("category") or ""
    if cat in ("STADIUM", "STAY", "INDOOR"):
        return cat
    detail = place.get("detail") or place.get("category_detail") or ""
    if cat in ("FOOD", "FOOD_OUT") and any(w in detail for w in BAR_WORDS):
        return "BAR"
    return {"FOOD_OUT": "FOOD", "FOOD": "FOOD", "CAFE": "CAFE", "SPOT": "SPOT", "WALK": "SPOT"}.get(cat, "SPOT")


def stay_min(place, phase: str) -> int:
    k = kind_of(place)
    if k in ("STADIUM", "STAY"):             # 숙소는 코스의 끝 — 머무는 시간을 세지 않는다
        return 0
    base = STAY_MIN.get(k) or 40
    if phase == "BEFORE" and k == "BAR":     # 경기 전에는 오래 못 앉아 있는다
        return 50
    return base


def build(course, lookup, game_time: str, leg_minutes=None) -> dict:
    """코스에 시각을 붙인다.

    course       : [{"key","phase","reason"}, ...]  — BEFORE… GAME … AFTER 순으로 정렬돼 있어야 한다
    lookup       : key → 장소 dict (category·detail 포함)
    game_time    : "18:30"  경기 시작 시각
    leg_minutes  : 연속 구간 이동시간(분) 목록, 길이 len(course)-1. None 이면 전부 DEFAULT_LEG_MIN

    반환 {"rows": [{"key","time","until","stayMin","legMin"}...],
          "gameStart","gameEnd","startTime","endTime","totalMin"}
    """
    n = len(course)
    legs = list(leg_minutes or [])[: max(0, n - 1)]
    legs += [DEFAULT_LEG_MIN] * (max(0, n - 1) - len(legs))

    g = next((i for i, c in enumerate(course) if c["phase"] == "GAME"), None)
    start = to_min(game_time)
    rows = [{"key": c["key"], "time": None, "until": None,
             "stayMin": stay_min(lookup.get(c["key"], {}), c["phase"]),
             "legMin": legs[i] if i < len(legs) else 0} for i, c in enumerate(course)]

    if g is None:                                    # 방어: 구장이 없으면 경기 시작을 첫 칸으로 보고 앞으로만 계산
        g, start_arrive = 0, start
    else:
        start_arrive = start - ENTER_BEFORE_MIN
    rows[g]["time"] = start_arrive
    rows[g]["until"] = start + GAME_DURATION_MIN     # 경기 종료 예상

    for i in range(g - 1, -1, -1):                   # BEFORE — 뒤에서 앞으로 역산
        depart = rows[i + 1]["time"] - legs[i]
        rows[i]["until"] = depart
        rows[i]["time"] = depart - rows[i]["stayMin"]

    for i in range(g + 1, n):                        # AFTER — 앞에서 뒤로
        rows[i]["time"] = rows[i - 1]["until"] + legs[i - 1]
        rows[i]["until"] = rows[i]["time"] + rows[i]["stayMin"]

    first, last = rows[0]["time"], rows[-1]["until"]
    return {
        "rows": [{**r, "time": to_hhmm(r["time"]), "until": to_hhmm(r["until"])} for r in rows],
        "gameStart": to_hhmm(start), "gameEnd": to_hhmm(start + GAME_DURATION_MIN),
        "startTime": to_hhmm(first), "endTime": to_hhmm(last), "totalMin": int(last - first),
    }


def text_lines(course, lookup, tl, label=None) -> list[str]:
    """답변에 붙일 타임라인 줄들. '16:40  ○○식당 (50분)' 형태."""
    label = label or {"BEFORE": "경기 전", "GAME": "경기 관람", "AFTER": "경기 후"}
    out = []
    for c, r in zip(course, tl["rows"]):
        p = lookup.get(c["key"], {})
        if c["phase"] == "GAME":
            out.append(f"{r['time']}  {p.get('name', '구장')} 입장 · {tl['gameStart']} 경기 시작 "
                       f"(종료 {tl['gameEnd']} 예상)")
        elif kind_of(p) == "STAY":
            out.append(f"{r['time']}  {p.get('name', '')} (숙소 도착) — {c.get('reason', '')}")
        else:
            out.append(f"{r['time']}  {p.get('name', '')} ({r['stayMin']}분) — {c.get('reason', '')}")
    return out
