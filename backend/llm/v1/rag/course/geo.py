"""[course] 동선 계산 — 하버사인 거리 · 방위 · 총 도보 · 동선이 나쁠 때 가까운 후보로 교체. LLM 호출 0회.

왜 필요한가: 후보를 "구장에서 몇 m" 로만 고르면, 경기 전 장소가 구장 북쪽 2.4km,
경기 후 장소가 남쪽 2.3km 로 잡혀 실제 걷는 거리가 5km 가까이 나올 수 있다.
실제 경로는 BEFORE → 구장 → AFTER 이므로 구간별로 재야 한다.

두 군데서 쓴다
  1) 후보를 LLM 에 보여줄 때  방위를 같이 준다 ("구장 북동쪽 900m") → LLM 이 한쪽 방향으로 모아서 고른다
  2) LLM 이 고른 뒤          총 도보를 재고, 너무 길면 같은 카테고리의 더 가까운 후보로 바꾼다
"""
import math

WALK_M_PER_MIN = 80
SWAP_GAIN_M = 500          # 이만큼 넘게 줄어들 때만 교체한다 (LLM 의 선택을 함부로 뒤집지 않는다)
LONG_ROUTE_M = 3000        # 총 도보가 이보다 길면 교체를 시도한다
LONG_LEG_M = 1800          # 한 구간만 유독 멀어도 교체를 시도한다 (경기 끝나고 22시에 30분 걷기 방지)
COMPASS = ["북", "북동", "동", "남동", "남", "남서", "서", "북서"]


def haversine_m(lat1, lng1, lat2, lng2):
    """두 좌표 사이 직선거리(m). 좌표가 없으면 None."""
    if None in (lat1, lng1, lat2, lng2):
        return None
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_label(anchor, p):
    """구장 기준 8방위. '북동' 처럼 돌려준다. 좌표 없으면 ''."""
    if None in (anchor.get("lat"), anchor.get("lng"), p.get("lat"), p.get("lng")):
        return ""
    dy = p["lat"] - anchor["lat"]
    dx = (p["lng"] - anchor["lng"]) * math.cos(math.radians(anchor["lat"]))
    if dx == 0 and dy == 0:
        return ""
    deg = (math.degrees(math.atan2(dx, dy)) + 360) % 360
    return COMPASS[int((deg + 22.5) // 45) % 8]


def leg_meters(points):
    """연속 구간 거리 목록. points = [{"lat","lng"}...] · 좌표 없는 구간은 None."""
    return [haversine_m(a.get("lat"), a.get("lng"), b.get("lat"), b.get("lng"))
            for a, b in zip(points, points[1:])]


def leg_minutes(points):
    return [max(1, round(m / WALK_M_PER_MIN)) if m else 10 for m in leg_meters(points)]


def total_walk(points):
    """(총 미터, 총 분). 좌표가 빠진 구간은 빼고 더한다."""
    legs = [m for m in leg_meters(points) if m]
    meters = sum(legs)
    return int(meters), max(1, round(meters / WALK_M_PER_MIN)) if meters else 0


def summary(points):
    m, mins = total_walk(points)
    if not m:
        return ""
    km = f"{m / 1000:.1f}km" if m >= 1000 else f"{m}m"
    return f"총 도보 약 {km} · {mins}분"


def _points(course, lookup):
    return [lookup[c["key"]] for c in course if c["key"] in lookup]


def optimize(course, lookup, cands, reason_text="구장에서 가까워 이동이 짧아요"):
    """동선이 너무 길면 BEFORE/AFTER 한 곳을 같은 카테고리의 더 가까운 후보로 바꾼다.

    LLM 이 쓴 이유는 그 장소에 대한 설명이라 교체하면 안 맞는다 → 교체한 칸만 코드 문구로 갈아끼운다.
    반환 (course, swapped: bool). course 는 새 목록이고 원본은 건드리지 않는다.
    """
    course = [dict(c) for c in course]
    points = _points(course, lookup)
    base_m, _ = total_walk(points)
    worst_leg = max([m for m in leg_meters(points) if m] or [0])
    if base_m <= LONG_ROUTE_M and worst_leg <= LONG_LEG_M:
        return course, False

    used = {c["key"] for c in course}
    best = None
    for i, c in enumerate(course):
        if c["phase"] == "GAME":
            continue
        cur = lookup.get(c["key"], {})
        for alt in cands:
            if alt["key"] in used or alt["category"] != cur.get("category"):
                continue
            trial = [dict(x) for x in course]
            trial[i] = {**c, "key": alt["key"]}
            m, _ = total_walk(_points(trial, {**lookup, alt["key"]: alt}))
            gain = base_m - m
            if gain >= SWAP_GAIN_M and (best is None or gain > best[0]):
                best = (gain, i, alt)
    if not best:
        return course, False
    _, i, alt = best
    course[i] = {**course[i], "key": alt["key"], "reason": reason_text}
    return course, True


# ── 출발지 기준 이어 짜기 ─────────────────────────────────────────────────────
# 사용자가 지도에서 출발지를 찍었으면, 각 장소는 "바로 앞 지점"을 중심으로 다시 고른다.
#   출발지 → BEFORE 1 (출발지 근처) → BEFORE 2 (BEFORE 1 근처) → 구장 → AFTER 1 (구장 근처) → AFTER 2 (AFTER 1 근처)
# 단, 종착지(구장)에서 앞 지점보다 멀어지는 곳은 막는다 — 구장 반대쪽으로 끌려가는 코스를 만들지 않는다.
HOP_SCALE_M = 1000        # 앞 지점에서 1km 떨어질 때마다 점수 -1 (관련도 차이보다 거리가 우선)
AWAY_SCALE_M = 500        # 앞 지점보다 구장에서 500m 멀어질 때마다 점수 -1
AWAY_LIMIT_M = 1000       # 앞 지점보다 구장에서 이만큼 넘게 멀어지는 곳은 후보에서 뺀다
KEEP_MARGIN = 0.15        # LLM 이 고른 곳이 최선과 이 차이 안이면 그대로 둔다 (LLM 이 쓴 이유를 살린다)


def _has_xy(p):
    return p is not None and None not in (p.get("lat"), p.get("lng"))


def _dist(a, b):
    return haversine_m(a.get("lat"), a.get("lng"), b.get("lat"), b.get("lng"))


def away_from_anchor(p, prev, anchor):
    """앞 지점보다 구장에서 몇 m 더 멀어지는가 (가까워지면 0). 구장 좌표가 없으면 0."""
    if not (_has_xy(anchor) and _has_xy(p) and _has_xy(prev)):
        return 0.0
    return max(0.0, _dist(p, anchor) - _dist(prev, anchor))


def chain_score(p, prev, anchor, relevance):
    return relevance(p) - _dist(prev, p) / HOP_SCALE_M - away_from_anchor(p, prev, anchor) / AWAY_SCALE_M


def chain_course(course, lookup, cands, origin, anchor, relevance, same_kind):
    """LLM 이 정한 코스의 칸(단계·종류·개수)은 그대로 두고, 각 칸의 장소를 앞 지점 기준으로 다시 고른다.

    relevance(p) -> float   질문·취향과 얼마나 맞는가 (클수록 좋다)
    same_kind(a, b) -> bool 같은 칸에 들어갈 수 있는 종류인가 (식사↔식사, 술집↔술집 …)
    반환 (course, changed: bool). 원본은 건드리지 않는다.
    """
    course = [dict(c) for c in course]
    if not _has_xy(origin):
        return course, False
    used = {c["key"] for c in course}
    changed = False
    prev, prev_label = origin, "출발지"
    for i, c in enumerate(course):
        if c["phase"] == "GAME":
            prev, prev_label = (anchor, "구장") if _has_xy(anchor) else (prev, prev_label)
            continue
        cur = lookup.get(c["key"])
        if not _has_xy(cur):
            continue
        pool = [p for p in cands if _has_xy(p) and same_kind(p, cur) and (p["key"] == c["key"] or p["key"] not in used)]
        allowed = [p for p in pool if away_from_anchor(p, prev, anchor) <= AWAY_LIMIT_M] or pool
        best = max(allowed, key=lambda p: chain_score(p, prev, anchor, relevance))
        keep = cur in allowed and (chain_score(best, prev, anchor, relevance)
                                   - chain_score(cur, prev, anchor, relevance)) <= KEEP_MARGIN
        if best["key"] != c["key"] and not keep:
            used.discard(c["key"])
            used.add(best["key"])
            minutes = max(1, round(_dist(prev, best) / WALK_M_PER_MIN))
            course[i] = {**c, "key": best["key"], "reason": f"{prev_label}에서 도보 {minutes}분"}
            changed = True
            cur = best
        prev, prev_label = cur, "앞 장소"
    return course, changed


# ── 출발지부터 한 단계씩 검색해 나가기 ────────────────────────────────────────
# 후보를 "앞 지점 주변"에서 새로 찾고, 그중 앞 지점과 가까우면서 구장 쪽으로 다가가는 곳을 고른다.
PROGRESS_SCALE_M = 800     # 구장에 800m 다가갈 때마다 +1 (HOP_SCALE_M 보다 작아서, 구장 방향으로 가는 걸음은 이득)
AWAY_SLACK_M = 300         # 경기 전 단계는 앞 지점보다 구장에서 이 이상 멀어지면 고르지 않는다


def progress_m(p, prev, anchor):
    """앞 지점보다 구장에 몇 m 가까워지는가 (멀어지면 음수). 좌표가 없으면 0."""
    if not (_has_xy(anchor) and _has_xy(p) and _has_xy(prev)):
        return 0.0
    return _dist(prev, anchor) - _dist(p, anchor)


def step_allowed(p, prev, anchor, phase, after_max_m):
    """경기 전: 구장에서 앞 지점보다 크게 멀어지지 않는다. 경기 후: 구장에서 너무 멀리 가지 않는다."""
    if not _has_xy(p):
        return False
    if not _has_xy(anchor):
        return True
    if phase == "BEFORE":
        return progress_m(p, prev, anchor) >= -AWAY_SLACK_M
    return _dist(p, anchor) <= max(after_max_m, _dist(prev, anchor) + AWAY_SLACK_M)


def step_score(p, prev, anchor, relevance):
    """앞 지점과 가까울수록, 구장에 다가갈수록, 질문과 잘 맞을수록 높다."""
    return relevance(p) - _dist(prev, p) / HOP_SCALE_M + progress_m(p, prev, anchor) / PROGRESS_SCALE_M
