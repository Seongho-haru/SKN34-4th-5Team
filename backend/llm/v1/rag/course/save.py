"""[course] 챗봇이 짠 코스 → travel 앱 코스 저장 payload.

챗봇이 답만 하고 끝나지 않고, 그 코스를 그대로 우리 서비스의 "코스"로 저장하게 하는 다리.
    챗봇 답변 places[]  ─(이 파일)→  coursePayload  ─(프론트가 POST)→  /courses/  →  Course + CourseStop

프론트는 답변에 딸려 온 coursePayload 를 그대로 POST 하기만 하면 된다 (필드 조립 불필요).
백엔드에 새 엔드포인트를 만들지 않으므로 성호 파일(travel/)은 건드리지 않는다.

## travel 실제 계약 (backend/travel/serializers.py, 2026-09-15 develop 기준)

    POST /courses/          ← travel.urls 가 루트에 붙어 있어 /travel/ 접두어가 없다
    Course  쓰기 가능: title · stadium · content(≤12000) · contentFormat("" | "html")
                      · duration(필수) · tags(문자열 배열) · startLat · startLng · stops
            읽기 전용: id · description · cover · author · likes · views · isSample · createdAt …
    Stop    position(0부터 빈틈없이) · name · lat · lng · category
            · placeId · visitId · address · tourContentId · isMapPoint · isDrawnPoint
            stops 는 1~12개
    응답에 editToken 이 들어온다 → 프론트가 localStorage 에 저장 (course-api.ts 가 이미 처리)
    POST 는 same-site 만 받는다(Origin 검사) → 반드시 브라우저에서 호출할 것

## 우리 값 중 CourseStop 에 자리가 없는 것

    time(도착 시각) · stayMin(체류) · reason(고른 이유) · phase(경기 전/후)
    → 이건 Course.content 에 타임라인 글로 넣는다. 그게 content 필드의 용도다.
"""

MAX_TITLE = 80
MAX_STADIUM = 120
MAX_DURATION = 80
MAX_CONTENT = 12000
MAX_STOPS = 12
PHASE_KO = {"BEFORE": "경기 전", "GAME": "경기 관람", "AFTER": "경기 후"}


def course_payload(places, *, stadium_ko=None, game=None, walk_summary="",
                   total_min=0, slots_info=None, travel_info=None):
    """places[] (챗봇 답변) → POST /courses/ 에 그대로 넣을 수 있는 dict.

    places      agent.answer() 가 돌려준 목록 (phase·name·lat·lng·category·address·placeId·time·stayMin·reason)
    game        {"date","time","home","away"} | None
    slots_info  slots.parse() 결과 (태그용). 없어도 된다
    travel_info transport.info() 결과 (이동수단 태그·주차/교통 안내 줄). 없어도 된다
    반환        {"title","stadium","content","contentFormat","duration","tags","startLat","startLng","stops"}
                좌표가 없는 장소는 stops 에서 빠진다 (CourseStop 은 lat/lng 가 필수라 400 이 난다)
    """
    sl = slots_info or {}
    tr = travel_info or {}
    usable = [p for p in places if p.get("lat") is not None and p.get("lng") is not None][:MAX_STOPS]

    stops = [{
        "position": i,                                   # 0부터 빈틈없이 — serializer 가 검사한다
        "name": (p.get("name") or "")[:255],
        "lat": p["lat"],
        "lng": p["lng"],
        "category": (p.get("category") or "SPOT")[:120],
        "placeId": p.get("placeId") or None,
        "address": (p.get("address") or "")[:500] or None,
        "isMapPoint": True,                              # 챗봇이 지도 좌표로 찍어 준 지점
    } for i, p in enumerate(usable)]

    first = usable[0] if usable else {}
    return {
        "title": _title(stadium_ko, game, sl.get("companionLabel"))[:MAX_TITLE],
        "stadium": (stadium_ko or "")[:MAX_STADIUM],
        "content": _content(usable, game, walk_summary, sl, tr)[:MAX_CONTENT],
        "contentFormat": "",                             # 일반 텍스트 (choices 는 "" 아니면 "html")
        "duration": _duration(total_min)[:MAX_DURATION],
        "tags": _tags(game, sl, tr),
        "startLat": first.get("lat"),
        "startLng": first.get("lng"),
        "stops": stops,
    }


def _title(stadium_ko, game, companion_label):
    head = stadium_ko or "직관"
    if game:
        head = f"{'-'.join(game['date'].split('-')[1:])} {head}"
    tail = f" ({companion_label})" if companion_label else ""
    return f"{head} 직관 코스{tail}"


def _duration(total_min):
    """'약 6시간 20분'. 시간표에서 계산한 코스 전체 소요시간."""
    total_min = int(total_min or 0)
    if total_min <= 0:
        return "약 반나절"
    h, m = divmod(total_min, 60)
    if h and m:
        return f"약 {h}시간 {m}분"
    return f"약 {h}시간" if h else f"약 {m}분"


def _tags(game, sl, tr=None):
    """검색·필터용 태그. 중복 없이, 빈 값 없이."""
    tags = ["직관코스", "챗봇추천"]
    if game:
        tags.append(f"{game.get('home', '')} vs {game.get('away', '')}".strip())
        tags.append("야간경기" if game.get("time", "") >= "17:00" else "낮경기")
    if sl.get("companionLabel"):
        tags.append(sl["companionLabel"])
    if (tr or {}).get("mode") in ("car", "transit"):
        tags.append("택시" if tr.get("taxi") else tr.get("label"))
    tags += list(dict.fromkeys(sl.get("prefs") or []))[:3]
    return [t for t in dict.fromkeys(tags) if t and t != "vs"]


def _content(places, game, walk_summary, sl, tr=None):
    """Course.content — 타임라인 글. CourseStop 에 자리가 없는 시각·이유가 여기 들어간다."""
    lines = []
    if game:
        lines.append(f"{game['date']} {game['time']} · {game.get('home', '')} 홈 vs {game.get('away', '')} 원정")
    for p in places:
        when = p.get("time") or ""
        stay = f" ({p['stayMin']}분)" if p.get("stayMin") else ""
        why = f" — {p['reason']}" if p.get("reason") else ""
        lines.append(f"{when}  [{PHASE_KO.get(p.get('phase'), '')}] {p.get('name', '')}{stay}{why}")
    if walk_summary:
        lines.append("")
        lines.append(walk_summary)
    lines += (tr or {}).get("lines") or []
    if sl.get("note"):
        lines.append(sl["note"])
    lines += (tr or {}).get("notes") or []
    lines.append("카카오맵 기준 정보라 가시기 전에 영업 여부만 한 번 확인해 보세요!")
    return "\n".join(lines)
