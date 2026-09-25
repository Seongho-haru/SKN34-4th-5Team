"""[course] 질문에서 슬롯 뽑기 — 취향 · 동행 · 여유시간 · 재추천. 전부 키워드 규칙이라 LLM 호출 0회.

코스를 가장 크게 바꾸는 건 음식 종류보다 **누구랑 가느냐** 다.
"애기랑 잠실 코스" 와 "회사 사람들이랑 잠실 코스" 는 같은 구장·같은 경기라도 답이 달라야 한다.

슬롯 6개
    prefs      치킨·카페·술 같은 취향 → 후보 점수 가산
    companion  아이·부모님·연인·혼자·회사·친구 → ban(제외) · boost(가산) · 반경 축소 · 안내 문구
    spare      tight(퇴근하고 바로) · normal · long(낮부터) → BEFORE 개수·명소 포함 여부
    retry      "다른 데 없어?" → 직전 추천 장소를 후보에서 뺀다
    scope      before(경기 전만) · after(경기 후만) · both → 그 구간만 짠다
    mode       walk · car · transit (transport.py) → 구간 시간 · 주차/대중교통 안내 · 운전이면 술집 제외
"""
import re

from . import transport
from ..nearby.agent import kinds_of as _nearby_kinds

# ── 취향 (음식·활동) : 질문 표현 → external_places 의 category_detail 에서 찾을 말 ──────
PREFERENCE = {
    "치킨": ["치킨"], "피자": ["피자"], "햄버거": ["햄버거"], "버거": ["햄버거"], "한식": ["한식"], "고기": ["육류", "고기"],
    "삼겹": ["육류", "고기"], "국밥": ["국밥"], "일식": ["일식", "초밥", "돈까스"], "초밥": ["초밥", "일식"],
    "중식": ["중식"], "양식": ["양식", "파스타"], "파스타": ["양식", "파스타"], "분식": ["분식", "떡볶이"],
    "떡볶이": ["분식", "떡볶이"], "면": ["국수", "냉면", "칼국수"], "냉면": ["냉면"],
    "카페": ["카페"], "커피": ["카페", "커피"], "디저트": ["디저트", "베이커리", "케이크"], "빵": ["베이커리"],
    "술": ["술집", "호프", "포장마차", "이자카야"], "맥주": ["술집", "호프"], "치맥": ["치킨", "호프"],
    "소주": ["술집", "한식"], "야식": ["술집", "치킨", "포장마차"], "안주": ["술집", "호프"],
    "산책": ["공원", "호수", "테마거리", "관광,명소"], "공원": ["공원"], "명소": ["관광,명소"],
    "구경": ["관광,명소"], "사진": ["관광,명소", "테마거리"], "조용": ["카페"],
}

# ── 동행 : 정규식 → 규칙 ─────────────────────────────────────────────────────
# ban   이 말이 category_detail 에 있으면 후보에서 아예 뺀다
# boost 있으면 점수 가산 (취향과 같은 크기)
# radius 이 동행이면 도보 반경을 줄인다 (아이·어르신은 멀리 못 걷는다). None 이면 기본 2500m
# note  답변 끝에 붙는 한 줄 (왜 이렇게 골랐는지 사용자에게 설명)
COMPANION = {
    "kid": dict(
        re=r"아이|애기|아기|애들|자녀|딸|아들|초등|유치원|어린이|가족",
        label="아이 동반",
        ban=["술집", "호프", "포장마차", "이자카야", "요리주점", "바(BAR)"],
        boost=["분식", "패밀리레스토랑", "돈까스", "피자", "햄버거", "공원", "아이스크림"],
        radius=1500,
        note="아이와 함께라 술집은 빼고, 자리 넓고 빨리 나오는 곳 위주로 골랐어요.",
    ),
    "parents": dict(
        re=r"부모님|어머니|아버지|엄마|아빠|할머니|할아버지|어른들|장인|장모",
        label="부모님 동반",
        ban=["패스트푸드", "햄버거"],
        boost=["한식", "국밥", "육류", "고기", "카페"],
        radius=1200,
        note="부모님과 함께라 많이 안 걷는 선에서, 앉아서 드시기 좋은 곳으로 골랐어요.",
    ),
    "couple": dict(
        re=r"여자친구|남자친구|여친|남친|애인|데이트|썸|아내|남편|와이프|커플",
        label="데이트",
        ban=[],
        boost=["카페", "디저트", "베이커리", "양식", "파스타", "관광,명소", "테마거리"],
        radius=None,
        note="데이트라 분위기 괜찮은 곳과 카페를 섞었어요.",
    ),
    "solo": dict(
        re=r"혼자|혼밥|나홀로|1인|혼자서|솔로",
        label="혼자",
        ban=[],
        boost=["분식", "국밥", "국수", "햄버거", "카페"],
        radius=None,
        note="혼자 가시는 거라 혼밥하기 편하고 빨리 나오는 곳 위주예요.",
    ),
    "company": dict(
        re=r"회사|직장|팀원|동료|부서|회식|상사|거래처",
        label="단체·회식",
        ban=["분식"],
        boost=["육류", "고기", "한식", "호프", "술집"],
        radius=None,
        note="여러 명이라 단체로 앉기 좋은 곳으로 골랐어요. 인원 많으면 미리 예약 확인해 보세요.",
    ),
    "friends": dict(
        re=r"친구|친구들|동기|동아리|형들|누나들|모임",
        label="친구들과",
        ban=[],
        boost=["치킨", "호프", "술집", "육류", "분식"],
        radius=None,
        note="친구들과 가시는 거라 같이 나눠 먹기 좋은 곳으로 골랐어요.",
    ),
}
_COMPANION_RE = {k: re.compile(v["re"]) for k, v in COMPANION.items()}

# ── 여유시간 ────────────────────────────────────────────────────────────────
TIGHT = re.compile(r"퇴근하고|퇴근\s*후|바로\s*(가|감|갈)|끝나고\s*바로|시간\s*(이\s*)?없|촉박|빠듯|늦게\s*도착|간단(히|하게)")
LONG = re.compile(r"낮부터|아침부터|일찍|하루\s*종일|종일|반차|연차|쉬는\s*날|여유\s*(있|롭)|일찍\s*가")

# ── 재추천 ("다른 데 없어?") ────────────────────────────────────────────────
RETRY = re.compile(r"다른\s*(데|곳|거|것|곳으로)|딴\s*데|말고\s*(다른)?|바꿔|다시\s*(추천|짜|골라)|새로\s*(추천|짜)|"
                   r"또\s*추천|별로(야|네|다|임)|맘에\s*안|마음에\s*안|싫어|바꿔줘")
# 우리 답변 형식 "- 경기 전: 가게명 — 이유" 와 타임라인 "16:40  가게명 (50분) — 이유" 에서 가게명만 뽑는다
_PREV_LIST = re.compile(r"^[-·•]\s*(?:경기\s*전|경기\s*관람|경기\s*후)\s*[:：]\s*(.+?)\s*(?:—|-|$)", re.M)
_PREV_TIME = re.compile(r"^\s*(?:익일\s*)?\d{1,2}:\d{2}\s{1,4}(.+?)\s*(?:\(\d+분\)|입장|—|$)", re.M)


# ── 범위 (경기 전만 / 경기 후만) ─────────────────────────────────────────────
# "일식이랑 카페 먹고 경기장 갈 거야" 는 경기 전만 묻는 말이다 — 경기 후 장소를 붙이면 안 된다.
# "산책 좀 하고 구장에 가고 싶어" 처럼 사이에 말이 끼거나 조사(에·으로)가 붙어도 경기 전만으로 본다.
BOTH = re.compile(r"전\s*후|앞\s*뒤|하루\s*종일|처음부터\s*끝까지")
BEFORE_ONLY = re.compile(
    r"경기\s*(보기|시작)?\s*전(에|에만|만)|경기\s*전\s*코스|(구장|경기장|야구장)\s*(가기|들어가기)\s*전|"
    r"(먹고|마시고|들렀다가?|갔다가?|구경하고|놀다가?|산책하고|하고)\s*(나서\s*)?(\S{1,6}\s+){0,2}([가-힣A-Za-z]{1,6}\s*)?"
    r"(경기장|야구장|구장|경기|직관|야구)(에|으로|로)?\s*(보러\s*)?(갈|가|들어|입장)")
AFTER_ONLY = re.compile(
    r"(경기|야구|직관)\s*(끝나고|끝난\s*(뒤|후)|후(?!기)에?만?|마치고|보고\s*나서|본\s*(뒤|후))|끝나고|뒤풀이")


def scope_of(question: str) -> str:
    """'before' | 'after' | 'both'. 한쪽만 분명할 때만 좁힌다 (애매하면 both)."""
    q = question or ""
    if BOTH.search(q):
        return "both"
    before, after = bool(BEFORE_ONLY.search(q)), bool(AFTER_ONLY.search(q))
    if before and not after:
        return "before"
    if after and not before:
        return "after"
    return "both"


def companion_of(question: str):
    """동행 키 (없으면 None). 여러 개 걸리면 제약이 센 쪽(아이 > 부모님 > 나머지)을 쓴다."""
    hits = [k for k, rx in _COMPANION_RE.items() if rx.search(question)]
    for k in ("kid", "parents", "company", "couple", "friends", "solo"):
        if k in hits:
            return k
    return None


def preferences(question: str):
    q = question.replace(" ", "")
    return [w for words in (v for k, v in PREFERENCE.items() if k in q) for w in words]


def spare_of(question: str) -> str:
    if TIGHT.search(question):
        return "tight"
    if LONG.search(question):
        return "long"
    return "normal"


def previous_places(history) -> set:
    """직전 봇 답변에서 추천했던 장소 이름들 (재추천 때 제외하려고)."""
    names = set()
    for m in reversed(history or []):
        if m.get("role") != "assistant":
            continue
        text = m.get("content") or ""
        found = _PREV_LIST.findall(text) + _PREV_TIME.findall(text)
        if found:
            names |= {n.strip() for n in found if n.strip()}
            break                      # 가장 최근 추천 한 번만 본다
    return names


def parse(question: str, history=None) -> dict:
    """질문 → 슬롯 묶음. agent 는 이 결과만 보고 후보를 거른다."""
    key = companion_of(question)
    comp = COMPANION.get(key) or {}
    retry = bool(RETRY.search(question))
    mode = transport.mode_of(question)
    if mode is None:                    # "아까 차로 간다고 했잖아" — 이번 질문에 없으면 직전 사용자 말에서 잇는다
        for m in reversed(history or []):
            if m.get("role") == "user" and (mode := transport.mode_of(m.get("content") or "")):
                break
    taxi = transport.is_taxi(question)
    ban = list(dict.fromkeys(list(comp.get("ban") or []) + transport.ban_words(mode, taxi)))
    return {
        "prefs": preferences(question),
        "companion": key,
        "companionLabel": comp.get("label"),
        "ban": ban,
        "boost": list(comp.get("boost") or []),
        "radius": comp.get("radius"),
        "note": comp.get("note"),
        "spare": spare_of(question),
        "retry": retry,
        "exclude": previous_places(history) if retry else set(),
        "scope": scope_of(question),
        # RAG 에 없는 종류 — 카카오 실시간 후보를 더한다 (편의점은 코스에 넣지 않는다)
        "extras": [k for k in _nearby_kinds(question) if k in ("stay", "walk", "indoor")],
        "mode": mode,
        "taxi": taxi,
    }


def prompt_line(slots: dict) -> str:
    """LLM 에 넘길 상황 한 줄 (없으면 빈 문자열)."""
    bits = []
    if slots["prefs"]:
        bits.append(f"취향: {', '.join(dict.fromkeys(slots['prefs']))}")
    if slots["companionLabel"]:
        bits.append(f"동행: {slots['companionLabel']}")
    if slots["spare"] == "tight":
        bits.append("여유: 촉박함 (경기 전은 한 곳만, 구장 가까운 곳으로)")
    elif slots["spare"] == "long":
        bits.append("여유: 넉넉함 (경기 전에 명소·산책을 한 곳 넣어도 좋음)")
    if slots.get("scope") == "before":
        bits.append("범위: 경기 전만 (AFTER 는 고르지 말고, BEFORE 를 취향 개수만큼 1~3곳)")
    elif slots.get("scope") == "after":
        bits.append("범위: 경기 후만 (BEFORE 는 고르지 말고, AFTER 를 1~3곳)")
    if slots.get("mode") == "car" and slots.get("taxi"):
        bits.append("이동: 택시")
    elif slots.get("mode") == "car":
        bits.append("이동: 자동차 (운전하므로 술집·주류 위주 장소는 고르지 말 것)")
    elif slots.get("mode") == "transit":
        bits.append("이동: 대중교통")
    elif slots.get("mode") == "walk":
        bits.append("이동: 도보")
    extras = slots.get("extras") or []
    if "stay" in extras:
        bits.append("추가 요청: 숙박 — 코스 맨 마지막(AFTER 끝)에 [STAY] 후보 1곳")
    if "walk" in extras:
        bits.append("추가 요청: 산책 — [WALK] 후보 1곳을 코스에 넣을 것")
    if "indoor" in extras:
        bits.append("추가 요청: 실내 놀거리 — [INDOOR] 후보 1곳을 코스에 넣을 것")
    if slots["retry"]:
        bits.append("재추천: 앞서 추천한 곳은 후보에서 빠졌으니 새로 골라 줄 것")
    return " / ".join(bits)
