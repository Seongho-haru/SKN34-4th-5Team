"""RAG 가 제대로 붙었는지 한 번에 확인 — shell 에 붙여넣기 없이 한 줄로.

    docker compose exec backend python manage.py rag_check
    docker compose exec backend python manage.py rag_check -q "잠실 코스 짜줘"
    docker compose exec backend python manage.py rag_check --course    # 코스 추천까지 확인

확인 항목
    1) 환경변수   CHAT_USE_RAG · LLM_MODEL · EMBEDDING_MODEL · LANGSMITH_*
    2) 스위치     use_rag() / chat_chain() — 지금 챗봇이 RAG 로 도는지
    3) 인덱스     llm_documentchunk 건수 (0 이면 build_index 필요)
    4) 실제 호출  질문 하나를 끝까지 태워 본다 (reasoning_effort 등 모델 파라미터 검증 포함)
"""
import os
import time

from django.core.management.base import BaseCommand
from django.db import connection

MASK = ("API_KEY", "PASSWORD", "SECRET", "SIGNING")


def show(k):
    v = os.getenv(k)
    if v is None:
        return "(없음)"
    if any(m in k for m in MASK):
        return f"설정됨 ({len(v)}자)" if v else "(빈 값)"
    return v or "(빈 값)"


class Command(BaseCommand):
    help = "RAG 연결 상태를 점검한다 (환경변수 · 스위치 · 인덱스 · 실제 호출)"

    def add_arguments(self, parser):
        parser.add_argument("-q", "--question", default="LG 지금 몇 위야?")
        parser.add_argument("--course", action="store_true", help="코스 추천 질문으로 확인")
        parser.add_argument("--stadium", default=None, help='예: "잠실야구장"')
        parser.add_argument("--twice", action="store_true",
                            help="같은 프로세스에서 두 번 호출해 콜드/웜 차이를 본다 (실서버는 웜 상태로 돈다)")

    def handle(self, *a, **o):
        ok, ng = self.style.SUCCESS, self.style.ERROR
        p = self.stdout.write

        p("\n[1] 환경변수")
        for k in ("CHAT_USE_RAG", "LLM_MODEL", "EMBEDDING_MODEL",
                  "LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
            p(f"    {k:<20} {show(k)}")

        p("\n[2] 스위치")
        try:
            from llm.v1.rag.pipeline import chat_chain, use_rag
        except Exception as e:
            p(ng(f"    pipeline import 실패: {e!r}"))
            return
        on = use_rag()
        p(f"    use_rag()            {ok('True  → RAG 로 답한다') if on else ng('False → 예전 helpful-assistant 체인')}")
        p(f"    chat_chain()         {type(chat_chain()).__name__ if chat_chain() else 'None'}")
        if not on:
            p("    (CHAT_USE_RAG=1 로 두고 backend 를 restart 하세요)")

        p("\n[3] 인덱스")
        try:
            with connection.cursor() as cur:
                cur.execute("SELECT count(*) FROM llm_documentchunk")
                n = cur.fetchone()[0]
                cur.execute("SELECT metadata->>'category', count(*) FROM llm_documentchunk "
                            "GROUP BY 1 ORDER BY 2 DESC LIMIT 8")
                rows = cur.fetchall()
            p(f"    청크 {ok(str(n)) if n else ng('0 — build_index 필요')} 건")
            for c, k in rows:
                p(f"      {c or '(없음)':<12} {k}")
        except Exception as e:
            p(ng(f"    DB 조회 실패: {e!r}"))

        p("\n[3-1] 코스용 메타데이터 (카카오 장소 청크)")
        try:
            import json as _json
            with connection.cursor() as cur:
                cur.execute("""SELECT metadata FROM llm_documentchunk
                               WHERE metadata->>'category' = 'FOOD_OUT' LIMIT 1""")
                row = cur.fetchone()
            if not row:
                p(ng("    FOOD_OUT 청크가 없습니다 — 카카오 장소가 적재 안 됐어요"))
            else:
                raw = row[0]
                p(f"    파이썬 타입   {type(raw).__name__}"
                  f"{'  (str 로 와서 _meta() 가 파싱합니다)' if isinstance(raw, str) else ''}")
                m = _json.loads(raw) if isinstance(raw, str) else raw
                need = ("name", "lat_y", "lng_x", "distance_m", "category_detail",
                        "kakao_place_id", "address", "stadium_code", "doc_id")
                miss = [k for k in need if not m.get(k)]
                p(f"    필요 키      {ok('전부 있음') if not miss else ng('빠짐: ' + ', '.join(miss))}")
                p(f"    예시         {m.get('name')} · {m.get('category_detail')} · "
                  f"{m.get('distance_m')}m · ({m.get('lat_y')}, {m.get('lng_x')})")
            with connection.cursor() as cur:
                cur.execute("""SELECT metadata FROM llm_documentchunk
                               WHERE metadata->>'category' = 'STADIUM'
                                 AND metadata->>'stadium_code' = 'JAMSIL' LIMIT 1""")
                row = cur.fetchone()
            if not row:
                p(ng("    JAMSIL STADIUM 청크 없음 — 코스에 구장 좌표를 못 붙입니다"))
            else:
                m = _json.loads(row[0]) if isinstance(row[0], str) else row[0]
                p(f"    구장 앵커    {m.get('stadium_name_ko')} ({m.get('lat_y')}, {m.get('lng_x')})")
        except Exception as e:
            p(ng(f"    확인 실패: {e!r}"))

        p("\n[4] 실제 호출")
        q = "잠실에서 친구들이랑 첫 직관인데 경기 전후 코스 짜줘. 치킨 좋아해" if o["course"] else o["question"]
        p(f"    질문: {q}")
        try:
            from llm.v1.rag.pipeline import answer
            t0 = time.perf_counter()
            r = answer(q, stadium_name=o["stadium"])
            ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            p(ng(f"    실패: {type(e).__name__}: {e}"))
            p("    (unsupported parameter 류면 ChatOpenAI 의 reasoning_effort 를 빼야 합니다)")
            return
        p(f"    route   {r.get('route')}")
        p(f"    소요    {ms:.0f}ms   timing={r.get('timing') or {}}")
        p(f"    근거    {len(r.get('sources') or [])}건")
        p(f"    답변    {r['answer'][:300]}")
        for pl in r.get("places") or []:
            p(f"      {pl.get('time','--:--')} [{pl['phase']:<6}] {pl['name']} ({pl['distance']}m)")
        if r.get("coursePayload"):
            cp = r["coursePayload"]
            p(f"    코스저장 payload: '{cp['title']}' · {cp['duration']} · stops {len(cp['stops'])}개")

        if o["twice"]:
            p("\n[5] 두 번째 호출 (같은 프로세스 = 실서버와 같은 웜 상태)")
            t0 = time.perf_counter()
            r2 = answer(q, stadium_name=o["stadium"])
            ms2 = (time.perf_counter() - t0) * 1000
            t1, t2 = r.get("timing") or {}, r2.get("timing") or {}
            p(f"    {'단계':<12}{'1회차(콜드)':>12}{'2회차(웜)':>12}")
            for k in ("embed_ms", "retrieval_ms", "llm_ms", "structured_ms"):
                if k in t1 or k in t2:
                    p(f"    {k:<12}{t1.get(k, 0):>10}ms{t2.get(k, 0):>10}ms")
            p(f"    {'합계':<12}{ms:>10.0f}ms{ms2:>10.0f}ms")
            p("    → 2회차가 실서버 체감 속도입니다. embed 가 여기서도 1초 넘으면 진짜 병목이에요.")
        p(ok("\n  통과 — 챗봇이 RAG 로 답하고 있습니다\n") if on else "\n")
