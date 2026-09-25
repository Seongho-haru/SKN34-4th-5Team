"""LangSmith 로 골든셋 평가 — rag_test step4/step5 의 채점을 LangSmith 실험으로 옮긴 것.

준비: backend/.env 에 LANGSMITH_API_KEY (LANGSMITH_TRACING=true 면 트레이스도 같이 남는다)
      골든셋은 rag_test/golden/*.jsonl — 컨테이너 안에서는 /app 밖이라 --golden 로 경로를 준다 (compose 에 볼륨 추가하거나 로컬 python 으로 실행)

  # 1) 데이터셋 올리기 (한 번만. 다시 올리면 같은 이름 데이터셋에 예제가 추가되니 --replace 로 지우고 올린다)
  python manage.py langsmith_eval --upload --golden ../rag_test/golden/golden_club.jsonl ../rag_test/golden/golden_venue.jsonl

  # 2) 평가 돌리기 (실험 이름에 prefix 를 붙여 before/after 를 비교)
  python manage.py langsmith_eval --run --prefix baseline
  python manage.py langsmith_eval --run --prefix after-numcheck --limit 20

채점 항목 (rag_test/scoring.py 와 같은 기준)
  hit_at_5        정답 doc_id 패턴이 sources 상위 5 안에 있는가          (expect=answer 이고 gold 가 있는 문항만)
  expect_ok       answer / clarify / refuse 판정이 기대와 같은가         (route·답변 문구로 판정)
  must_ok         must 표현 중 하나라도 답변에 있는가 (공백 무시)         (expect=answer 문항만)
  number_grounded 답변 속 두 자리 이상 숫자가 근거 청크·질문에 있는가     (지어낸 숫자 = 환각)
  latency 는 LangSmith 가 자동 기록 (P50/P95 는 실험 화면에서)
"""
import json
import re
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import connection

DATASET = "kbo-rag-golden"
REFUSE = re.compile(r"확인한 자료|찾을 수 없|알 수 없|확인할 수 없|확인이 어렵|확인 불가|안내드리기 어렵|답변드리기 어렵|"
                    r"포함되어 있지 않|예매처에 문의|(자료|일정|정보|기록)[^.\n]{0,25}없|저는 KBO 야구 직관만")
NUM = re.compile(r"\d[\d,.]*\d")


def norm(s):
    return re.sub(r"\s+", "", s or "").lower()


def load_golden(paths):
    rows = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ── 평가 함수들 (langsmith.evaluate 규약: (inputs, outputs, reference_outputs) → bool | dict) ──
def hit_at_5(inputs, outputs, reference_outputs):
    gold = reference_outputs.get("gold") or []
    if reference_outputs.get("expect") != "answer" or not gold:
        return None                                       # 해당 없음 → 평균에서 제외
    ids = [s.get("doc_id") or "" for s in (outputs.get("sources") or [])[:5]]
    return any(re.fullmatch(g, d) for g in gold for d in ids)


def _kind(outputs):
    route, ans = outputs.get("route", ""), outputs.get("answer", "")
    if "guard:clarify" in route or "어느 구장" in ans:
        return "clarify"
    if "dispatcher:scope" in route or "guard:refund" in route or "guard:pohang" in route or REFUSE.search(ans):
        return "refuse"
    return "answer"


def expect_ok(inputs, outputs, reference_outputs):
    return _kind(outputs) == reference_outputs.get("expect", "answer")


def must_ok(inputs, outputs, reference_outputs):
    must = reference_outputs.get("must") or []
    if reference_outputs.get("expect") != "answer" or not must:
        return None
    a = norm(outputs.get("answer"))
    return any(norm(m) in a for m in must)


def number_grounded(inputs, outputs, reference_outputs):
    """답변 숫자가 근거(검색된 청크 원문)나 질문에 있는지. 숫자가 없으면 해당 없음."""
    nums = {n.replace(",", "") for n in NUM.findall(outputs.get("answer") or "")}
    if not nums:
        return None
    ids = [s.get("doc_id") for s in (outputs.get("sources") or []) if s.get("doc_id")]
    evidence = inputs.get("question", "")
    if ids:
        with connection.cursor() as cur:
            cur.execute("SELECT content FROM llm_documentchunk WHERE metadata->>'doc_id' = ANY(%s)", [ids])
            evidence += " ".join(r[0] for r in cur.fetchall())
    evidence = evidence.replace(",", "")
    return all(n in evidence for n in nums)


class Command(BaseCommand):
    help = "골든셋을 LangSmith 데이터셋으로 올리고(--upload) 파이프라인을 평가한다(--run)"

    def add_arguments(self, parser):
        parser.add_argument("--upload", action="store_true")
        parser.add_argument("--replace", action="store_true", help="같은 이름 데이터셋이 있으면 지우고 다시 만든다")
        parser.add_argument("--golden", nargs="*", default=[], help="골든셋 jsonl 경로들")
        parser.add_argument("--run", action="store_true")
        parser.add_argument("--dataset", default=DATASET)
        parser.add_argument("--prefix", default="kbo-rag", help="실험 이름 접두어 (before/after 구분)")
        parser.add_argument("--limit", type=int, default=0, help="앞에서 N문항만")
        parser.add_argument("--concurrency", type=int, default=2)

    def handle(self, *args, **o):
        try:
            from langsmith import Client, evaluate
        except ImportError as e:
            raise CommandError("pip install langsmith") from e
        client = Client()

        if o["upload"]:
            if not o["golden"]:
                raise CommandError("--golden 경로를 주세요 (rag_test/golden/*.jsonl)")
            rows = load_golden(o["golden"])
            if o["replace"] and client.has_dataset(dataset_name=o["dataset"]):
                client.delete_dataset(dataset_name=o["dataset"])
            ds = client.read_dataset(dataset_name=o["dataset"]) if client.has_dataset(dataset_name=o["dataset"]) \
                else client.create_dataset(o["dataset"], description="KBO 직관 챗봇 골든셋 (club 55 + venue 30 + course)")
            client.create_examples(
                dataset_id=ds.id,
                inputs=[{"question": r["question"], "history": r.get("history") or [], "stadium_name": None, "intent": r.get("intent")} for r in rows],
                outputs=[{"expect": r.get("expect", "answer"), "gold": r.get("gold") or [], "must": r.get("must") or [],
                          "stadium": r.get("stadium"), "grade": r.get("grade"), "group": r.get("group"), "id": r.get("id")} for r in rows],
                metadata=[{"id": r.get("id"), "group": r.get("group"), "expect": r.get("expect", "answer")} for r in rows],
            )
            self.stdout.write(f"업로드 완료: {len(rows)}문항 → 데이터셋 '{o['dataset']}'")

        if o["run"]:
            from llm.v1.rag.pipeline import answer

            def target(inputs):
                r = answer(inputs["question"], history=inputs.get("history"), stadium_name=inputs.get("stadium_name"),
                           intent=inputs.get("intent"))
                return {"answer": r["answer"], "route": r["route"], "sources": r["sources"], "n_places": len(r.get("places") or [])}

            data = o["dataset"]
            if o["limit"]:
                data = list(client.list_examples(dataset_name=o["dataset"], limit=o["limit"]))
            result = evaluate(target, data=data, evaluators=[hit_at_5, expect_ok, must_ok, number_grounded],
                              experiment_prefix=o["prefix"], max_concurrency=o["concurrency"],
                              metadata={"llm": __import__("os").getenv("LLM_MODEL") or "gpt-5.6-luna"})
            self.stdout.write(f"실험 완료: {getattr(result, 'experiment_name', '')} — LangSmith 화면에서 hit_at_5·expect_ok·must_ok·number_grounded·latency 확인")
