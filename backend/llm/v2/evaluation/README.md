# Local JEV / supervisor evaluation (no RAG)

## Separate reviewed boundary dataset (2026-10-10)

`reviewed_boundary.json` adds **44 unique hand-authored semantic cases (24 PASS / 20 NON_PASS)** and **26 distinct deterministic contracts**, without changing seed28 or the two exact first-selection observations. Labels were author-reviewed against production guard/capability/course instructions only, **not independently approved**. `build_reviewed.py` contains explicit authored labels and refuses to overwrite the dataset/manifest. `reviewed_boundary_manifest.json` freezes IDs before execution: semantic dev33/heldout11 and contract dev20/heldout6 (total dev53/heldout17); split is authored order, not provider output. This is a balanced boundary suite, not a distribution estimate.

```bash
PY="$HOME/.hermes/cache/scratch/jev-eval-venv/bin/python"
$PY backend/llm/v2/evaluation/run.py offline --dataset backend/llm/v2/evaluation/reviewed_boundary.json --output /new/scratch/contracts.json
$PY backend/llm/v2/evaluation/run.py jev --credentials .env --dataset backend/llm/v2/evaluation/reviewed_boundary.json --ids schedule-only,stadium-address,weather-only,history-reference,course-followup --output /new/scratch/jev.json
```

Live JEV requires explicit IDs and caps each invocation at five; the caller must also enforce the **total** authorized run budget. Do not rerun the paid commands for this dispatch: five JEV requests and six generation requests already completed, zero supervisor calls. Generation yielded three UNREVIEWED candidates, saved separately in scratch with raw safe responses and token usage, with no promotion or semantic credit. Currency cost is unknown, not fabricated.

Expanded semantic cases intentionally have **no first-selection oracle**; supervisor rejects this dataset rather than pretending an empty tool list is gold. The original seed's strict selected-tool validation remains intact. Output `dimensions` separates guard exact/false-allow/false-deny/precision/recall, capability exact/micro precision/recall/overextra/missing, course exact and deterministic contract evidence. Empty precision denominators are null, not perfect scores. Fixed delegations never earn semantic credit.

Contracts reuse existing actual compiled production graph tests, production main registry/fixed six and middleware, with scripted models and safe tool handlers. Reports distinguish `exposed_per_model_call` from `invoked_safe_handlers` when the source graph test captures them; empty captures in middleware-only tests are not proof no tools execute. They cover literal PASS, role intersection, unknown fail-closed primitive allocation, union/prerequisites, hidden calls, concurrent isolation, attachment ID/empty args/completion, NEW/EDIT state resets, post-course suppression, delegation, grounded clarification and answer-loop termination. Unknown capability labels cannot add primitives; fixed six remain available after literal PASS by design. No live backend tool/data service execution is claimed.

Evidence: `$HOME/.hermes/cache/scratch/jev-expanded-ctx-95fe1d70b0a4/report.md`. Final contracts **26/26**, preserved seed contracts **13/13**, focused evaluation **15/15**, production chain **48/48** passed. Full offline default suite actually ran **81 checks: 78 passed, 3 errors** because the installed environment lacks `mcp` and `langchain_mcp_adapters`; no alternate scratch environment was found and nothing was installed. The initial expanded contract adapter run had one class-resolution error (25/26); it was corrected without changing dataset IDs/labels and the frozen failed run is retained.

Actual five-request JEV smoke: guard **5/5**, course **5/5**, capability exact **2/5**, micro precision **0.625**, recall **1.0**, three extras and zero missing capabilities. Extras were `web_research` for schedule/weather and `nearby_places` for a course edit. Labels were not loosened. This smoke includes four dev IDs and one heldout ID; it is not a full heldout score. Remaining: independent label review; supervisor-first-selection oracles; actual grounded-clarification/full-loop, delegation, schedule-freshness and final-answer datasets/evidence. Existing scripted tests only verify contracts for those dimensions, not live answer quality. No RAG, telemetry, upload, production edits, installs, DB mutation, commit/push/PR/deploy.


Standalone evaluation additions only: production prompts, middleware and normal requirements are unchanged. `seed.json` contains **28 hand-authored implementation-reviewed seeds**, not Synthesizer output: 15 semantic cases and 13 deterministic contract checks. Independent reviewer sign-off is still required before treating the labels as a release benchmark. `Golden` / `EvaluationDataset` are built locally; nothing is pushed to Confident AI. Exact guard/capability/course labels use a custom DeepEval metric with no semantic judge.

## Reproduce from repository root

```bash
EVAL_SCRATCH="$HOME/.hermes/cache/scratch"
uv venv --python 3.12 "$EVAL_SCRATCH/jev-eval-venv"
uv pip install --python "$EVAL_SCRATCH/jev-eval-venv/bin/python" -r backend/llm/v2/evaluation/requirements.txt
PY="$EVAL_SCRATCH/jev-eval-venv/bin/python"
$PY backend/llm/v2/evaluation/run.py offline --output "$EVAL_SCRATCH/jev-offline-new.json"
$PY backend/llm/v2/evaluation/test_evaluation.py -v
DEEPEVAL_TELEMETRY_OPT_OUT=1 $PY backend/llm/v2/tests/run_attachment_offline.py llm.v2.tests.test_chain
```

Python 3.12 is compatible with pinned Django 6.1.1 (`Requires-Python >=3.12`). Evaluation pins DeepEval 4.2.8 and the tested LangChain versions without modifying backend requirements. The verified environment was installed from the existing runtime freeze plus DeepEval; the exact full freeze is in scratch `jev-installed.txt`.

Live commands (authorized smoke only; select **at most five JEV cases total**, not repeated retries):

```bash
# Optional --credentials points to an existing private .env; only provider/model keys are selected.
$PY backend/llm/v2/evaluation/run.py jev --credentials /private/existing.env --ids greeting,jailbreak,compound,history-injection,new-course --output "$EVAL_SCRATCH/jev-live-new.json"
$PY backend/llm/v2/evaluation/run.py supervisor --credentials /private/existing.env --ids greeting,standings,compound,new-course,edit-course --output "$EVAL_SCRATCH/jev-supervisor-new.json"
$PY backend/llm/v2/evaluation/run.py generate --credentials /private/existing.env --contexts 1 --output "$EVAL_SCRATCH/jev-candidates-new.json"
```

`--contexts` accepts 1–3 policy contexts only. Generation uses the actual DeepEval Synthesizer, Korean styling, zero evolutions / quality retries, one candidate per context, maximum 12 provider calls and a 180-second wall-clock cap. Every candidate is **candidate/unreviewed**, has no generated expected output and must be manually labeled in a separate reviewed dataset before evaluation. Seed originals and frozen result files cannot be overwritten by the runner. Generation failure is distinct from metric failure; raw generation responses and usage are frozen locally when a response exists. JEV cases have 25-second transport / 35-second wall-clock limits and no application retry loop. OpenAI generation and supervisor use zero SDK retries. All commands opt out of DeepEval telemetry and all inherited LangSmith/LangChain tracing flags before import, and wrap imports/execution in `tracing_context(enabled=False)` even if a caller enabled contextual tracing. DeepEval 4.2.8 dotenv discovery and legacy keyfile merging are disabled before dependency import (`DEEPEVAL_DISABLE_DOTENV=1`, `DEEPEVAL_DISABLE_LEGACY_KEYFILE=1`); only an explicitly supplied `--credentials` file selects the four provider/model keys, without applying unrelated settings. A fresh isolated-subprocess regression denies implicit credential-file reads and all network operations, then exercises this selector using nonsecret scratch fixtures. They do not change global provider configuration. Dataset ingestion requires explicit `implementation-reviewed` / `hand-authored` provenance and validates enums/types; generated/unreviewed rows are rejected rather than promoted. Malformed expected labels are configuration errors; malformed actual labels and invalid tool calls fail scoring. `--ids` is rejected for offline/generation, empty executions fail, and generation requires exactly one nonempty unlabeled candidate per selected context.

## Parts and evidence boundaries

- **offline**: real middleware functions with synthetic classifier responses; not LLM quality evidence. Covers 0.49999/0.5 threshold, invalid guard/course labels, provider exception, role intersection, group union/unknown groups, denied hidden calls, inherited specialist decision/no reclassification, course NEW/EDIT transitions and post-course suppression, attachment-only ID/empty-args/done enforcement. Concurrent isolation reuses the existing compiled graph regression on one shared graph instance.
- **jev**: actual production `classify` payload / parser with a timeout-bounded JEV client, independent seed labels and exact capability set matching. History and screen context are references, not policy instructions. Strict labels may expose intentionally overlapping capability predictions and require review rather than silently relaxing the oracle.
- **supervisor**: actual main registry and schemas, `MAIN_RULES`, JEV prompt wrapper and dynamic middleware, real production model, **first tool selection only**. Patches only classifier decisions to independent seed labels, applies production main `before_agent` state updates with the actual message reducer (including NEW history/context/memory reset), and uses the main `run_jev=True` destination reinforcement so allocation/selection errors are not conflated with JEV errors. Focused tests compare first model inputs against the compiled production graph for NEW, EDIT, contextual fallback and selected-place override. No tools are executed; this is not full agent end-to-end answer quality. Requires production `MAIN_MODEL_CALL_BUDGET > 1` (configured by `ORCHESTRATOR_MODEL_CALL_BUDGET`); supervisor evaluation explicitly rejects budgets <= 1 because production's budget-one first call is toolless and this adapter does not reproduce that final-call middleware. Production budget behavior is unchanged. Multi-step prerequisite cases accept a nonempty subset of independently specified acceptable first tools.
- **generate**: policy-context candidate generation only; no documents, retrieval context, RAG, DB searches or cloud dataset upload.

Django is initialized against process-owned in-memory SQLite with routers disabled. Existing regression runner creates only its own memory tables. No application DB mutation, browser service, container, deployment, commit or push is needed.

## Recorded first-selection regressions

[first_selection_regressions_20261010.json](first_selection_regressions_20261010.json) records exactly two coordinator-supplied actual Docker graph/provider observations from 2026-10-10 KST, separately from desired regression policy. Both are **reproduced issues**, not quality passes or fixes; tools were prevented from executing after first-selection capture and no chat was persisted. These records are separate from `seed.json` and its oracle; no new live calls were made to save them.

Stdlib-only record check (from repository root):

```bash
python3 -I - <<'PY'
import json
from pathlib import Path
p = Path('backend/llm/v2/evaluation/first_selection_regressions_20261010.json')
cases = json.loads(p.read_text(encoding='utf-8'))['cases']
assert len(cases) == 2
assert len({c['id'] for c in cases}) == 2
assert [c['query'] for c in cases] == ['경기일정 조회 해줘', '음 그냥 가고싶은데 주변에서 뭘해야해?']
assert all(c['status'] == 'reproduced_issue' for c in cases)
print(f'PASS: two unique regression records with exact queries: {p.resolve()}')
PY
```

## Recorded implementation run

- 13/13 deterministic seeds passed (`jev-offline.json`).
- 42/42 existing `test_chain` regressions passed (`jev-regression.log`).
- Focused evaluation checks passed (`jev-focused.log`).
- JEV, supervisor and Synthesizer initialization were **blocked by missing provider credentials** in the existing private source. No live model responses, generated candidates or live quality scores exist. Blocked JSON artifacts are not test failures and not successful evaluation evidence.
- The graph was inspected at `$HOME/.hermes/cache/scratch/jev-eval-graph/graphify-out/graph.json`; source code remains the contract authority.

Frozen original implementation evidence: `$HOME/.hermes/cache/scratch/jev-evaluation-report.md` (the blocked attempts above are historical, not current smoke results). Authorized repair/smoke evidence: `$HOME/.hermes/cache/scratch/jev-repair-20261010-ctx-ea12b27b014a/report.md`. The subsequent bounded smoke produced five JEV responses (3 exact passes, 2 capability mismatches), five supervisor responses (4 first-selection passes, 1 unexpected prerequisite), and one unreviewed Korean candidate from two generation/styling calls. Expected seed labels were not tuned to these outputs. Independent release-benchmark approval remains required; first-selection results are not end-to-end answer/tool quality.
