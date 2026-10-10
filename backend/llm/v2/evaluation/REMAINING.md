# Five remaining evaluation groups

60 new unique hand-authored, author-reviewed cases: **first_action12, clarification12, delegation12, schedule12, final_answer12**. `remaining_manifest.json` freezes **dev46 / heldout14** before scoring; sibling paraphrases share families and stay together. No independent signoff or blind split claim. Seed28, reviewed_boundary70 and its manifest, exact recorded first-selection originals, production/UI/env/liveDB remain unchanged. All fixtures are explicitly synthetic, fixed **2026-10-10 18:00 KST**, observations have unique evidence IDs; `example.invalid` URLs are non-real evidence markers. `demo=false` schedule observations simulate a real-data marker in a synthetic test; they are never real game records.

## Runnable commands (repository root, no install needed)

```sh
PY="$HOME/.hermes/cache/scratch/jev-eval-venv/bin/python"
export PYTHONPATH="$HOME/.hermes/cache/scratch/jev-independent-audit-95fe1d70b0a4/docker-deps"
$PY backend/llm/v2/evaluation/remaining.py validate
$PY backend/llm/v2/evaluation/test_remaining.py -v
$PY backend/llm/v2/evaluation/remaining.py supervisor-input --output /absolute/new/scratch/supervisor-input.json
$PY backend/llm/v2/evaluation/remaining.py score --actual /absolute/scratch/captured-structured-outputs.json --output /absolute/new/scratch/score.json
$PY backend/llm/v2/evaluation/test_evaluation.py -v
$PY backend/llm/v2/evaluation/run.py offline --dataset backend/llm/v2/evaluation/reviewed_boundary.json --output /absolute/new/scratch/boundary.json
$PY backend/llm/v2/evaluation/run.py offline --output /absolute/new/scratch/seed.json
$PY backend/llm/v2/tests/run_attachment_offline.py
```

`build_remaining.py` is the reproducible authored construction source; running it on this checkout deliberately refuses to overwrite the frozen datasets/manifest. Build only in an empty copy of the evaluation directory. `remaining.py` reuses existing runner's telemetry/tracing isolation, DeepEval Golden/EvaluationDataset/LLMTestCase/BaseMetric, known production tool names and supervisor adapter. Validate rejects unknown keys, duplicate IDs/inputs/evidence IDs, malformed types/enums/lengths/tools and leaking families. Reports refuse output overwrite; score requires all 60 IDs, rejects empty/incomplete runs. No implicit actual output generation: score only consumes supplied outputs. `supervisor-input` exports 12 adapter-compatible rows; it makes zero model calls.

## Structured output contract and semantic rubric

Every Golden holds independent expected constraints, frozen tool observations, context and a semantic rubric. JSON scorer output is **contract_score** only; **semantic_correctness=not_measured**. Semantic correctness requires reviewer comparison of actual natural-language messages/tool arguments with observations and rubric. Structured extraction can omit hallucinated prose; passing extraction does not prove the prose truthful. No automated semantic LLM judge, no live/provider result or production performance claim.

- **first_action:** `selected_tools:list[known tool]`, `invalid_tool_calls:bool`. Nonempty subset of independently allowed prerequisites, or exactly no call. Two real strings remain exactly `경기일정 조회 해줘` and `음 그냥 가고싶은데 주변에서 뭘해야해?`. Their desired first routes are independent policy labels, not reinterpreted historical JEV observations. Adapter decisions are frozen authored labels, not classifier evaluation.
- **clarification:** `events`, `tools`, `notices`, `ui`, `main_questions`. Events include lookup → UI → user_answer → followup_lookup → final across turns; partial order and negative tool assertions apply. UI object is question/choices/offer_writer/free_input; known context forbids UI, enumerable choices must match observed candidates, max10 and free input required, partial/failure notices explicit. This is an abstract captured event protocol, not a fabricated production trace. Reviewer checks actual turn boundaries, no unnecessary asks/repeated lookups, real option grounding and disclosed lookup failure.
- **delegation:** `events`, `tools`, `parallel`, `goals`, `main_questions`, `nested_delegation`. Direct/narrow/exact tools, dependent partial order, independent same-response batches, single goal per specialist, main-owned questions, no nested delegation. Internal specialist sequencing permitted. Goal count is structural only: reviewer checks actual goal wording/independence and no duplicate investigations.
- **schedule:** `schedule`, `notices`, `games`, `claims_no_games`. Schedule matches frozen range/status/upcoming_only/availability/stale/count/has_more/offset; games are date/time/status projections of the frozen result. Today all-state / inclusive7 / explicit date or range / strictly-future scheduled / pagination / stale saved / stale empty / error / demo markers are separate cases. Empty or invented game projection fails. Reviewer checks request/tool argument interpretation and user-facing range/warnings, not only extraction.
- **final_answer:** `claims`, `notices`; each claim binds evidence_id/date/home/away/status/figure/source_url/demo together. Unsupported date↔team reassociation fails even when every date/team token appears somewhere in evidence (regression uses two games and swapped dates), as do unsupported figures, links, status and demo-vs-real markers. This is evidence-tuple validation, not a number bag or proof of semantic claim truth. Failure/unknown/conflicting-source examples require no invented positive claim and explicit uncertainty.

Positive outputs in `test_remaining.positive` are deliberately oracle-derived **scorer unit-test inputs**, never captured model outputs; 60 bad counterexamples plus unknown-key outputs must fail. Three existing actual compiled production graph regressions run scripted models/safe handlers: grounded candidates before main UI, independent specialist parallelism and final-answer termination. They are contract evidence, not live full-loop quality, and are not assigned as the 60 cases' production success. No new framework or scripted expected-output production replay.

## Authority, results, next actual-model plan

Read production `chain.MAIN_RULES`, UI validators, dynamic capability mapping, specialist wrapper, baseball GamesInput/_game_range/get_games, existing graph regressions, prior runner/docs and scratch graph `jev-eval-graph/graphify-out/graph.json`; production code is authority. Local worktree has no `upstream` remote/ref (origin is team URL); no Git synchronization, reset, branch rewrite or push attempted. Existing dirty production work preserved.

Execution evidence and runtime versions: `/Users/yunseongho/.hermes/cache/scratch/jev-five-ctx-7e699725b214/`. New scorer regression and original suites are recorded separately. No installs, paid calls, upload, credential-file read, global environment changes, runtime policy tuning, commit/PR/deploy or live DB mutation.

Next: obtain independent label/rubric review, then explicit provider-call budget authorization. Reuse `remaining.supervisor_rows` with `run.supervisor` for <=5 explicitly selected cases per invocation; its existing adapter is executable but live execution was intentionally not attempted. For multitur/full-loop evaluation capture actual compiled-graph model messages and safe frozen tool observations with real provider output; normalize events/claim tuples, preserve raw safe text and have reviewers grade semantic correctness separately. Run heldout once after dev fixes without tuning frozen labels to outputs; disclose any observed heldout IDs. No full-loop actual-model adapter or heldout performance estimate is delivered in this dataset-only task.
