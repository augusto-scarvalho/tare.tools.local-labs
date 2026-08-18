# Backlog V2 — Wave A / P0 Implementation Handoff (2026-08-10)

Independent-audit record for the session that reconciled Backlog V2 against the repo and
implemented Wave 0 / P0 (LAB-QA-001/002/003). Read-only w.r.t. the lab except the code committed
below. Mobile-delivered: this file + the self-test report are attached to the chat response.

## 1. Executive summary
Backlog V2 reframes the lab from "make it faster" to "characterize usefulness/reliability across a
Pareto surface", and demands the lab qualify **itself** first. This session did exactly the
P0 foundation (Wave A) and nothing GPU-heavy:
- **LAB-QA-001** — a benchmark-harness self-test suite (`tests/benchmark_harness/`, 16/16 green, no
  GPU) that reproduces and guards the two scoring incidents that inverted a conclusion this month.
- **LAB-QA-002** — run identity capture (dataset/scorer/harness hashes) so a historical score is
  auditable without the current filesystem.
- **LAB-QA-003** — lexicographic promotion semantics (eligibility→correctness→quality→performance),
  in code, reusing the existing eligibility gates.
- Full **backlog reconciliation** (`BACKLOG_V2_STATUS.md`): every item marked with a concrete plan;
  Waves 1+ recorded as planned backlog (need GPU/long runs), deliberately NOT executed.

## 2. Repo / branch / HEAD
- Repo `C:\projects\local-model-lifecycle` · branch **master**.
- **Starting HEAD:** `23c3f62` · **Ending HEAD:** `a47a9c2`.
- Working tree: clean except untracked `.harness/` (this handoff). **All 5 commits LOCAL (not pushed).**
- Session commits:
```
a47a9c2 docs: Backlog V2 reconciliation against the repo + Wave A/P0 status
ae5a5d1 feat(promotion): lexicographic promotion semantics (LAB-QA-003)
068dc59 feat(bench-qa): capture run identity — dataset/scorer/harness (LAB-QA-002)
1fa3261 test(bench-qa): LAB-QA-001 benchmark harness self-test (16 cases, no GPU)
e0155fd feat(bench-qa): single-source harness glue module (LAB-QA-001)
```

## 3. Backlog reconciliation (full table in `BACKLOG_V2_STATUS.md`)
- **Completed this session:** LAB-QA-001 ✅, LAB-QA-002 ✅, LAB-QA-003 ✅.
- **Partial (pre-existing base):** LAB-SERVE-002 (serve_profiles), LAB-AGENT-001 (agentic_gate/
  agent_bench), LAB-PROV-001 (models registry), LAB-VLM-001 (vlm probes), gates.py (eligibility).
- **Missing, needs GPU/long-run (deferred):** LAB-SERVE-001, REL-001/002, CACHE-001, AGENT-002,
  CODE-001, CTX-001, ENERGY-001/002, OPS-001/002, PROV-002, ENGINE-001/002, JUDGE-001, OPT-001,
  CLOSE-001/002.
- **Parked (§20):** unchanged (custom CUDA, sub-4 KV, learned placement, EAGLE/DSpark, distributed,
  MCP, tare integration, agent product, image-gen, RL, scheduler, k8s).

## 4. Architecture / files changed (diffstat 23c3f62..a47a9c2)
```
 benchmark_harness_qa.py                       190 ++  (NEW: single-source harness glue)
 tests/benchmark_harness/…selftest.py          240 ++  (NEW: 16-case suite)
 tests/benchmark_harness/{README,manifest,report}   NEW
 src/model_lifecycle/analysis/promotion.py     138 ++  (NEW: lexicographic promotion)
 a2_concision_bench.py                          24 ±   (import glue + identity sidecar)
 score_subset.py / a2_score_humaneval.py        23 ±   (import glue)
 runs/quality-market/DATASET_IDENTITY.json      NEW    (identity anchor)
 BACKLOG_V2_STATUS.md                           NEW    (reconciliation)
 11 files changed, 879 insertions(+), 12 deletions(-)
```

## 5. Experiments / verification executed (all deterministic, no GPU)
| ID | What | Result | Raw evidence |
|---|---|---|---|
| LAB-QA-001-selftest | 16-case harness self-test | **16/16 PASS** | `tests/benchmark_harness/benchmark_selftest_report.json` |
| gates self-check | eligibility gates regression | PASS | `python -m model_lifecycle.analysis.gates` |
| promotion self-check | lexicographic decision | PASS | `python -m model_lifecycle.analysis.promotion` |
| refactor-neutrality | re-score real samples in WSL evalplus venv after refactor | **base=57/60 plus=56/60 (unchanged)** | command in §8 |

No model/GPU experiments were run (Backlog V2 §26.7 — do not burn GPU on listed items).

## 6. Failures / retractions / open questions
- No failures this session; all self-tests green.
- Retraction carried forward and now regression-guarded: the "fable-tc wins on code 90% vs 0%"
  inversion (`5a24781`, retracted `81eed6d`/`e8fb3d3`) — the missing-prompt + stale-cache bugs each
  now have a self-test that reproduces the bug and verifies the fix.
- Open questions unchanged/added in `BACKLOG_V2_STATUS.md` and the earlier `OPEN_QUESTIONS.md`:
  HumanEval+ saturation (→ LAB-CODE-001), fable-fusion non-termination (→ LAB-CLOSE-002, now
  measurable via `flag_truncated`), no-mmap residual (→ LAB-CLOSE-001).

## 7. External references actually consulted
**None were independently fetched this session** (no network access used). LAB-SERVE-001/CODE-001/
CTX-001/ENERGY-001 plans in `BACKLOG_V2_STATUS.md` are derived from the backlog's own citations
(SGLang bench_serving, BFCL, RULER, TokenPowerBench, SWE-bench, EvalPlus). Their APIs/behaviors are
**NOT VERIFIED** by me here — verify against the upstream repos before implementing. This is stated
plainly rather than implied.

## 8. Commands supporting material claims (with outputs)
```
$ git rev-parse HEAD                          -> a47a9c2 (start 23c3f62)
$ python tests/benchmark_harness/benchmark_harness_selftest.py
    [ok] known-good … [ok] missing-prompt … [ok] stale result cache … (16 lines)
    LAB-QA-001: 16/16 passed — ALL GREEN
$ PYTHONPATH=src python -m model_lifecycle.analysis.promotion   -> promotion self-check OK
$ PYTHONPATH=src python -m model_lifecycle.analysis.gates       -> gates self-check OK
$ wsl … /home/augus/evalplus-venv/bin/python3 score_subset.py runs/a2/…thinkingcap-27b-mtp…samples.jsonl
    RESULT …thinkingcap-27b-mtp-q4…humaneval__samples base=57/60 plus=56/60   (refactor neutral)
$ python -c "…dataset_hash…"  -> humaneval-plus n=164 be4388a8…, gsm8k n=1319 847f2ed6…
```

## 9. Source excerpts actually consulted / written (path + context)

**`benchmark_harness_qa.py` — `assemble_humaneval_solution` (the missing-prompt fix, single source):**
```python
def assemble_humaneval_solution(prompt: str, completion: str) -> str:
    # INCIDENT 2026-08-10 (81eed6d): bare completion scored concise models 0/60 (NameError on a
    # prompt-provided helper). evalplus does NOT prepend the prompt -> prepend it here.
    return prompt + "\n" + completion
```

**`benchmark_harness_qa.py` — `bust_stale_results` (the stale-cache fix):**
```python
def bust_stale_results(results_path) -> bool:
    p = pathlib.Path(results_path)
    if p.exists():
        p.unlink(); return True   # evalplus reuses <padded>_eval_results.json -> bust it
    return False
```

**`tests/benchmark_harness/benchmark_harness_selftest.py` — the two incident regressions:**
```python
@case("missing-prompt (non-self-contained) -> detected/FAIL", …)
def _():
    assert _mini_score(FX_COMPLETION_CONCISE, FX_TEST) is False        # reproduces the bug
    assert _mini_score(q.assemble_humaneval_solution(FX_PROMPT, FX_COMPLETION_CONCISE), FX_TEST) is True

@case("stale result cache -> invalidation forced", …)
def _():
    # emulate evalplus skip-if-exists; assert bust_stale_results makes the fresh verdict win
```

**`src/model_lifecycle/analysis/promotion.py` — `decide` (lexicographic, reuses gates.evaluate):**
```python
def decide(candidate, baseline, *, margins=None, gates=None) -> PromotionDecision:
    ge = evaluate_gates(candidate, gates)                 # Stage 1: eligibility (safety/ops)
    if not ge.eligible: return PromotionDecision("REJECT", "eligibility", ge.failures, …)
    # termination -> correctness -> quality -> only THEN performance (>=15% wall-clock = PROMOTE)
```

**`a2_concision_bench.py` — sample writer now uses the shared function + writes identity (LAB-QA-002):**
```python
solution = assemble_humaneval_solution(prompts[tid], r["completion"])
…
identity = run_identity(benchmark_name=args.workload, …, problems=problems, sampling={…},
                        timestamp=datetime.now(timezone.utc).isoformat(), …)
(out / f"{stem}__identity.json").write_text(json.dumps(identity, indent=2))
```

## 10. Committed diffs (staged vs unstaged)
All changes are **committed** (staged→committed); the working tree is clean (§2), so there is no
unstaged/uncommitted diff. Inspect any commit via `git show <sha>`. Diffstat in §4.

## 11. Rollback / reproduction
- **Reproduce the QA suite:** `python tests/benchmark_harness/benchmark_harness_selftest.py` (exit 0).
- **Reproduce module self-checks:** `PYTHONPATH=src python -m model_lifecycle.analysis.{gates,promotion}`.
- **Rollback the whole session:** `git reset --hard 23c3f62` (returns to pre-session HEAD; nothing
  pushed). To drop a single item, revert its commit (e.g. `git revert ae5a5d1` for promotion).
- **Refactor safety net:** the refactor is behavior-preserving — `score_subset.py` still yields
  56/60 on the real samples in the WSL evalplus venv (§8).

## 12. Next executable step (for a GPU session)
**LAB-SERVE-001** — adapt SGLang `python -m sglang.bench_serving` behind a thin wrapper against the
`llama-server` OpenAI endpoint; add its harness-QA cases to `tests/benchmark_harness/` first (the
standing LAB-QA-001 rule). It is the P0 gateway to Waves 1–2.
