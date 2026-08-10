# tests/benchmark_harness — LAB-QA-001 Benchmark Harness Qualification

**The scorer is an instrument and can fail.** This lab shipped two independent HumanEval scoring
bugs in one pipeline, and they inverted a scientific conclusion (commit `5a24781` claimed
"fable-tc wins on code 90% vs 0%"; retracted by `81eed6d`/`e8fb3d3` — the true result was
ThinkingCap 93.3% vs fable-tc 88.3%). From now on the evaluation harness is qualified **before**
its numbers are used to promote or reject a model.

## Run
```
python tests/benchmark_harness/benchmark_harness_selftest.py     # exit 0 = all green
```
Deterministic, **no GPU, no EvalPlus, seconds.** It exercises the real functions in
`benchmark_harness_qa.py` (imported by `a2_concision_bench.py` and `score_subset.py`) plus a tiny
in-process code scorer (`_mini_score`) that execs solution+test in a subprocess — the same contract
EvalPlus uses, on a fixture instead of the dataset. Emits `benchmark_selftest_report.json`.

## Cases and the incident each guards
| Case | Motivating incident / reason |
|---|---|
| known-good → PASS | sanity: a correct program must pass |
| known-bad → FAIL | sanity: guards against a scorer that always passes |
| **missing-prompt → detected/FAIL** | **2026-08-10 `81eed6d`**: storing the bare completion scored concise models **0/60** (NameError on a prompt-provided helper). The test reproduces the bug (bare FAILS) and verifies the fix (assembled PASSES). |
| wrong (unknown) task_id | a sample whose id isn't in the dataset must be flagged, not scored |
| duplicate task_id | a second sample for one id would silently shadow the first |
| missing sample | a smaller-than-claimed denominator |
| extra sample | an id outside the declared subset |
| truncated generation | a capped/non-terminating generation must not count as a genuine wrong answer (cf. fable-fusion, LAB-CLOSE-002) |
| malformed JSONL | a corrupt line must be reported, not silently skipped |
| **stale result cache → invalidation forced** | **2026-08-10 `81eed6d`**: EvalPlus reuses `<padded>_eval_results.json` if present → re-scoring a corrected file returned **stale** verdicts. The test emulates EvalPlus's skip-if-exists and proves `bust_stale_results` fixes it. |
| wrong benchmark version | a score under a different version must not be compared as-is |
| wrong dataset hash | an edited/swapped dataset changes its content hash |

## Metamorphic tests (same-meaning input → same result)
reorder samples · rerun scorer · irrelevant metadata · fresh-vs-cached evaluation.

## The standing rule
**No new benchmark becomes a promotion gate without at least one self-test or smoke-validation
here.** When adding a benchmark (e.g. BigCodeBench, SWE-bench, RULER — Backlog V2 Waves C/D), add
its harness-qualification cases to this suite first.

## Integration sentinel (WA-CLOSE-002) — the REAL EvalPlus, not the emulation
The 16 cases above are fast/unit and use an in-process `_mini_score` + an emulated cache. Separately,
`evalplus_sentinel.py` drives the **actually-installed EvalPlus end-to-end** on a tiny deterministic
fixture (HumanEval/0: prompt+canonical → PASS; prompt+wrong → FAIL; and the real stale-result
boundary). No GPU, ~1-2 min. Run in the EvalPlus venv:
```
/home/augus/evalplus-venv/bin/python tests/benchmark_harness/evalplus_sentinel.py
```
Records the real evalplus version (0.3.1 as of 2026-08-10) + raw stdout in
`evalplus_sentinel_report.json`. Keep this marked integration/e2e; do not fold it into the unit run.

## Files
- `benchmark_harness_selftest.py` — the unit runner (16 cases, emulated, fast, no venv).
- `evalplus_sentinel.py` — the integration/e2e sentinel (real EvalPlus, WSL venv). *(WA-CLOSE-002)*
- `benchmark_fixture_manifest.json` — fixture + coverage description.
- `benchmark_selftest_report.json` / `evalplus_sentinel_report.json` — last-run machine-readable results.
