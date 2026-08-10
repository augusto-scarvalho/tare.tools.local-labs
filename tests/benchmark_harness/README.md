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

## Files
- `benchmark_harness_selftest.py` — the runner (16 cases).
- `benchmark_fixture_manifest.json` — fixture + coverage description.
- `benchmark_selftest_report.json` — last run's machine-readable result (regenerated each run).
