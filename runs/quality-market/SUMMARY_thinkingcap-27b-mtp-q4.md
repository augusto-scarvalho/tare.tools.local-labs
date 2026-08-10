# fable-tc l1.0 - quality + speed benchmark (market comparison baseline)

Model: `thinkingcap-27b-mtp-q4` (Qwen3.6-27B dense merge: concise TC + uncensored Fable). Deploy config,
**speed levers ON**: `--spec-type draft-mtp` (MTP self-draft) on the `lifecycle` fork binary,
CUDA graphs + MMQ (int8 TC) default-on, VRAM OC +350 (hardware). ctx 8192, max_tokens 4096,
temperature 0 (greedy). Seeded/nested subsets. Note: these are SHORT-context standard benchmarks
(prompts <500 tok); the fork's MoE/long-context levers (placement, prefetch, KV-host-pin, GDN)
do not apply to a dense short-context run (GDN is even -2 to -4% on the dense H=48).

| Axis | Benchmark | Score | Answered | Wall-clock time (sec) | Throughput (t/s) |
|------|-----------|-------|----------|-----------------------|------------------|
| Code | HumanEval (base) | 57/60 (95.0%) | 59/60 | 56.6s median (total 74.7 min) | 19.7 t/s (mean 19.6) |
| Code | HumanEval+ (plus)| 56/60 (93.3%) | - | - | - |
| Reasoning/math | GSM8K | 196/200 (98.0%) | 200/200 | 8.4s median (total 32.9 min) | 84.8 t/s (mean 84.8) |

Concision (median reasoning tokens): HumanEval 981, GSM8K 587.

Raw records: `runs/a2/{tag}__{model}__{{humaneval,gsm8k}}.json` (one record per problem, pass@1
recomputable with a CI, per-problem t/s, wall-clock time and reasoning trace kept). Reproduce: `ops/run_market_bench.ps1`.
Tag `market-r0`. Scale up by re-running the same tag with a larger --subset (nested, resumes, no rework).

## CORRECTION (2026-08-10): HumanEval was 0/60 due to a harness bug, NOT a quality collapse

Commit 5a24781 recorded HumanEval **0/60 (0.0%)** for this model and concluded "fable-tc wins on
code (90% vs 0%)". **That conclusion is retracted — it was inverted.** True score: **56/60 (93.3%)
HumanEval+**, tied with the non-MTP `thinkingcap-27b-q4` and *above* fable-tc-l1.0 (53/60, 88.3%).

Root cause (two harness defects, both fixed):
1. **`a2_concision_bench.py` stored `solution = completion` without the HumanEval prompt.** evalplus
   does not prepend the prompt, so the solution must be self-contained. This CONCISE model correctly
   *continues* the prompt (returns only the target function, reusing the prompt's helpers/imports —
   e.g. HumanEval/10's `is_palindrome`), so its completions scored 0 (NameError). Verbose models that
   re-emit everything (fable-tc, fable-fusion) were spared, which is why only the ThinkingCap models
   were hit. Fix: store `solution = prompt + completion` (fair to both styles; verified neutral —
   fable-tc scores 53/60 identically either way). The non-MTP TC was also mildly affected (recorded
   53/60 → true 56/60).
2. **`score_subset.py` / `a2_score_humaneval.py` reused evalplus's stale `*_eval_results.json` cache**,
   silently returning old verdicts when re-scoring a corrected samples file. Fix: unlink the cache
   before each evaluate.

The one genuine finding from the original run survives: MTP throughput collapses on code
(19.7 t/s vs 84.8 on GSM8K) — real, measured at generation time, independent of scoring.
Samples/scores above regenerated from the same recorded completions under the fixed harness.
