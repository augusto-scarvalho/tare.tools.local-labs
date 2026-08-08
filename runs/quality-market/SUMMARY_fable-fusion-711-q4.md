# fable-tc l1.0 - quality + speed benchmark (market comparison baseline)

Model: `fable-fusion-711-q4` (Qwen3.6-27B dense merge: concise TC + uncensored Fable). Deploy config,
**speed levers ON**: `--spec-type draft-mtp` (MTP self-draft) on the `lifecycle` fork binary,
CUDA graphs + MMQ (int8 TC) default-on, VRAM OC +350 (hardware). ctx 8192, max_tokens 4096,
temperature 0 (greedy). Seeded/nested subsets. Note: these are SHORT-context standard benchmarks
(prompts <500 tok); the fork's MoE/long-context levers (placement, prefetch, KV-host-pin, GDN)
do not apply to a dense short-context run (GDN is even -2 to -4% on the dense H=48).

| Axis | Benchmark | Score | Answered | Wall-clock time (sec) | Throughput (t/s) |
|------|-----------|-------|----------|-----------------------|------------------|
| Code | HumanEval (base) | 24/60 (40.0%) | 24/60 | 110.4s median (total 81.7 min) | 36.6 t/s (mean 36.5) |
| Code | HumanEval+ (plus)| 24/60 (40.0%) | - | - | - |
| Reasoning/math | GSM8K | 159/200 (79.5%) | 160/200 | 33.5s median (total 175.9 min) | 39.9 t/s (mean 39.1) |

Concision (median reasoning tokens): HumanEval 4096, GSM8K 1104.

Raw records: `runs/a2/{tag}__{model}__{{humaneval,gsm8k}}.json` (one record per problem, pass@1
recomputable with a CI, per-problem t/s, wall-clock time and reasoning trace kept). Reproduce: `ops/run_market_bench.ps1`.
Tag `market-r0`. Scale up by re-running the same tag with a larger --subset (nested, resumes, no rework).
