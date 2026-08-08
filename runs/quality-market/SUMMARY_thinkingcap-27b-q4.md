# fable-tc l1.0 - quality + speed benchmark (market comparison baseline)

Model: `thinkingcap-27b-q4` (Qwen3.6-27B dense merge: concise TC + uncensored Fable). Deploy config,
**speed levers ON**: `--spec-type draft-mtp` (MTP self-draft) on the `lifecycle` fork binary,
CUDA graphs + MMQ (int8 TC) default-on, VRAM OC +350 (hardware). ctx 8192, max_tokens 4096,
temperature 0 (greedy). Seeded/nested subsets. Note: these are SHORT-context standard benchmarks
(prompts <500 tok); the fork's MoE/long-context levers (placement, prefetch, KV-host-pin, GDN)
do not apply to a dense short-context run (GDN is even -2 to -4% on the dense H=48).

| Axis | Benchmark | Score | Answered | Wall-clock time (sec) | Throughput (t/s) |
|------|-----------|-------|----------|-----------------------|------------------|
| Code | HumanEval (base) | 54/60 (90.0%) | 58/60 | 25.1s median (total 31.6 min) | 43.6 t/s (mean 43.3) |
| Code | HumanEval+ (plus)| 53/60 (88.3%) | - | - | - |
| Reasoning/math | GSM8K | 195/200 (97.5%) | 200/200 | 17.4s median (total 62.6 min) | 43.4 t/s (mean 42.8) |

Concision (median reasoning tokens): HumanEval 999, GSM8K 598.

Raw records: `runs/a2/{tag}__{model}__{{humaneval,gsm8k}}.json` (one record per problem, pass@1
recomputable with a CI, per-problem t/s, wall-clock time and reasoning trace kept). Reproduce: `ops/run_market_bench.ps1`.
Tag `market-r0`. Scale up by re-running the same tag with a larger --subset (nested, resumes, no rework).
