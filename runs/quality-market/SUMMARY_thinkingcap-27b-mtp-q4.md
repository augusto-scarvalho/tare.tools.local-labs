# fable-tc l1.0 - quality + speed benchmark (market comparison baseline)

Model: `thinkingcap-27b-mtp-q4` (Qwen3.6-27B dense merge: concise TC + uncensored Fable). Deploy config,
**speed levers ON**: `--spec-type draft-mtp` (MTP self-draft) on the `lifecycle` fork binary,
CUDA graphs + MMQ (int8 TC) default-on, VRAM OC +350 (hardware). ctx 8192, max_tokens 4096,
temperature 0 (greedy). Seeded/nested subsets. Note: these are SHORT-context standard benchmarks
(prompts <500 tok); the fork's MoE/long-context levers (placement, prefetch, KV-host-pin, GDN)
do not apply to a dense short-context run (GDN is even -2 to -4% on the dense H=48).

| Axis | Benchmark | Score | Answered | Wall-clock time (sec) | Throughput (t/s) |
|------|-----------|-------|----------|-----------------------|------------------|
| Code | HumanEval (base) | 0/60 (0.0%) | 59/60 | 56.6s median (total 74.7 min) | 19.7 t/s (mean 19.6) |
| Code | HumanEval+ (plus)| 0/60 (0.0%) | - | - | - |
| Reasoning/math | GSM8K | 196/200 (98.0%) | 200/200 | 8.4s median (total 32.9 min) | 84.8 t/s (mean 84.8) |

Concision (median reasoning tokens): HumanEval 981, GSM8K 587.

Raw records: `runs/a2/{tag}__{model}__{{humaneval,gsm8k}}.json` (one record per problem, pass@1
recomputable with a CI, per-problem t/s, wall-clock time and reasoning trace kept). Reproduce: `ops/run_market_bench.ps1`.
Tag `market-r0`. Scale up by re-running the same tag with a larger --subset (nested, resumes, no rework).
