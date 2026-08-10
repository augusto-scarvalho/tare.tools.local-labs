# Fleet leaderboard — market-r0 (quality + time + traits)

Consolidated cross-model comparison for the `market-r0` benchmark run. Deploy config, **speed
levers ON** (MTP self-draft on the `lifecycle` fork; CUDA graphs + MMQ; VRAM OC +350). ctx 8192,
max_tokens 4096, temperature 0 (greedy). HumanEval+ n=60 (seeded/nested), GSM8K n=200.
Short-context standard benchmarks — the fork's MoE/long-context levers do not apply here.

Last updated 2026-08-10. Scores reflect the **fixed** scoring harness (see Correction below).

## Quality

| Model | HumanEval (base) | HumanEval+ (plus) | GSM8K | Trait |
|---|---|---|---|---|
| **fable-tc-l1.0-q4** | 54/60 (90.0%) | 53/60 (88.3%) | 193/200 (96.5%) | concise + **uncensored (Fable)** — deploy model |
| thinkingcap-27b-q4 | 57/60 (95.0%) | **56/60 (93.3%)** | 195/200 (97.5%) | best raw accuracy, spec OFF |
| thinkingcap-27b-mtp-q4 | 57/60 (95.0%) | **56/60 (93.3%)** | **196/200 (98.0%)** | quality-identical to TC-orig; MTP = speed toggle |
| fable-fusion-711-q4 | 24/60 (40.0%) | 24/60 (40.0%) | 159/200 (79.5%) | non-terminating on short ctx (see note) |

## Time & throughput (end-to-end wall-clock, sum of per-problem `wall_s`)

| Model | HE time | HE t/s | GSM time | GSM t/s | **Total time** | HE answered | GSM answered |
|---|---|---|---|---|---|---|---|
| **fable-tc-l1.0-q4** | 15.4 min | 74.3 | 41.7 min | 77.8 | **57.1 min** ⚡ | 55/60 | 196/200 |
| thinkingcap-27b-q4 | 34.1 min | 43.2 | 67.6 min | 42.5 | **101.7 min** | 58/60 | 200/200 |
| thinkingcap-27b-mtp-q4 | 80.3 min | 19.5 | 36.7 min | 84.2 | **117.0 min** | 59/60 | 200/200 |
| fable-fusion-711-q4 | 85.2 min | 35.2 | 187.7 min | 37.4 | **272.9 min** 🐌 | 24/60 | 160/200 |

Concision (median reasoning tokens) — HE / GSM8K: fable-tc 619/512 · TC-orig 987/595 ·
TC-mtp 977/586 · fable-fusion **4096**/1103 (4096 = the max_tokens cap → hitting the ceiling
without answering).

Note on the metric: `wall_s` above is the true client-measured end-to-end time per problem. The
per-model `SUMMARY_*.md` "Wall-clock" column is ~7% lower because it counts decode only
(`predicted_ms`), excluding prefill + TTFT + overhead. Both are kept; this table uses the honest
end-to-end number for "time to finish the test".

## Verdict — best fleet model right now: **fable-tc-l1.0-q4**

Pareto-optimal. Within a few points of the top on quality (88.3% code, 96.5% math) at **half the
wall-clock** of the ThinkingCaps (≈77 t/s vs 43), and the only **uncensored + concise** option. It
is already the deploy model. Paying ~5pp of code for 2× speed + the Fable trait is a good trade.

By objective:
- **General deploy / speed + freedom:** `fable-tc-l1.0-q4` ✅ (current)
- **Max raw accuracy, time no object:** `thinkingcap-27b-q4` (the **non-MTP** one — see below)
- **Agentic 128k:** `fable-fusion-711` was the plan, but the data here calls for re-validation.

### MTP is a per-task toggle, not a better model
`thinkingcap-27b-mtp-q4` and `thinkingcap-27b-q4` have **identical quality** (93.3% / ~98%, greedy
→ MTP is quality-neutral). MTP only changes speed, and it **flips sign by task**: vs the non-MTP
twin it nearly doubles GSM8K throughput (84 vs 42 t/s) but halves HumanEval (19.5 vs 43.2 t/s) —
code is hard to draft, arithmetic is easy. Net, MTP makes the *total* run ~15 min slower. Deploy
rule: MTP ON for math, OFF for code — don't treat the MTP GGUF as a distinct quality tier.

### fable-fusion-711 doesn't just run slow — it fails to terminate
It left **36/60** HumanEval and **40/200** GSM8K problems unanswered, burning the 4096-token cap in
reasoning that never converges (median HE reasoning = 4096, the ceiling). Its 40% code / 79.5% math
and 273-min total (4.8× fable-tc) are largely a *non-termination* problem, not pure wrongness. These
are short-context benchmarks and its intended niche is long-context/agentic, but non-termination is
a serious agentic liability — re-validate before further investment in the [[agentic-local-model-plan]].

## Correction (2026-08-10) — HumanEval scores were a harness artefact

Commit `5a24781` recorded `thinkingcap-27b-mtp-q4` HumanEval **0/60** and concluded "fable-tc wins
on code (90% vs 0%)". **Retracted — the conclusion was inverted.** Two harness defects, both fixed:

1. **`a2_concision_bench.py` stored `solution = completion` without the HumanEval prompt.** evalplus
   does not prepend the prompt, so `solution` must be self-contained. A concise model correctly
   *continues* the prompt (returns only the target function, reusing the prompt's helpers/imports,
   e.g. HumanEval/10's `is_palindrome`) → scored 0 (NameError). Verbose models that re-emit
   everything (fable-tc, fable-fusion) were spared, so only the ThinkingCap models were zeroed. The
   non-MTP TC was mildly hit too (53/60 → true 56/60). Fix: `solution = prompt + completion`
   (verified neutral — fable-tc scores 53/60 either way).
2. **`score_subset.py` / `a2_score_humaneval.py` reused evalplus's stale `*_eval_results.json`
   cache** when re-scoring a corrected samples file → silent stale verdicts. Fix: unlink before each
   `evaluate`.

Samples/scores/SUMMARYs regenerated from the **same recorded completions** under the fixed harness —
no model was re-run. The one genuine finding from the original run survives: MTP throughput collapses
on code (19.5 t/s vs 84 on math), measured at generation, independent of scoring.

## Reproduce

`ops/run_market_bench.ps1 -Model <model> -Spec <draft-mtp|none>`. Raw per-problem records (t/s,
wall_s, reasoning trace, gold) in `runs/a2/market-r0__<model>__{humaneval,gsm8k}.json`; code
scoring via `score_subset.py` in the WSL evalplus venv. Per-model detail in `SUMMARY_<model>.md`.
