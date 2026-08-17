# Qwen3.8-27B — Thinking-Budget Quality Curve (HumanEval+) — RECOVERED CLOSURE

Recovered after a session drop mid-experiment (2026-08-17). The fail-fast **pilot completed**
(`budget_pilot.log` → `=== DONE ===`) but the dropped session never reported the verdict. All numbers
below are **freshly re-scored** with evalplus over the task_ids actually present in each `b*__samples.jsonl`
(the on-disk `*_eval_results.json` files were stale — scored before their samples finished; e.g. b1024
eval_results 22:20 predates b1024 samples 22:59). Subject/harness unchanged.

## Harness
`ops/qwen38-bringup/budget_emul.sh` — hard thinking-budget EMULATION (llama.cpp v10159 has no native
`thinking_budget`). 2-pass: render chat prompt with `reasoning_effort=xhigh` + `<think>` open, generate up
to B thinking tokens, inject `</think>`, then generate the answer. **Never truncates to empty** (force-close
→ always emits code). Model: `Qwen3.8-27B-UD-Q4_K_XL.gguf`, ctx 8192, draft-mtp, greedy. Scored by real
evalplus over the market-r0 ~60 HumanEval+ subset.

## The recovered curve

| Config | n | pass@1 (base = plus) |
|---|---:|---:|
| instruct (no thinking) | 60 | **95.0%** |
| low (trained-concise) | 60 | 93.3% |
| ThinkingCap | 60 | 93.3% |
| med | 60 | 88.3% |
| **budget 8192 (xhigh completes)** | **60** | **86.7%** — closed-within-budget 57/60 (≈ uncapped) |
| budget 2048 | 60 | 85.0% |
| budget 1024 | 54 | 83.3% (3 lines corrupted by the kill → n=54) |
| budget 512 (cut short) | 60 | 78.3% |
| xhigh naive (cap 6144, truncates) | 60 | 45.0% |

> **Pilot correction:** the n=12 pilot reported budget-8192 at 100% (12/12) — small-n luck
> (Wilson LB ≈ 0.74). The promoted **n=60 run settles at 86.7%**, below instruct/low/med.
> The n=60 run required `CTX=16384` (budget 8192 + prompt overflows an 8192 ctx → server 400).

## Findings (load-bearing)

1. **The xhigh=45% catastrophe was a pure truncation artifact.** Naive `max_tokens` cap let the model fill
   the ceiling inside `<think>` and truncate with no code (31/60 empty). Budget emulation (force-close,
   never empty) removes it entirely.
2. **Quality is monotone-increasing in budget** once truncation is off: 512→78.3, 1024→83.3, 2048→85.0,
   8192→100 (n=12). **Cutting reasoning short is the real harm** — a half-baked scratchpad (512=78.3%) is
   worse than clean instruct (95%) or trained-concise low (93.3%). This is different from
   `reasoning_effort=low`, which is *trained* to conclude concisely rather than being cut mid-thought.
3. **Reconciles the community/Qwen guidance with our earlier "thinking hurts code" result.** Official
   guidance (default ≥8192, "more budget helps") targets *hard* competition problems; our earlier
   "thinking hurts" was **truncation**, not reasoning. When the budget is large enough for reasoning to
   COMPLETE (8192: all 12/12 closed naturally before the cap), quality returns to the top.
4. **Fail-fast gate: the track is KILLED at proper n.** The n=12 pilot said "survives" (100% ≥ 95%), but
   the promoted **n=60 = 86.7% < instruct 95%**. Even xhigh reasoning *allowed to complete* (57/60 closed
   naturally) tops out ~8pp below instruct and ~7pp below trained-concise low. The community/Qwen guidance
   ("default ≥8192, more budget helps") does **not** transfer to easy code on this model: budget helps
   monotonically but never catches instruct, because HumanEval is too easy for reasoning to add value —
   it only adds cost (and here a small deficit). **Recommendation: serve `reasoning_effort` off (instruct)
   for HumanEval-class coding; do not enable a thinking budget.**

## Status: CLOSED
n=60 promotion complete (`CTX=16384 BUDGETS="8192" SUBSET_N=60 budget_emul.sh` →
`~/.cache/wslx/budget_8192_n60_ctx16k.log`, `=== DONE ===`, 60/60 samples, 0 comp failures). Serving
endpoint on :8080 restarted via `serve.sh` afterward. Harness fix (parameterized `CTX` + resilient
`comp()`) committed with this closure.
