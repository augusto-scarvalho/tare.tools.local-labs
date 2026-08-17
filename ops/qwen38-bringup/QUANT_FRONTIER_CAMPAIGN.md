# Qwen3.8-27B — Quant Frontier Campaign

Goal: across the quant ladder (Q4→IQ2), for 3 tasks — (1) code quality, (2) thinking-budget on a hard
reasoning task, (3) long-context — decide which quants are worth running and how to configure thinking.
Method: fail-fast, sweep-first, gate between stages. All local, nothing pushed.

Subject: `Qwen3.8-27B`, unsloth **dynamic (UD)** GGUFs + bartowski imatrix Q4_K_M. llama.cpp fast path,
draft-mtp (exact at temp 0). Serving via `code_eval.sh` / `gsm8k_eval.sh` on :8080.

## Quant ladder (all MTP-verified PASS — draft-mtp works on every one)

| tag | file | size |
|---|---|---|
| q4kxl | unsloth UD-Q4_K_XL | 16.7G |
| bartowski-q4km | bartowski Q4_K_M (imatrix) | 16.6G |
| iq4xs | unsloth IQ4_XS | 14.6G |
| q3kxl | unsloth UD-Q3_K_XL | 12.5G |
| iq3xxs | unsloth UD-IQ3_XXS | 11.1G |
| q2kxl | unsloth UD-Q2_K_XL | 9.9G |
| iq2m | unsloth UD-IQ2_M | 9.6G |

## Stage 1 — Code quality (HumanEval+ 164, instruct, evalplus) — COMPLETE

| Quant | size | HumanEval base | HumanEval+ | format |
|---|---|---|---|---|
| iq4xs | 14.6G | 0.939 | **0.896** | 164/164 |
| q2kxl | 9.9G | 0.945 | **0.896** | 164/164 |
| q4kxl | 16.7G | 0.933 | 0.890 | 164/164 |
| bartowski-q4km | 16.6G | 0.927 | 0.890 | 164/164 |
| q3kxl | 12.5G | 0.927 | 0.884 | 164/164 |
| iq2m | 9.6G | 0.927 | 0.884 | 164/164 |
| iq3xxs | 11.1G | 0.915 | 0.872 | 164/164 |

**Finding: NO quality cliff on code.** The full plus range 0.872–0.896 (2.4pp) fits inside the n=164 noise
band (±~4.8pp at p≈0.89). From 16.7G down to **9.6G (IQ2_M, ~2.4-bit)**, HumanEval+ is flat and every quant
is 164/164 fenced (zero format failures). Unsloth dynamic quants keep critical tensors high-bit, so code
ability survives aggressive quantization. **Practical:** the 9.6G IQ2_M loses nothing measurable on code
vs the 16.7G Q4_K_XL — freeing ~7GB VRAM. bartowski imatrix ≈ unsloth (tie).

**Gate:** no quant dropped on code grounds. All 7 carry to Stage 2, whose extra purpose is to see whether a
HARDER task (GSM8K math reasoning) reveals a cliff that code did not.

## Stage 2 — Does thinking help? — GATE COMPLETE: NO (null-to-negative)

Tested on the base quant (q4kxl) across three difficulty tiers. Thinking never wins:

| Task | instruct | thinking | note |
|---|---|---|---|
| Code (HumanEval+ 164) | 89.0% | ≤86.7% (budget 8192, n=60) | truncation + ceiling |
| Easy math (GSM8K 60) | 96.7% | 95.0% (high) | ceiling |
| Hard math (MATH-500 L5 50) | **92.0%** (2 trunc) | **80.0%** (high, 5 trunc) | thinking rambles/truncates |

**Verdict: for Qwen3.8-27B, thinking is a null-to-negative lever on every benchmark we can run.** The model
is near its competence ceiling on all of them, so reasoning only adds cost + runaway-truncation risk
(median 1468 thinking tok on MATH-L5, 5/50 hit the 8192 cap without boxing). Even the hardest tier
(competition MATH Level-5) is 92% at instruct. **Deploy instruct (enable_thinking:false).** Fail-fast gate:
**skip the thinking-budget-per-quant sweep** — null lever, not worth hours ×7.

Harnesses: `gsm8k_eval.sh`, `math_eval.sh` (scorer self-test 10/11, MATH-500 boxed + sympy equivalence).

### Stage 2 repurposed — QUANT-CLIFF probe on hard math (MATH-500 L5, instruct) — COMPLETE

| Quant | size | MATH-500 L5 (n=50) | HumanEval+ |
|---|---|---|---|
| q3kxl | 12.5G | **94.0%** | 0.884 |
| q4kxl | 16.7G | 92.0% | 0.890 |
| iq3xxs | 11.1G | 92.0% | 0.872 |
| bartowski-q4km | 16.6G | 90.0% | 0.890 |
| iq4xs | 14.6G | 90.0% | 0.896 |
| q2kxl | 9.9G | 90.0% | 0.896 |
| iq2m | 9.6G | 88.0% | 0.884 |

**No cliff on hard math either.** Range 88–94% over the whole ladder, all inside the n=50 noise band
(±~9pp). Even MATH-500 Level-5 (competition tier) does not separate 16.7G from 9.6G (~2.4-bit). Combined
with Stage 1 (code, also flat), the finding is robust: **Qwen3.8-27B with unsloth dynamic quants is flat
from Q4_K_XL down to IQ2_M on both code and hard math, at instruct.** The 9.6G IQ2_M is a legitimate
deploy target, freeing ~7GB VRAM vs the 16.7G Q4_K_XL with no measurable quality loss.

## Bottom line (Stages 1+2)
1. **Quantization headroom is huge.** Run the 9.6G IQ2_M; lose nothing measurable on code or hard math.
2. **Thinking is off.** instruct beats thinking on every benchmark; deploy `enable_thinking:false`.
3. Freed VRAM (~7GB) is available for context — tested next in Stage 3.

## Stage 3 — Long-context — PENDING
Needle/recall vs ctx per surviving quant; does the smaller quant hold retrieval further, and does its VRAM
headroom buy usable context.

## Ops notes
- **WSL bg reaper:** long multi-quant/multi-arm background jobs get externally killed (observed 2×, always
  at a new server boot; dmesg shows `dxgkio_query_adapter_info` GPU ioctl errors). Mitigation: **one
  quant/arm per bg job** (~8–10 min) — these complete reliably. Per-problem records make any reaped run
  scoreable.
