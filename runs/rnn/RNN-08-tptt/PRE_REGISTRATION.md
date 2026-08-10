# RNN-08 / RNN-09 — PRE-REGISTRATION (fixed before results)

Packet: TPTT dependency canary + controlled retrofit smoke. Date 2026-08-10. HEAD at start `09a012e`.
Question (§1): **Does a recurrent/memory-aware TPTT retrofit help beyond ordinary LoRA under the same
training budget?** Load-bearing comparison = **TPTT+LoRA vs LoRA-only** (not TPTT vs base).

## Dependency canary — QUALIFIED (prerequisite, already run)
Isolated venv `/home/augus/tptt-venv` (does not touch sglang/evalplus). torch 2.6.0+cu124 · transformers
4.49.0 · peft 0.14.0 · tptt 0.12.1 @ `242e2140…` · RTX 3090 / driver 591.86 / CUDA 13.1. Canary:
inject+forward+backward+optim-step OK for C; LoRA-only OK for B; adapter reload BIT_EXACT; VRAM returns
to baseline. Evidence `runs/rnn/RNN-08-tptt/canary/rnn08_canary.json`. **float32** (LiZA linear path is
fp32; bf16 triggers a conv-bias dtype error).

## Model (§5/§6)
- **Qwen/Qwen2.5-0.5B** (base, no alignment), **conventional Transformer** (NOT a Qwen3.5/3.6 GDN hybrid,
  per §6 — hybrids belong to the separate GDN line). float32, single RTX 3090.
- Rationale: smallest ungated (no HF token available → Gemma3-270m inaccessible) faithful member of the
  Qwen2.5 family that TPTT documents (`Qwen2.5-1.5B` is the exact upstream example; 0.5B is the same
  architecture, chosen to keep the smoke ≤2 GPU-hr). Recorded per §5.

## Arms (§1) — identical training budget (§11)
| | A BASE | B LoRA-only (CONTROL) | C TPTT+LoRA |
|---|---|---|---|
| model | Qwen2.5-0.5B | + peft LoRA | + `get_tptt_model(delta_rule)` then same peft LoRA |
| trainable params | 0 (no train) | 1,081,344 | 1,081,344 (LoRA only — **parameter-matched**; `set_trainable_parameters` deliberately NOT used, as its broad patterns would full-FT attention and break the control) |
| memory mechanism | none | none | LiZA delta-rule linear attention mixed with softmax: `out=(1-mag)·softmax+mag·linear` |
| mag schedule | — | — | LiZACallback `gradual` 0.0→0.5 over `transition_step=96` (authors' default shape; avoids broken init) |

LoRA (both B,C): r=8, alpha=16, dropout=0, target `q_proj,k_proj,v_proj,o_proj`, bias none.
TPTT (C): operator_mode `delta_rule`, linear_precision float32, max_chunk_size 64, mag_weight final 0.5.

## Training budget (§9/§11) — MATCHED across B and C
- dataset **databricks/databricks-dolly-15k** (CC-BY-SA-3.0); fixed subset: **512 train / 64 eval**
  (disjoint), selected with seed 42; dataset fingerprint pinned in results.
- format `### Instruction:\n{instruction}\n{context}\n### Response:\n{response}<eos>`; **prompt tokens
  masked** (labels=-100) — identical masking both arms. max_len 512.
- optimizer AdamW, lr 2e-4, weight_decay 0.0, cosine schedule, warmup 10 steps.
- per_device_batch 4 × grad_accum 2 = effective 8; **epochs 3 → ~192 optimizer steps**; seed 42.
- identical data order/seed/steps/tokens across B and C (§11). Any unavoidable difference recorded.

## Evaluation (§12) — three axes, all three arms (§13)
- **A upstream-relevant:** held-out dolly SFT loss (64 ex) — sensitive instruction-fit metric.
- **B retention / effective-context (RULER-style, qualified harness):** single-needle retrieval accuracy
  at ctx **256** (≤ train len) and **1024** (> train len), within the model's practical range.
- **C general-quality regression smoke:** wikitext-2-raw-v1 test perplexity (50 seqs, len 512) —
  detects memory-gain-at-cost-of-collapse.
- Report BASE vs LoRA, BASE vs TPTT, **LoRA vs TPTT** for every metric (§13).

## Mechanism / state-isolation proof (§16) — required before attributing any effect
- record #modules replaced (self_attn→LiZAttention), LiZA linear-cache/state shape, mag active (>0).
- **state-isolation smoke:** eval sample B's logits must be identical whether or not sample A was run
  first (no cross-sample state carryover). Cross-contamination ⇒ INVALIDATES quality results.

## Metrics captured (§14)
QUALITY (A/B/C above) · TRAINING (loss, tokens, steps, wall, tok/s, peak VRAM, trainable params) ·
INFERENCE (prefill/decode latency, peak VRAM, extra recurrent-state bytes for C).

## Outcome vocabulary (§15) — NO promotion of tiny changes
CLEAR_POSITIVE_SIGNAL / NO_DETECTABLE_GAIN / QUALITY_REGRESSION / COMPUTE_ONLY_GAIN / INCONCLUSIVE.
No p-values from tiny eval sets. Decision tree (§21): TPTT_DEPENDENCY, UPSTREAM_MECHANISM,
CONTROL_VALIDITY, TPTT_VS_LORA, RNN_RESEARCH_DIRECTION.

## Budget ceilings (§9) — do not increase after seeing results
- Strong preference **≤2 GPU-hr total**; **hard ceiling 4 GPU-hr** (owner authorization required to cross).
- Peak VRAM target <20 GB. Expected actual: minutes-scale (0.5B, ~192 steps, ~2–3 GB).

## Stop conditions (§20)
Dependency instability · OOM needing unrelated compromises · numeric fwd/bwd failure · **TPTT state leaks
between samples** · control not comparable · upstream recipe not reproducible · runtime > 4 GPU-hr · eval
identity ambiguous. A failed canary would have been a valid result (it passed).

## Not doing (§22)
No Qwen3.6-27B training · no llama.cpp changes · no Memory Caching/Titans/HAM here · no multi-day runs ·
no endpoint change · no tare.tools · no push.
