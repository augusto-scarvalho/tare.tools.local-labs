# PRE-REGISTRATION AMENDMENT — RNN-08b (corrected TPTT re-run)

Amends (does NOT replace) `runs/rnn/RNN-08-tptt/PRE_REGISTRATION.md`, which stays **unchanged and
historical**. Written **before any RNN-08b B/C results exist**. Start HEAD `2d4ef6a`.

## 1. Why an amendment (§6)
The original RNN-08 pre-registration specified: **max_len 512, batch 4, grad_accum 2, max_chunk_size 64**.
During execution the delta-product scan **OOM'd (68 GB requested on a 24 GB card)** at that config. It was
repaired to a feasible config and the *executed* RNN-08 run used **max_len 256, batch 2, grad_accum 4,
max_chunk_size 32, use_linear_checkpoint=true**. That was a reasonable feasibility repair, but the
historical pre-registration no longer exactly describes what ran. This amendment records the discrepancy
and freezes the corrected protocol. The historical file is deliberately NOT edited to pretend the new
config was always planned.

## 2. Frozen protocol for RNN-08b (§7) — do NOT increase after seeing results
```
model              Qwen/Qwen2.5-0.5B (base, conventional Transformer, §21)
dtype              float32
MAXLEN             256
per_device_batch   2
grad_accum         4        (effective batch 8)
epochs             3        (=> 192 optimizer steps)
train_tokens       393,216
optimizer          AdamW, lr 2e-4, wd 0.0, cosine, warmup 10
seed               42
LoRA               r=8, alpha=16, target q/k/v/o_proj, dropout=0, bias none
TPTT               operator_mode=delta_rule, mag 0.0->0.5 (LiZACallback gradual, transition 96),
                   use_linear_checkpoint=true, max_chunk_size=32, linear_precision=float32
dataset            databricks/databricks-dolly-15k (CC-BY-SA-3.0), fingerprint 2537fb912ac88184,
                   512 train / 64 eval (seed 42, disjoint), prompt-masked labels
```
Identical data/budget/seed across arms B and C (§10/§11). Trainable params matched (LoRA only; see §8).

## 3. The one correction vs RNN-08 (§2,§3,§4)
RNN-08's quality was INVALIDATED by TPTT recurrent-state persisting across independent samples. Classified
**STATE_OWNERSHIP_MISMATCH** (§3): the lab's independent-sample train/eval semantics require explicit
per-sequence ownership; TPTT's shared `LCache` is plausibly intentional for continuous-stream use, so this
is a semantics mismatch, not asserted as an upstream bug.
Fix = one explicit mechanism `IndependentSequenceTPTT` (`ops/rnn_tptt_lifecycle.py`) that **resets the
LCache at each independent OUTER-sequence boundary** (training batch / eval example); intra-sequence chunk
recurrence untouched; reset runs between forwards so gradients are never disabled.

## 4. Pre-run gates (§5,§9,§11) — ALL PASS (run before training; evidence gates/rnn08b_gates.json)
- INDEPENDENT_SAMPLE_ISOLATION = PASS (with wrapper max|Δ|=0.0 BIT_EXACT; without wrapper leak=21.35).
- INTRA_SEQUENCE_RECURRENCE = PASS (state norm 32tok=172.79, 128tok=173.49 → accumulates across chunks;
  reset→0.0). TPTT recurrent state ≈ 5.66 MiB/request (144 tensors, delta state [1,14,64,64] fp32).
- TPTT_SAVE_RELOAD = QUALIFIED (BIT_EXACT). TRAINING_RESET = PASS (train-mode already isolated under
  checkpointing, leak 0.0; grads flow after reset).

## 5. Memory-axis calibration — PREDECLARED grid + rule (§13) [BASE-only, before B/C]
RNN-08 retention (single-key teacher-forced NLL) saturated at 1.0 → NON-DISCRIMINATIVE. RNN-08b calibrates
a harder axis using **BASE ONLY**, then fixes it before evaluating B/C.
- **Grid:** context ∈ {1024, 2048, 4096} × difficulty ∈ {single_key(1 distractor), multi_key_4(retrieve
  1 of 4 present keys; candidates = the 4 present values), multi_distractor_8(1 key; correct vs 8 random)}.
  Scoring = generation-free teacher-forced NLL; accuracy = correct candidate has lowest NLL. n=16/config.
- **Cost order (least→most expensive):** primary context ascending; secondary difficulty rank
  single(0) < multi_key_4(1) < multi_distractor_8(2).
- **Selection rule (deterministic, predeclared):** evaluate in cost order; **select the FIRST config whose
  BASE accuracy is strictly inside the window (0.20, 0.90)** (not floor/ceiling). If none qualifies →
  `MEMORY_AXIS = NOT_QUALIFIED` and continue with quality/control metrics only. **No search using TPTT
  results; no inventing a favorable benchmark.**

## 6. Evaluation with lifecycle (§12,§14) — per-example reset + order-invariance
Every independent eval example resets TPTT state first (Dolly SFT loss, WikiText ppl, memory axis,
latency). Order-invariance smoke: same per-example scores under order ABC vs CAB (deterministic).
Primary quality axes kept identical to RNN-08 for comparability: **held-out Dolly SFT loss + WikiText
ppl**; add the calibrated memory axis if qualified.

## 7. Parameter accounting (§8) — reported transparently
`trainableParamsMatched = true` (LoRA 1,081,344 == TPTT 1,081,344). `totalParamsMatched = false`;
`frozenStructuralDelta = 12,288` (frozen LiZA structure). The frozen delta does NOT change the experiment.

## 8. Outcome vocabulary (§17) + decision policy (§18); budget (§19)
Classify TPTT_DEPENDENCY / STATE_LIFECYCLE / SAVE_RELOAD / CONTROL_VALIDITY / MEMORY_AXIS / TPTT_VS_LORA /
RNN_DIRECTION. Target **<1 GPU-hr**, hard ceiling **2 GPU-hr**. No scaling, no kernel work, serving CLOSED,
no push.
