# HANDOFF — RNN-08b: Corrected TPTT Controlled Re-run (2026-08-10)

Corrects RNN-08's INVALID control (TPTT recurrent-state leaked across independent samples) with one
explicit state-lifecycle mechanism, re-runs the matched BASE/LoRA/TPTT comparison, and now reaches a
**valid** verdict. RNN-08 evidence preserved immutable. Isolated venv; no push; serving CLOSED; no
llama.cpp/deploy changes.

## 1. Git
- Branch **master**. **Start HEAD `2d4ef6a`** → **End HEAD `4f5b139`**
  (`research(rnn): RNN-08b corrected TPTT re-run — valid lifecycle, QUALITY_REGRESSION`, 12 files).
- Untracked: `.harness/` + local adapter dirs under `runs/rnn/RNN-08{,b}-tptt*/` (out of Git per §22).
  Adapters referenced by sha256: RNN-08b LoRA `827c0ca…`, TPTT `51bdc40…`. `runs/rnn/RNN-08-tptt/`
  **untouched**. Not pushed.

## 2. Decision tree (§17/§18) — now VALID
```
TPTT_DEPENDENCY   = QUALIFIED
STATE_LIFECYCLE   = QUALIFIED
SAVE_RELOAD       = QUALIFIED
CONTROL_VALIDITY  = GOOD
MEMORY_AXIS       = NOT_QUALIFIED
TPTT_VS_LORA      = QUALITY_REGRESSION
RNN_DIRECTION     = PARK_THIS_TPTT_CONFIGURATION / PIVOT_MEMORY_CACHING
```

## 3. Interpretation of RNN-08 (§1/§3) — preserved
RNN-08: TPTT_DEPENDENCY QUALIFIED, mechanism REPRODUCED, CONTROL_VALIDITY INVALID, QUALITY INCONCLUSIVE.
Root cause classified **STATE_OWNERSHIP_MISMATCH** (not asserted an upstream bug): TPTT's shared `LCache`
persisted delta-rule state across independent forward calls; the lab's independent-sample semantics
require explicit per-sequence ownership. RNN-08's TPTT quality numbers (3.32 / 282 / 0.45) remain
**INVALIDATED**, not evidence against TPTT.

## 4. Pre-registration amendment (§6) — `PRE_REGISTRATION_AMENDMENT_RNN08b.md`
Original RNN-08 pre-reg (max_len 512 / batch 4 / ga 2 / chunk 64) OOM'd (68 GB) → repaired to the
executed config. RNN-08b freezes that corrected protocol (max_len 256 / batch 2 / ga 4 / chunk 32 /
`use_linear_checkpoint=true`, 192 steps, seed 42). Historical pre-reg left unedited.

## 5. State-lifecycle implementation (§4) — `ops/rnn_tptt_lifecycle.py`
`IndependentSequenceTPTT(lcache)`: `.reset()` at each **independent outer-sequence boundary** (training
micro-batch via a `ResetTrainer.training_step` hook; every eval example). Intra-sequence chunk recurrence
untouched. reset runs BETWEEN forwards → never detaches/disables the current graph's gradients. ONE
mechanism, two integration points (no ad-hoc resets).

## 6. Pre-run gates (§5/§9/§11) — ALL PASS (`gates/rnn08b_gates.json`)
| gate | evidence | status |
|---|---|---|
| INDEPENDENT_SAMPLE_ISOLATION | with wrapper max\|Δ\|=**0.0**; without=**21.35** | PASS |
| INTRA_SEQUENCE_RECURRENCE | state norm 32tok=172.79 → 128tok=173.49 (accumulates across chunks); reset→0.0 | PASS |
| TPTT_SAVE_RELOAD | fresh-model load → logits **BIT_EXACT (0.0)** | QUALIFIED |
| TRAINING_RESET | train-mode leak **0.0** (already isolated under checkpointing); LoRA grads flow after reset | PASS |
Intra-sequence recurrence proof also = state-byte evidence (§16): **~5.66 MiB/request** (144 tensors;
delta state `[1,14,64,64]` fp32), separate from full-attn KV.

## 7. Three-arm configs + budget equality + parameter accounting (§7/§8/§10)
A BASE (no train) · B LoRA-only (r8/α16, q/k/v/o) · C `get_tptt_model(delta_rule, mag 0→0.5, chunk 32,
use_linear_checkpoint)` + same LoRA. Identical data/budget (dolly fingerprint `2537fb912ac88184`, 512
train/64 eval, 192 steps, seed 42, 393,216 tokens). **`trainableParamsMatched=true` (1,081,344 == 1,081,344)**;
`totalParamsMatched=false`; **`frozenStructuralDelta=12,288`** — reported transparently, experiment unchanged.

## 8. Memory-axis calibration (§13) — MEMORY_AXIS = NOT_QUALIFIED (`memcalib/rnn08b_memcalib.json`)
BASE-only, predeclared grid (context {1024,2048,4096} × difficulty {single, multi_key_4,
multi_distractor_8}), deterministic rule "first cost-ordered config with 0.20<base_acc<0.90". **All 9
configs saturated at base_acc 1.0** → no discriminative config → NOT_QUALIFIED. No favorable benchmark
invented (§13). Selection was fixed BEFORE any B/C evaluation (BASE-only). Quality rests on SFT loss +
WikiText ppl.

## 9. Quality + order-invariance + performance (§12/§14/§15) — `rnn08b_results.json` / `rnn08b_analysis.json`
| metric | BASE | LoRA | TPTT (corrected) | TPTT vs LoRA |
|---|---:|---:|---:|---:|
| held-out SFT loss ↓ | 2.1221 | **2.0022** | 3.2692 | **+63.3%** |
| WikiText ppl ↓ | 16.255 | 16.571 | 240.80 | **+1353%** |
| retention @256 / @1024 (non-discriminative) | 1.0/1.0 | 1.0/1.0 | 1.0/0.65 | — |
| prefill ms/400tok | 31.9 | 32.0 | 213.8 | +568% |
- **Order-invariance smoke PASS** (per-example SFT loss identical under order ABC vs CAB, max\|Δ\|=0.0)
  ⇒ eval is order-independent under the lifecycle; the comparison is trustworthy.
- Training: LoRA 72 s / 5433 tok·s⁻¹ / 3.97 GB · TPTT 701 s / 561 tok·s⁻¹ / 10.5 GB → **9.6× cost**.

## 10. Verdict (§17/§18)
With a **valid** lifecycle, TPTT+LoRA materially **loses** to LoRA-only at 9.6× cost →
**QUALITY_REGRESSION** → `PARK_THIS_TPTT_CONFIGURATION / PIVOT_MEMORY_CACHING`. Crucially the corrected
numbers (3.27 / 241) ≈ RNN-08's confounded ones (3.32 / 282): **state leakage was NOT the cause of the
poor quality**; this specific retrofit (delta_rule, mag 0→0.5, LoRA-only, 0.5B, 192-step matched budget)
genuinely regresses. **Not extrapolated to all TPTT methods** (§17) — it evaluates *this configuration
under this parameter-efficient matched budget*.

## 11. Failures / negative evidence (honesty)
- Primary valid negative: this TPTT config regresses vs LoRA (above).
- Gate iteration: first training-reset gate asserted state-persists-in-train, which is false under
  gradient checkpointing (cache not retained in train forward); corrected to a loss-based train-mode
  isolation test (leak 0.0). Honest, recorded.
- MEMORY_AXIS NOT_QUALIFIED (probe saturates) — a measurement limitation, not a TPTT property.
- mag-mixing init leaves TPTT far from base at start (train loss 3.24); 192 steps insufficient to recover.

## 12. Source excerpts consulted (file/function/line)
- `tptt/modeling_tptt.py`: mixing `out=(1-mag)*softmax+mag*linear` (L885-886); frozen causal-conv
  (L143/146); `LCache` API `{inputs_states, reset, update}`. `LiZACallback.__init__` gradual 0→0.5/100.
- `get_tptt_model(..., use_linear_checkpoint, max_chunk_size)` signature (R0/R1+08 notes).

## 13. Reproduction (§18) + artifact identity
```
cd /mnt/c/projects/local-model-lifecycle
PYTHONPATH=ops PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/augus/tptt-venv/bin/python \
  ops/rnn_tptt_gates.py    --outdir runs/rnn/RNN-08b-tptt-corrected/gates
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/augus/tptt-venv/bin/python \
  ops/rnn_tptt_memcalib.py --outdir runs/rnn/RNN-08b-tptt-corrected/memcalib
PYTHONPATH=ops PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/augus/tptt-venv/bin/python \
  ops/rnn_tptt_experiment_08b.py --outdir runs/rnn/RNN-08b-tptt-corrected
/home/augus/tptt-venv/bin/python ops/rnn_tptt_analyze_08b.py \
  --results .../rnn08b_results.json --gates .../gates/rnn08b_gates.json \
  --calib .../memcalib/rnn08b_memcalib.json --out .../rnn08b_analysis.json
```
Identity: Qwen/Qwen2.5-0.5B; dolly fingerprint `2537fb912ac88184`; tptt `242e2140…`; transformers 4.49.0;
LoRA adapter sha `827c0ca…`, TPTT adapter sha `51bdc40…`; git HEAD `4f5b139`. Env `env_manifest.txt`.

## 14. Exactly one recommended next packet
**PIVOT to Memory-Caching reference reproduction (RNN-04) on a small FLA recurrent model, mechanism-only
on synthetic MQAR** (the R0/R1-scoped cheap path). Rationale: the parameter-efficient TPTT retrofit is
PARKED for this config; the lab's core long-horizon hypothesis is cached-recurrent-state (Memory
Caching), which has no official code and should be reproduced at mechanism level before any Qwen
transplant. Keep it CPU/small-GPU, no training of the deploy models. (Do NOT scale TPTT; do NOT re-open
serving.)

## 15. Guardrails (unchanged)
No model/family scaling · no kernel work · no Memory Caching/Titans/HAM *implementation* in THIS packet ·
no llama.cpp/serving change · no Qwen3.6 training · no tare.tools · no push. Serving CLOSED.
