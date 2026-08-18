# HANDOFF — RNN-05A: Fixed-Backbone Memory Caching (2026-08-10)

Can cached historical recurrent states + a small learned reader improve recall over a **frozen** backbone's
ordinary single-state inference, with the backbone weights **identical across all arms**? Substrate is the
RNN-04 qualified **Linear Attention** memory (real Gated-DeltaNet deferred to RNN-05B). Serving CLOSED, TPTT
PARKED, no llama.cpp/deploy/Qwen changes, not pushed.

**Headline: NO.** On a frozen single-state-trained backbone, Memory Caching does **not** help. Param-free
aggregation badly degrades recall; a minimal trained reader (3072 params) learns only to *avoid harm* and
yields **no net gain** (holdout −0.06 at the main operating point). The RNN-04 GRM benefit was
**co-adaptation** of a jointly-trained backbone, **not** a free lunch exploitable post-hoc on frozen weights.
This **argues against** a naive frozen-backbone transplant and keeps `QWEN_GDN_TRANSPLANT_GATE = CONDITIONAL /
DEFER` (not advanced).

## 1. Git
- Branch **master**. **Start HEAD `40ced93`** → **End HEAD `25691f6`**
  (`25691f611d1c2ee3006eb4d9d27ba14e28200608`). Three small commits: `8b61801` stage0 (carry-forward +
  docstring), `51f3c6f` experiment harness, `25691f6` results.
- RNN-04 raw evidence untouched (immutable). `ops/rnn_mc_substrate.py` change is **comments/docstring only**
  (executed behaviour identical → RNN-04 remains bit-reproducible). **Nothing pushed.**
- Out of Git (commit discipline): backbone checkpoint (external, scratchpad), `.harness/`, RNN-08/08b adapter
  dirs, venv/clone/caches. No weights committed.

## 2. Stage 0 — RNN-04 carry-forward (reconciliation)
`runs/rnn/RNN-05A-fixed-backbone/RNN04_CARRYFORWARD.md` preserves all 9 RNN-04 audit corrections (linear
attention not DeltaNet; trained-per-N; equal-byte confound; independent-compressor-only; post-training scope;
gate CONDITIONAL/DEFER). Corrected the misleading "DeltaNet / delta rule" claim in the substrate module
docstring to state the **executed** path is additive Linear Attention (Eq. 2); the beta/delta text is now
marked reference-only / NOT EXECUTED. No behavioural change.

## 3. Frozen backbone identity (Stage 1) — `backbone_identity.json`
- Smallest transparent recurrent backbone = RNN-04 Linear-Attention `MQARModel` (d_model 128, d_k=d_v 24,
  depthwise causal conv k=3). Short calibration (D∈{40,56,72}, base_acc {0.584, 0.522, 0.548}) → non-saturated
  **MEMORY_AXIS=QUALIFIED**, selected **D=56** (base nearest 0.5). Trained **once**, single-state, 5000 steps
  (25.8 s). dev_base **0.7808**, holdout_base **0.7869**.
- Frozen: all params `requires_grad=False` (proved). Checkpoint saved **external** (NOT committed):
  `…/scratchpad/rnn05a_backbone.pt`, **SHA-256 `8b5977439f4e31c41e77326b698511bba581f335d80d12359cad158fd621762e`**,
  270 849 bytes. Per-tensor SHA-256 recorded; base-logits fingerprint recorded before augmentation.

## 4. Lifecycle checkpoint/restore proofs (Stage 2) — `lifecycle_proofs.json`
Closes the RNN-04 gap (RNN-04 tested only independent). **Both** proven on the frozen backbone:
- **INDEPENDENT_COMPRESSOR_SEMANTICS = QUALIFIED**: `independent(seg) == warmstart(seg,S_prev) − S_prev` for
  arbitrary S_prev → cross-segment **state** leak 4.8e-7 (fp noise), self-recompute 0.0, warm-carry 4.23
  (modes provably distinct). Leakage tested at the recurrent-state level with fixed features (the causal conv
  has a legitimate cross-boundary receptive field — that is not state leakage).
- **CONTINUOUS_CHECKPOINT_SEMANTICS = QUALIFIED**: reload **BIT_EXACT** (0.0); warm-start restore+continue ==
  full-prefix run (state 3.8e-6, reads 1.9e-6 < 1e-4). **CHECKPOINT_RESTORE = BIT_EXACT.**

## 5. Backbone immutability gate — `immutability_gate.json`
Around reader training: hashed all tensors before/after, and re-ran the single-state base on holdout.
**FROZEN_BACKBONE_VALIDITY = PASS**, **BACKBONE_WEIGHT_MUTATION = 0** (changed_tensors `[]`, excluding the
reader `mem.w_u.weight`), **base-logits max-abs Δ = 0.0** (bit-identical). Continuous reader arm also mutation
0. The load-bearing constraint (identical backbone across arms) holds exactly.

## 6. Trainable-parameter accounting — `param_accounting.json`
frozen_backbone **63 261** · reader/router **3 072** (`mem.w_u.weight` [24,128], 4.63 %) · total **66 333**.
recurrent-state **2 304 B/req** (constant across all arms & all N — the point). historical cache
**(N−1)·2 304 B**. Reader = the single GRM gate connector; **no** backbone params trained.

## 7. Fixed-backbone arms (Stage 3) — SAME frozen checkpoint; holdout primary, dev secondary
| arm | reader | lifecycle | trainable | dev | **holdout** | Δ vs base | outcome |
|---|---|---|--:|--:|--:|--:|---|
| **A BASE** | single | — | 0 | 0.781 | **0.7869** | — | reference |
| B POST-MC indep | moving-avg | independent | 0 | 0.431 | **0.4094** | −0.3775 | TRAINING_FREE **NEGATIVE** |
| C POST-MC continuous | moving-avg | continuous | 0 | 0.626 | **0.6282** | −0.1587 | TRAINING_FREE **NEGATIVE** |
| D TRAINED reader | GRM (w_u) | independent | 3072 | 0.726 | **0.7271** | −0.0598 | READER **NEGATIVE** (headline) |
| D′ TRAINED reader | GRM (w_u) | continuous | 3072 | 0.774 | **0.7812** | −0.0057 | READER **NO_EFFECT** (≈base) |
| E query control | u=q (param-free) | independent | 0 | 0.452 | **0.4348** | −0.3521 | isolates trained-reader gain |

Reader config chosen on **dev**; every headline number **confirmed on a fresh pinned holdout** (dev ids-hash
`9517f7eb…`, holdout `c9f42cbf…`; specs in `benchmarks`). dev/holdout deltas agree in sign → the negative
**replicates**.

## 8. Fixed-backbone memory curve (Stage 4) — `pareto_fixed_backbone.csv`
SAME frozen backbone **and** SAME trained reader; only seg_size (N) changes. `curve_meta`:
**backbone_and_reader_constant = true**, changed_tensors `[]` → the method is valid; but
**monotone_nondecreasing = false** → **FIXED_BACKBONE_MEMORY_CURVE = NOT_QUALIFIED** (accuracy *falls* with
N, so MC gives no rising memory→accuracy Pareto on a frozen backbone).
| N | seg | trained_reader | moving_avg | cache_B | recur_upd_ms | ckpt_copy_ms | read_ms | gate_ms | total_ms |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 256 | 0.7869 | 0.7869 | 0 | — | 0.0 | 0.0 | 0.0 | — |
| 2 | 128 | **0.7925** | 0.7634 | 2 304 | 0.382 | 0.017 | 0.036 | 0.254 | 1.307 |
| 4 | 64 | 0.7817 | 0.6697 | 6 912 | 0.304 | 0.032 | 0.091 | 0.616 | 1.806 |
| 8 | 32 | 0.7271 | 0.4094 | 16 128 | 0.616 | 0.036 | 0.140 | 0.829 | 2.713 |
| 16 | 16 | 0.6855 | 0.1536 | 34 560 | 2.043 | 0.113 | 0.427 | 2.716 | 6.255 |

Only at **N=2** does the trained reader marginally exceed base (+0.0056, **sub-margin → NO_EFFECT**). Storage
vs compute separated by **direct measurement** (each component timed independently; total is end-to-end — no
read-time-by-subtraction). Storage rises linearly with N; latency (esp. gate/read) rises super-linearly while
accuracy falls. acc-vs-seq_len (base/reader/mavg): 256 .790/.719/.423 · 384 .777/.580/.244 · 512 .771/.443/.154.

## 9. Interpretation (why frozen-backbone MC fails)
For additive linear memory the online (final) state **already equals the sum of all cached segment states**,
so it holds full recall; a convex aggregation over online + *partial* cached states can at best recover base
and generally injects cross-binding crosstalk. A frozen single-trained backbone never learned cache-routable
representations, and a 3 072-param gate cannot manufacture them → it learns to weight the online/full state
and **avoid harm** rather than extract gain. **Continuous ≫ independent for the trained reader** (0.781 vs
0.727) because warm-start cached states are cumulative (later ≈ full), so aggregation stays near reading the
full state. **Contrast RNN-04:** GRM's +0.11 there came from **co-training** the backbone with MC active;
RNN-05A isolates and refutes the frozen-transfer version of that claim.

## 10. Outcome vocabulary — `rnn05a_outcomes.json`
```
FROZEN_BACKBONE_VALIDITY          = PASS
INDEPENDENT_COMPRESSOR_SEMANTICS  = QUALIFIED
CONTINUOUS_CHECKPOINT_SEMANTICS   = QUALIFIED
TRAINING_FREE_MC_INDEPENDENT      = NEGATIVE   (Δ −0.3775)
TRAINING_FREE_MC_CONTINUOUS       = NEGATIVE   (Δ −0.1587)
TRAINED_READER_MC                 = NEGATIVE   (Δ −0.0598 holdout, −0.0547 dev; replicated)
FIXED_BACKBONE_MEMORY_CURVE       = NOT_QUALIFIED (method valid; accuracy falls with N)
QWEN_GDN_TRANSPLANT_GATE          = CONDITIONAL / DEFER   (NOT advanced)
```
`TRAINING_FREE_MC` and `READER_TRAINED_MC` kept **separate** (never merged). The RNN-04 post-training
moving-average negative is **not** re-used — it was re-measured here on the frozen backbone.

## 11. Negative / honest evidence
- Every MC variant on the frozen backbone is ≤ base on holdout; the best (continuous trained reader) only
  **ties** base. No positive result appeared, so — per stop conditions — **no hyperparameter search** was run.
- The one supra-base point (reader, N=2, +0.0056) is within noise/heuristic margin → reported as NO_EFFECT,
  not a win.
- Substrate is Linear Attention, not Gated-DeltaNet; toy MQAR scale; single backbone seed. These bound the
  claim: *frozen additive-linear* MC fails here; a real gated-delta memory (non-collapsing) is untested.

## 12. Exact reproduction
```
# WSL Ubuntu-24.04, isolated venv (torch 2.6.0+cu124, numpy 2.5.2, py 3.12.3), RTX 3090
V=/home/augus/tptt-venv/bin/python
PYTHONPATH=/mnt/c/projects/local-model-lifecycle/ops PYTHONHASHSEED=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $V \
  /mnt/c/projects/local-model-lifecycle/ops/rnn_mc_05a.py \
  --outdir /mnt/c/projects/local-model-lifecycle/runs/rnn/RNN-05A-fixed-backbone \
  --ckpt <SCRATCH>/rnn05a_backbone.pt
# supporting (unchanged from RNN-04): rnn_mc_bench.py --selftest ; rnn_mc_substrate.py --unittest
```
Deterministic (fixed seeds, blake2b process-stable benchmark). Whole run ≈ 5–6 GPU-min (≪ 1 GPU-hr budget).
Outputs: rnn05a_results.json, backbone_identity.json, lifecycle_proofs.json, immutability_gate.json,
param_accounting.json, pareto_fixed_backbone.csv, rnn05a_outcomes.json, run.log.

## 13. Guardrails (unchanged)
Serving CLOSED · TPTT PARKED · no llama.cpp/deploy/Qwen change · no real DeltaNet yet · no deep (2-layer)
memory (RNN-05C) · no kernel/tare work · no push · no weights/venv/clone in Git. **Qwen gate NOT advanced.**

## 14. Exactly one recommended next packet
**RNN-05B — Actual DeltaNet / Gated-DeltaNet reproduction** (the executed substrate here and in RNN-04 is
still Linear Attention). It is the required gate before any Qwen step and directly tests the one thing this
packet could not: whether a **non-collapsing gated-delta** memory changes the frozen-backbone verdict (linear
memory collapses so the online state already subsumes the cache — a real delta rule may not). RNN-05B should
fold in the co-adaptation control this packet motivates: on the **same** backbone family, quantify how much of
any MC gain is co-training vs frozen-reader transfer. Keep small-GPU, synthetic + one small real recall task,
no deploy-model training, gate still DEFERRED.
