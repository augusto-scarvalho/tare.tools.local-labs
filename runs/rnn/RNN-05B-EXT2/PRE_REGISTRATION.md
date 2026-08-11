# RNN-05B-EXT2 — PRE-REGISTRATION (written BEFORE any outcome-bearing run)

**FINAL planned synthetic H3 test.** Question (§1): for ONE already-trained, stable, FROZEN GDN/DN
representation, does progressively increasing *inference-time* retention pressure produce **graded** loss of old
associations, and can HISTORICAL recurrent-state snapshots recover associations the FINAL recurrent state no
longer retrieves? This ISOLATES inference-time forgetting from TRAINING INSTABILITY (the RNN-05B-EXT confound).

This file, `machine_config.json`, and the executed constants derive from ONE frozen `Ext2Config`.
`challengeGridSha256 = 66ff24765d17c4fa95dcfcbaf4a7b374c66aa1ba52e507cd047b486ad512a9e5` (self-check
`PASS`) must appear IDENTICALLY here, in the machine config, in
`BASE_QUALIFICATION.json`, in run metadata, and in the final results. Mismatch = STOP.

## Backbone reuse decision (§2)
The RNN-05B DN/GDN backbones were **NOT saved to disk** and were trained on a **different (capacity) MQAR**
distribution, so exact-artifact reuse is INVALID and IMPOSSIBLE. Per the §2 fallback: **train each preregistered
seed ONCE** under one stable recipe, **save + SHA-256 + freeze**, and use the **identical** weights for EVERY
stress point. `BACKBONE_REUSE = RETRAIN_ONCE_THEN_FREEZE`.

## Architecture (RNN-05B-qualified family, UNCHANGED)
`MQARDeltaModel` d_model=128, d_k=64, d_v=64, conv_k=4; MC/chunk segment
seg=64. No recurrence-equation edits, no deeper readers, no GDN-mechanism edits, no new kernels. Reader =
the existing `w_u` grm connector only. Eager sequential scan remains the correctness reference (§20).

## Task — memory-bound MQAR at FIXED seq_len (temporal pressure, NOT capacity overload)
- seq_len = **512** (FIXED -> snapshot/segment positions constant across doses; §8 identity holds).
- num_pairs = **12** (FAR below the RNN-05B capacity cliff ~40 @ d_k=64), num_queries = 8,
  num_keys = 128, num_vals = 64. Writes in the EARLY 25% of the body; queries at the end.

## Stress axis (§5) — NESTED MONOTONIC, inference-only
`postwrite_gap_distractor_density_nested`: distractor keys fill the POST-WRITE retention gap in a FIXED ascending order; the dose
ladder is **nested** (a higher dose = the SAME base example with MORE gap slots converted to distractor keys, a
superset). Writes, queries, target values and ALL positions are IDENTICAL across doses. Pair count is NOT
increased. Dose ladder = [0.0, 0.08, 0.16, 0.24, 0.32, 0.4, 0.48, 0.56, 0.64].

## ONE stable recipe (§2)
Each seed trained ONCE, single-state, on a **MIXTURE over the dose ladder** (domain-randomized dose per step) so
one representation is competent across the range; steps=2500, lr=0.003, batch=96,
pool_train=4096. Then FREEZE + SHA-256. The SAME frozen weights face every stress point. Seeds
(ALL count; **no seed screening** — RNN-05B-EXT audit §7): GDN [42, 43, 44] (load-bearing, Qwen target),
DN [42, 43, 44] (load-bearing), LA [42] (mechanistic control). Disjoint TRAIN /
DEV / FINAL-HOLDOUT example ranges; pinned id hashes.

## Control-flow invariant (§6) — BASE qualification BEFORE any MC
Frozen backbones -> ALL preregistered doses for ALL seeds -> persist `BASE_QUALIFICATION.json`
(challengeGridSha256, backboneSha256[], sourceGitHead, configSha256, exampleSetSha256, stressAxis, per-seed
retention curves, qualified common region, verdict) -> verify hashes + grid digest -> graded-region gate ->
**ONLY THEN** MC/reader. The MC entrypoint LOADS+VERIFIES the artifact; absent/mismatched/unqualified => STOP.

## Graded-region gate (§7)
> Train each seed ONCE (single-state, mixture over dose ladder [0.0, 0.08, 0.16, 0.24, 0.32, 0.4, 0.48, 0.56, 0.64]); FREEZE + SHA-256; run ALL doses for ALL seeds on the FROZEN weights as BASE qualification; QUALIFY the FIXED_BACKBONE_GRADED_REGION iff every GDN seed has max BASE>=0.75, min BASE<=0.45, and >=2 doses with BASE in (0.4,0.8), AND the mid-band doses OVERLAP across all GDN seeds; else H3_TESTABILITY=BLOCKED_FIXED_BACKBONE and STOP with NO MC and NO EXT3.

`FIXED_BACKBONE_GRADED_REGION = QUALIFIED | BLOCKED`. No graded region => `H3_TESTABILITY =
BLOCKED_FIXED_BACKBONE`, STOP, no MC, **no EXT3**. (Qualifying on one cell in an arbitrary band is explicitly
insufficient; a COMMON overlapping graded region across the frozen GDN seeds is required.)

## Primary paired experiment (§10)
For every qualified frozen backbone: **A** BASE final recurrent state only · **B** parameter-free historical
snapshot aggregation (moving average) · **C** the same small trained `w_u` reader (backbone frozen; tensor
hashes before/after must prove BACKBONE_WEIGHT_MUTATION = 0). Identical examples. Reader saved durably + SHA-256.
LA remains the mechanistic control.

## Retention-curve metrics (§11) — preregistered LINEAR interpolation
Per seed/method: accuracy vs dose; AURC_RETENTION (normalized trapezoid), D50, D80/D20 transition width,
DELTA_AURC = AURC_MC − AURC_BASE, DELTA_D50. The interpolation/curve procedure is LINEAR on the ladder and is
NOT changed after seeing MC. Raw per-dose scores remain authoritative.

## Recovery / harm (§12) — expose denominators
Per seed & dose: n_base_wrong, n_recovered, RECOVERY_RATE; n_base_correct, n_harmed, HARM_RATE;
NET_RECOVERY_COUNT = n_recovered − n_harmed; NET_RECOVERY_RATE. Denominators always exposed; per-seed rates are
NOT averaged and reported as pooled query rates.

## Target-aware ablation (§13)
Per target, the proximal snapshot = the first snapshot at/after its WRITE segment. Compare FULL /
DROP_TARGET_PROXIMAL / DROP_IRRELEVANT (late) / DROP_RANDOM (deterministic, EXCLUDING the proximal set and the
irrelevant index — asserted in code) / SHAM. Report aggregate effects AND effects restricted to
BASE_WRONG->MC_CORRECT. Gate argmax is DESCRIPTIVE only; causal support requires this ablation (§14).

## Statistics (§15) & SESOI (§16)
Hierarchy: training seed / frozen backbone -> sequence -> target. Cluster-aware (backbone-level) bootstrap;
with 3 backbones this establishes **direction / stability / heterogeneity**, NOT population
inference. PRIMARY SESOI on DELTA_AURC = **0.05**. *Justification*: MC's cost is storing
(n_ckpt−1) historical matrix snapshots (each 16384 B) + per-segment read/gate latency — at
seq_len 512, seg 64 that is a ~7x live-state-memory multiplier;
DELTA_AURC 0.05 (5 pts of average retention area) is the smallest average lift that plausibly
justifies that storage+latency for a post-hoc memory mechanism. The old **3% margin is retained ONLY as an
`OPERATOR_HEURISTIC`** for direction labels, NOT as scientific authority. Decision: CI clearly above +SESOI ->
meaningful positive; CI spanning +SESOI and trivial -> INCONCLUSIVE/DIRECTIONAL; CI fully inside ±SESOI ->
PRACTICALLY_EQUIVALENT; negative -> REGRESSION. p>0.05 is NOT equivalence.

## Efficiency / performance (§18-19)
Report live matrix/conv bytes, historical-snapshot bytes, reader bytes, peak VRAM; recurrent-update / snapshot-
read / reader / total latency with prewarm -> warm steady-state (compile/cold separated). Derive
RECOVERY_PER_MiB, DELTA_AURC_PER_KiB, DELTA_AURC_PER_ADDED_ms. Efficiency is NOT inferred from bytes alone.

## Decision policy (§21) — FINAL synthetic H3
- **Case A** no graded fixed-backbone region -> H3_TESTABILITY=BLOCKED_FIXED_BACKBONE, QWEN_GDN_TRANSPLANT_GATE=
  DEFER, SYNTHETIC_DENSE_MC=PARK, STOP, NO EXT3.
- **Case B** qualified region, DELTA_AURC practically equivalent to 0 or negative -> H3=
  NOT_DETECTED_IN_QUALIFIED_REGIME, gate=DEFER, DENSE_POST_HOC_MEMORY_CACHING=PARK, STOP, NO EXT3.
- **Case C** defensible positive (graded region + positive DELTA_AURC/DELTA_D50 + directionally consistent GDN +
  recovery >> harm + target-proximal ablation supports mechanism + random/irrelevant do NOT reproduce it + LA
  does NOT show it + path counters prove historical-state use) -> H3=POSITIVE_CANDIDATE, gate=PASS_CANDIDATE
  (authorizes only DESIGN of a separate Qwen qualification packet; no automatic Qwen run).

## Guardrails
No Qwen weights · no llama.cpp/serving/deploy · no TPTT · no RNN-05C · no StateX/DART/Sparse-Delta/GDN-2/FG2-GDN/
ReplaySSM · no FLA · no new kernels · not pushed. Budget target < 1 GPU-hr (hard 2). RNN-05B/EXT evidence
immutable. External research pointers recorded for a possible RNN-06 packet only.
