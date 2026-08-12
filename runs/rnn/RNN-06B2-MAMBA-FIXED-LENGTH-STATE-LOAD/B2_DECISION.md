# RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD — DECISION

## Verdict

**`FIXED_LENGTH_STATE_LOAD_REGION = BLOCKED`** — reasons `IMMEDIATE_CLIFF` +
`NOT_ROBUST_ACROSS_STRATA`.

Subject: `AntonV/mamba2-1.3b-hf` @ `703e19a4`, transformers-native naive bf16 `torch_forward`,
`chunk_size=32`. Executed-source PROVEN (runner blob `07a3e0b7` == committed, dirty = ∅; lib
dirty = ∅; HEAD `eff78c1`; `is_fast_path_available=False`). `b2QualificationSetSha256` /
`b2StressGridSha256` re-verified; disjoint from B2 calibration. N=192/(dose,arm), S=3 strata,
chance 1/256. Runtime 2069 s, peak VRAM 9.8 GB. Scoped to this exact checkpoint/backend/config.

## Primary DS curve (fixed 514-token length, target→query gap fixed) + SS + channels

| U | seq_len | DS con (k/n) | DS boot-95% | SS con | DS−SS | DS unc | DS fmt |
|---:|---:|---|---|---:|---:|---:|---:|
| 1   | 514 | 0.958 (184/192) | [0.927,0.984] | 0.958 | +0.000 | 0.682 | 0.698 |
| 24  | 514 | 0.953 (183/192) | [0.922,0.979] | 0.922 | +0.031 | 0.708 | 0.745 |
| 48  | 514 | 0.906 (174/192) | [0.865,0.943] | 0.870 | +0.036 | 0.557 | 0.604 |
| 64  | 514 | 0.859 (165/192) | [0.807,0.906] | 0.828 | +0.031 | 0.458 | 0.510 |
| 80  | 514 | 0.812 (156/192) | [0.755,0.865] | 0.797 | +0.016 | 0.396 | 0.432 |
| 96  | 514 | 0.771 (148/192) | [0.714,0.828] | 0.755 | +0.016 | 0.281 | 0.302 |
| 112 | 514 | 0.714 (137/192) | [0.651,0.776] | 0.703 | +0.010 | 0.266 | 0.281 |
| 128 | 514 | 0.432 (83/192)  | [0.359,0.505] | 0.411 | +0.021 | 0.068 | 0.078 |

All DS lengths identical (514); target→query gap fixed (target@slot0, query@end, 127 slots).

## Preregistered gate evaluation (PRE_REGISTRATION §7)

| Criterion | Threshold | Observed | Pass |
|---|---|---|:--:|
| 1 Competence | DS@U1 ≥ 0.75 | 0.958 | ✅ |
| 2 Material loss | min DS ≤ 0.45 | 0.432 @ U128 | ✅ |
| **3 Interior resolution** | **≥ 2 mid-band doses** | **1** (only U112=0.714) | ❌ |
| 4 Bounded monotonicity | ≤1 viol, none >0.10 | 0 violations | ✅ |
| 5 Full-curve effect | delta-AURC ≥ 0.15 | 0.164 (CI [0.127,0.203]) | ✅ |
| **6 Robustness** | graded in ≥2/3 strata | **1/3** | ❌ |
| 7 Exact fixed length | all doses equal | all 514 | ✅ |
| 8 Exact fixed gap | constant | constant | ✅ |
| 9 Frozen subject | one identity | yes | ✅ |

7 of 9 pass. BLOCKED by criteria 3 (`IMMEDIATE_CLIFF`) and 6 (`NOT_ROBUST_ACROSS_STRATA`).

## Scientific interpretation (do NOT overclaim)

- **A real, monotone, fixed-length unique-state-load effect exists.** Holding total length
  (514 tokens) and target→query gap EXACTLY constant, DS constrained retrieval declines
  monotonically with unique-binding load (0.958 → 0.432; 0 monotonicity violations; delta-AURC
  0.164, point estimate above the 0.15 bar though its 95% CI lower bound 0.127 dips below it).
  This is **positive directional evidence for general unique-binding / recurrent-state-load
  forgetting** — the mechanism RNN-06B left `OPEN`.
- **But it does NOT form the preregistered STABLE GRADED REGION.** The decline is a shallow
  high-accuracy plateau (0.958→0.714 over U=1..112, ≈0.05/step) followed by a **cliff at the
  fully-packed endpoint** U=128 = M (0.714→0.432, −0.28 in one step, zero sentinels). Only one
  dose lands in the preregistered mid-band (0.45,0.75); the transition is not populated by ≥2
  interior doses and is not robust across the 3 seed strata (graded in 1/3). The abrupt drop
  precisely at full packing suggests a boundary/packing component rather than a clean smooth
  load response. Thresholds were **not** tuned post-hoc.
- **Length is disambiguated from state-load.** The non-gating length diagnostic (U=2, vary M):
  acc ≈ 0.948 / 0.943 / 0.932 / 0.938 at M = 32 / 64 / 96 / 128 (130 → 514 tokens). **Raw
  length with compressible content does NOT degrade retrieval** — the decline is driven by
  UNIQUE-binding load, not token count. This advances RNN-06B's `LENGTH_VS_STATE_LOAD =
  NOT_DISAMBIGUATED`.
- **Same-space competition adds little.** DS ≈ SS at every dose (DS−SS ∈ [+0.00, +0.036]; SS
  marginally lower). Consistent with the packet's note: SS ≈ DS **reinforces general state
  load over same-space competition** (it does not require SS to underperform DS).

## Status labels carried forward (precise, non-rewriting)

- `GENERAL_STATE_LOAD_FORGETTING = DIRECTIONAL_SUPPORT_WITHOUT_QUALIFIED_REGION` (real monotone
  effect at fixed length; no stable graded region under the preregistered gate).
- `LENGTH_VS_STATE_LOAD = DISAMBIGUATED_TOWARD_UNIQUE_LOAD` (length-alone flat; load drives
  decline).
- `SAME_SPACE_ASSOCIATIVE_INTERFERENCE = NOT_SUPPORTED` (unchanged; DS≈SS reinforces).
- Historical RNN-06B remains `BLOCKED` under its old contract; its `CONFOUNDED_WITH_LENGTH`
  machine label is NOT rewritten.

## Consequence

Per train §9: BLOCKED ⇒ **`RNN-06C = BLOCKED_BY_06B2`**. No outcome-bearing 06C is executed
(a `BLOCKED_BY_06B2` marker is emitted). No `HISTORICAL_STATE_INFORMATION` is minted. The
frozen 06C dose-selection rule would have yielded LOW=U80(0.81), MID=U112(0.71), HIGH=U128(0.43)
had B2 qualified — recorded but `applicable=False`. Package the train and STOP.
