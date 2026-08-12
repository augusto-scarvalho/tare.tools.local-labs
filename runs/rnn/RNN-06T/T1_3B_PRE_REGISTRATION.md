# RNN-06T — 3B PRE-REGISTRATION (wide-target generalization)

Frozen BEFORE 3B outcomes. Executes because 3A `HISTORICAL_RECOVERY_TRANSPORT = QUALIFIED`. Same v2
anti-oracle construction, M=192, K=4 schedule [38,76,115,153], MAX_CONFIDENCE frozen — but the target
support is broadened to **[8,144]** with region strata aligned to the schedule:
S0 [8,38], S1 [39,76], S2 [77,115], S3 [116,144]. No single fixed snapshot has observed every target
(a late target is seen only by a late snapshot), so the adaptive selector faces a genuinely harder
problem than in 3A.

## Calibration → freeze BEST_FIXED_SNAPSHOT (before qualification)

Calibration set (seed 20260980, disjoint) is used ONLY to select `BEST_FIXED_SNAPSHOT` = the single
schedule slot with the highest mean accuracy across the wide band. It is frozen before the
qualification set is scored. Qualification set (seed 20260981) is fresh + disjoint from calibration
and all prior sets. Single-pass capture on the official fast path; fixed batch 16.

## Primary comparison + frozen thresholds

- **MAX_CONFIDENCE vs BEST_FIXED_SNAPSHOT** (the point of 3B: can adaptive confidence choose among
  historical states when no fixed snapshot is guaranteed to have seen the target).
- Also report all four fixed-slot accuracies and per-region breakdowns descriptively.

`SESOI_RECOVERY = 0.15`, `SESOI_ADAPTIVE = 0.05`, CI lower bound > 0.05 (recovery) / > 0 (adaptive),
robustness ≥ 3 of 4 region strata.

## Gate

- `WIDE_TARGET_RECOVERY ∈ {QUALIFIED, PARTIAL, NOT_REPLICATED}`: QUALIFIED iff MAX_CONFIDENCE − FINAL
  ≥ 0.15 with CI lower bound > 0.05 and robust ≥ 3/4; PARTIAL if the point estimate ≥ 0.15 but CI/
  robustness fail; else NOT_REPLICATED.
- `ADAPTIVE_SELECTION ∈ {QUALIFIED, DIRECTIONAL, NOT_QUALIFIED}` from MAX_CONFIDENCE − BEST_FIXED_
  SNAPSHOT: QUALIFIED iff Δ ≥ 0.05, paired CI lower bound > 0, robust ≥ 3/4; DIRECTIONAL iff Δ > 0
  below the bar; NOT_QUALIFIED iff Δ ≤ 0.

MAX_CONFIDENCE is frozen before these data; no new selector is tuned. Paired on identical examples;
stratified bootstrap over the 4 region strata.
