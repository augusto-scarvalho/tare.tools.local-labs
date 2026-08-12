# RNN-06B3-MAMBA-CONTROLLED-STATE-LOAD-PERTURBATION — DECISION

## Verdict

**`STATE_LOAD_FORGETTING_PERTURBATION = QUALIFIED`** · **`TRANSITION_SHAPE = GRADED`**.

Subject: `AntonV/mamba2-1.3b-hf` @ `703e19a4`, transformers-native naive bf16 `torch_forward`,
`chunk_size=32`. Executed-source PROVEN (runner blob `95e6b7aa` == committed; runner/lib dirty =
∅; HEAD `7f813be`; `is_fast_path_available=False`). `b3QualificationSetSha256` /
`b3StressGridSha256` re-verified; disjoint from B3 calibration. N=192/(dose,arm), S=3 strata,
chance 1/256. Runtime 2776 s, peak VRAM 8.8 GB. Scoped to this exact checkpoint/backend/config.

## Construction-activation evidence (the audit fixes fired)

`nestedBindingIdentityCheck = PASS` — 384 checks (192 examples × 2 arms), **0 failures**: as U
increases, already-active bindings keep identity AND position (temporal-order churn eliminated).
`fixedLengthChecks` all 770 tokens; `fixedGapCheck_targetTokenConsistent = True` (target@slot0,
query@end). `min sentinel reserve = 16` at U=176 — **no U=M cell** (full-packing boundary
excluded). `examplesEvaluated=3072`, `cellsEvaluated=16` (8 DS + 8 SS).

## Primary DS curve (order-stable, subpacked, fixed 770-token length, fixed gap)

| U | sentinels | DS con (k/n) | 95% CI | SS con | DS−SS | DS unc | DS fmt |
|---:|---:|---|---|---:|---:|---:|---:|
| 1   | 191 | 0.990 (190/192) | [0.963,0.997] | 0.990 | +0.000 | 0.891 | 0.896 |
| 24  | 168 | 0.917 (176/192) | [0.869,0.948] | 0.917 | +0.000 | 0.599 | 0.625 |
| 48  | 144 | 0.865 (166/192) | [0.809,0.906] | 0.844 | +0.021 | 0.495 | 0.531 |
| 72  | 120 | 0.771 (148/192) | [0.706,0.825] | 0.755 | +0.016 | 0.432 | 0.490 |
| 96  |  96 | 0.755 (145/192) | [0.690,0.811] | 0.693 | +0.063 | 0.365 | 0.422 |
| 128 |  64 | 0.651 (125/192) | [0.581,0.715] | 0.562 | +0.089 | 0.297 | 0.333 |
| 152 |  40 | 0.568 (109/192) | [0.497,0.636] | 0.542 | +0.026 | 0.224 | 0.276 |
| 176 |  16 | 0.573 (110/192) | [0.502,0.641] | 0.557 | +0.016 | 0.266 | 0.323 |

## Preregistered gate (PRE_REGISTRATION §6) — ALL pass

| Criterion | Threshold | Observed | Pass |
|---|---|---|:--:|
| 1 Competence | acc(U1) ≥ 0.75 | 0.990 | ✅ |
| 2 Material paired loss | acc(U1)−acc(U176) ≥ SESOI 0.20 | **0.417** | ✅ |
| 3 Fixed length | all equal | 770 | ✅ |
| 4 Fixed gap | constant | 191 slots | ✅ |
| 5 Nested identity PASS | 0 failures | 0/384 | ✅ |
| 6 Positive sentinel reserve | min ≥ 16, no U=M | 16 | ✅ |
| 7 Frozen substrate | one identity | yes | ✅ |
| 8 Robust across strata | ≥2/3 | 3/3 (0.453/0.406/0.391) | ✅ |
| 9 Paired CI excludes trivial | CI lower > 0.05 | CI [0.349, 0.490] | ✅ |

`block_reasons = []` ⇒ **QUALIFIED**.

## Effect size, direction & shape

- **Paired loss (U1→U176) = 0.417**, stratified-bootstrap 95% CI **[0.349, 0.490]** — far above
  SESOI 0.20 and the trivial region. **Discordant pairs: 81 low-correct/high-wrong vs 1
  high-correct/low-wrong** — the load overwhelmingly moves examples correct→wrong (directional,
  not noise).
- Descriptive curve stats: `MEAN_RELATIVE_RETENTION_DEFICIT = 0.231` (the honestly-named quantity
  RNN-B2 mislabeled "delta-AURC"; history not rewritten); `DEFICIT_AURC_NORMALIZED = 0.238` (true
  normalized trapezoidal deficit-AURC over the load axis).
- **`TRANSITION_SHAPE = GRADED`**: total_loss 0.422, 3 interior doses (U48/72/96/128 region),
  max single step 0.104 (< 0.60·total). Smooth graded decline, not a boundary cliff.
- **DS ≈ SS** (DS−SS ∈ [0.000, 0.089]; SS marginally lower) ⇒ same-space competition adds little
  beyond general unique-binding load — reinforces general state load as the driver.

## Scientific significance (resolves the audit's open items)

With the temporal-order churn eliminated (order-stable ordinal↦slot↦binding, nested-identity
PASS) AND the full-packing boundary excluded (≥16 sentinel reserve, never U=M), increasing unique
recurrent-state load STILL causes a reproducible, robust, graded retrieval-loss perturbation at
EXACTLY fixed length and gap. This upgrades the audit interpretation:
- `UNIQUE_LOAD_EFFECT`: from `DIRECTIONAL_SUPPORT_WITH_ORDER_CHURN_CONFOUND` → **CONFIRMED
  (order-stable, subpacked)**.
- `FULL_PACKING_BOUNDARY`: from `OPEN_CONFOUND` → **EXCLUDED (effect present without U=M)**.
- `GENERAL_RECURRENT_STATE_SATURATION`: from `NOT_YET_QUALIFIED` → **QUALIFIED (controlled
  forgetting perturbation)**.
Historical RNN-06B2 (`FIXED_LENGTH_STATE_LOAD_REGION = BLOCKED`) is NOT reclassified; this is a
NEW contract on new held-out data.

## Consequence & frozen 06C dose selection

Per train §11: QUALIFIED ⇒ **RNN-06C may execute**. The frozen B3→06C dose rule yields
(applicable=True): **HIGH = U152** (acc 0.568, max paired loss), **LOW = U96** (acc 0.755,
most-loaded still-competent), **MID = U72** (acc 0.771, smallest dose reaching SESOI). 06C's L
(high-load) branch uses U=152.
