# RNN-06B3 — Calibration decision (EXPLORATORY → frozen construction)

Exploratory calibration (`B3_CALIBRATION.json`, `b3CalibrationSetSha256 =
342f0961cb67b8f501ba15de8151d333d6a936f7bb3ce88433303bc1559717bc`, N=48/cell, DS arm) over the
predeclared candidate family M ∈ {128,192} × reserve ∈ {16,32} (sentinel REPEAT1). **No grid
extension was needed** (the single permitted append-only extension was not used). **No examples
cherry-picked.** `nested_identity_all_pass = True` for the calibration grid (order-stable
construction verified live).

## Surface (DS constrained acc; ORDER-STABLE; subpacked, never U=M)

| M / reserve | len | U=1 | U=48 | U=72 | U=96 | U=128 | U=max(sub) | drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 / 16 | 514 | 0.979 | 0.917 | 0.896 | 0.812 | 0.688 | 0.688 (U112) | 0.292 |
| 128 / 32 | 514 | 0.979 | 0.917 | 0.896 | 0.812 | —     | 0.812 (U96)  | 0.167 |
| 192 / 16 | 770 | 0.979 | 0.958 | 0.917 | 0.792 | 0.729 | **0.479 (U176)** | **0.500** |
| 192 / 32 | 770 | 0.979 | 0.958 | 0.917 | 0.792 | 0.729 | 0.521 (U160) | 0.458 |

## Findings

- **The unique-state-load effect survives the B2 confound removal.** With the temporal-order
  churn fixed (permanent ordinal↦slot↦binding, nested-identity PASS) and the full-packing
  boundary excluded (≥16 sentinel reserve, never U=M), increasing unique load STILL produces a
  material, monotone retrieval decline. This directly addresses the audit's
  `UNIQUE_LOAD_EFFECT = DIRECTIONAL_SUPPORT_WITH_ORDER_CHURN_CONFOUND` and
  `FULL_PACKING_BOUNDARY = OPEN_CONFOUND`.
- **M=192, reserve=16 is the strongest, cleanest construction:** competent low load (0.979),
  a plateau (U≤48), then a smooth graded decline to 0.479 at the max subpacked dose U=176 (16
  sentinels retained) — drop 0.50 with multiple interior doses. This looks GRADED (not a
  boundary cliff).

## Frozen construction for B3 qualification

- **M = 192** (fixed total length = 4·192+2 = **770 tokens** for every dose).
- **MIN_SENTINEL_RESERVE = 16** ⇒ max qualification dose `U ≤ M − 16 = 176`; every dose retains
  ≥16 sentinels (no U=M).
- **sentinel = REPEAT1**; **target at slot 0**; query at end; **target→query gap = 191 slots
  (constant)**; **order-stable ordinal↦slot↦binding** map (nested-identity enforced).
- **Primary arm DS** (disjoint-space load); **SS** secondary diagnostic (not gating).
- Batching: seq-length-adaptive; budget set so batch = 2 at 770 tokens (peak VRAM monitored).

The B3 qualification set is generated fresh and disjoint from P0 / 06B / B2-calib / B2-qual /
B3-calib; the stress grid, SESOI, gate, transition-shape categories, and the B3→06C dose rule
are preregistered in `PRE_REGISTRATION.md` before outcomes.
