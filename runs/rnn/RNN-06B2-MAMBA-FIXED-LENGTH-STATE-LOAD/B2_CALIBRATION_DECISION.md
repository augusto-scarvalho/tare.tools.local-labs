# RNN-06B2 — Calibration decision (EXPLORATORY → frozen construction)

Exploratory calibration (`B2_CALIBRATION.json`, `b2CalibrationSetSha256 =
727c53678f8393882a743a49f3919bcd734c97d6a891590110e0ad4d7d09a2b2`, N=48/cell, DS arm) swept
fixed slot count M ∈ {48,64,96,128} × sentinel ∈ {REPEAT1, CYCLE4} × a coarse unique-load
ladder. **No examples were cherry-picked**; the whole surface informs the choice only of the
construction (M, ladder, sentinel, batching), which is then FROZEN.

## Surface (DS constrained acc; all cells at FIXED length per M)

| M (len) | U=1 | U=8 | U=32 | U=48 | U=64 | U=96 | U=128 | sentinel |
|---|---|---|---|---|---|---|---|---|
| 48 (194)  | 0.938 | 1.000 | 0.854 | 0.708 | — | — | — | REPEAT1 |
| 64 (258)  | 0.917 | 0.979 | 0.875 | 0.854 | 0.646 | — | — | REPEAT1 |
| 96 (386)  | 0.938 | 1.000 | 0.875 | 0.854 | 0.750 | 0.479 | — | REPEAT1 |
| 128 (514) | 0.938 | 0.979 | 0.875 | 0.833 | 0.792 | 0.688 | **0.438** | REPEAT1 |
| 128 (514) | 1.000 | 0.958 | 0.896 | 0.875 | 0.833 | 0.812 | 0.438 | CYCLE4 |

## Two findings that shaped the choice

1. **Length-alone (compressible) barely degrades.** At low load (U=1, sequence dominated by
   repeated sentinels) accuracy stays ≈0.92–1.00 across M=48→128 (194→514 tokens). Raw token
   count with low unique information does NOT collapse retrieval.
2. **Fixed-length unique load DOES degrade, gradedly.** At M=128 REPEAT1, holding length at 514
   tokens and target→query gap constant, accuracy declines smoothly with unique-binding load
   (0.938 → 0.438). This is the general-state-load effect RNN-06B left OPEN.

## Frozen construction for B2 qualification

- **M = 128** (fixed total length = 4·128+2 = **514 tokens** for every dose).
- **sentinel scheme = REPEAT1** (most gradual/graded decline; CYCLE4 stays flatter then drops
  only at U=M, giving thinner interior resolution).
- **target at slot 0; query at end; target→query gap = 127 intervening slots (constant)**.
- **Primary arm = DS** (disjoint-space unique load ⇒ general state load without same-scored-
  space competition). **Secondary arm = SS** (scored-space load ⇒ general load + same-space
  competition). Non-gating **length diagnostic**: U=2 across M ∈ {32,64,96,128}.
- Batching: seq-length-adaptive, `batch ≤ 1536 // 514 = 2` at M=128 (fits in <18 GiB, as in
  06B P128). Calibration peak VRAM well within budget.

The B2 qualification set is generated fresh and disjoint from P0 / 06B / B2 calibration; the
stress grid and thresholds are preregistered in `PRE_REGISTRATION.md` before outcomes.
