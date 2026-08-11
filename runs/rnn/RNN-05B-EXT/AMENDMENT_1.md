# RNN-05B-EXT — AMENDMENT 1 (pre-committed BEFORE the amended calibration is run)

## Why
The original pre-registered 9-condition grid was **BLOCKED** (commit `9862e1b`, immutable evidence in
`rnn05bext_results.json` / `rnn05bext_outcomes.json` / `run.log`). Calibration (BASE-only, GDN seed 42, 2500
steps/condition) found GDN base accuracy **bimodal** — retain (≥0.86) at low/med distractor, collapse (≤0.09)
at high distractor — a sharp retain-or-collapse cliff with nothing in the [0.40, 0.80] testability band:

| seq_len | 0.15 | 0.35 | 0.55 |
|---|---|---|---|
| 512  | 0.974 | 0.968 | 0.075 |
| 768  | 0.974 | 0.860 | 0.089 |
| 1024 | 0.919 | 0.073 | 0.031 |

The block is a **grid-granularity** artifact: the band falls in the gap **between** the coarse distractor
tiers, not a genuine all-ceiling or all-collapse. Every transition brackets the band —
L512/L768 between distractor 0.35→0.55, L1024 between 0.15→0.35.

## What this amendment changes (and what it does NOT)
- **Only** the candidate distractor granularity: an explicit finer condition list (`amend1_grid.json`),
  cheap-first, bracketing each observed transition:
  L512 {0.42, 0.46, 0.50}, L768 {0.40, 0.44, 0.48}, L1024 {0.24, 0.30} — 8 conditions.
- **Unchanged**: the headroom/selection rule (first GDN seed-42 BASE ∈ [0.40, 0.80]; qualify iff all 3 GDN
  seeds ∈ [0.20, 0.90] and mean ∈ [0.40, 0.80]); the generator; the architecture; the seeds; the step counts;
  every downstream analysis; `CALIBRATION_RULE_IDENTITY` still holds (band constants untouched).

## Discipline
This is **BASE-only** difficulty refinement. **No MC or reader result has been observed** — the original run
STOPPED at calibration before any MC/reader training — so §6's prohibition on expanding the grid *after seeing
MC results* is not engaged, and §3 explicitly permits BASE-driven difficulty selection. This amendment is
**pre-committed** (this file + `amend1_grid.json` + the `grid_override_json` code hook are committed **before**
the amended calibration runs), so the finer grid is genuinely pre-registered, not chosen post-hoc. It is a
**single bounded** refinement — **not** an open-ended search. The original BLOCKED result remains immutable.

## Stop condition
If no amended condition lands in [0.40, 0.80] (i.e. the transition is a true discontinuity with no graded
regime), report **`H3_TESTABILITY = BLOCKED_BY_SHARP_TRANSITION`** and STOP definitively — no further grid
refinement, no nearest-condition fallback. That outcome is itself informative: GDN forgetting here is
all-or-nothing, so a "stable backbone that has lost *some* old associations" (the regime H3 needs) may not
exist under interference pressure for this substrate.

Amended calibration + (if QUALIFIED) the full frozen H3 experiment run into `runs/rnn/RNN-05B-EXT/amend1/`.
