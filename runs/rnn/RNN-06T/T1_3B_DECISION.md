# RNN-06T — 3B DECISION (wide-target generalization)

## Verdicts

- **`WIDE_TARGET_RECOVERY = QUALIFIED`**
- **`ADAPTIVE_SELECTION = QUALIFIED`**

Official `state-spaces/mamba2-1.3b` fast path, single-pass capture, band [8,144], 4 region strata.
`BEST_FIXED_SNAPSHOT = slot 115` frozen from calibration (`c0635296…`) before scoring the fresh
disjoint qualification set (`d8012e61…`, N=192, disjoint from all prior + the calib set). MAX_CONFIDENCE
frozen. Executed-source PROVEN (runner blob `9299fe4b…`, dirty ∅). fast_path_active, 0 boundary
failures, weights immutable. Runtime 201 s.

## Arm accuracies (N=192)

| arm | acc |
|---|---:|
| FINAL | 0.286 |
| BEST_FIXED_SNAPSHOT (slot 115, from calib) | 0.438 |
| MAX_CONFIDENCE (frozen adaptive) | 0.776 |
| ORACLE_BEST_GOLD (diag) | 0.870 |
| all fixed slots | 38:0.240 · 76:0.396 · 115:0.438 · 153:0.484 |

**No single fixed snapshot is competitive across the band** (best is 0.484), because each target region
is recoverable only from a different snapshot — the per-region breakdown makes the mechanism explicit:

| region (target slot) | FINAL | slot38 | slot76 | slot115 | slot153 | MAX_CONF |
|---|---:|---:|---:|---:|---:|---:|
| S0 [8,38] | 0.19 | **0.96** | 0.73 | 0.42 | 0.33 | 0.94 |
| S1 [39,76] | 0.19 | 0.00 | **0.85** | 0.54 | 0.35 | 0.77 |
| S2 [77,115] | 0.21 | 0.00 | 0.00 | **0.79** | 0.38 | 0.65 |
| S3 [116,144] | 0.56 | 0.00 | 0.00 | 0.00 | **0.88** | 0.75 |

Pre-target snapshots score 0.00 (they never saw the target); the best snapshot marches rightward with
the target region. MAX_CONFIDENCE tracks it (selection histogram [66,39,40,47] — all four snapshots
used substantially).

## WIDE_TARGET_RECOVERY — QUALIFIED

`MAX_CONFIDENCE − FINAL = +0.490`, 95% CI [0.417, 0.557], robust **4/4** regions. Historical recovery
holds across the whole band, not just when the target sits before an early snapshot.

## ADAPTIVE_SELECTION — QUALIFIED

`MAX_CONFIDENCE − BEST_FIXED_SNAPSHOT = +0.339`, paired 95% CI **[0.271, 0.401]**, robust **3/4**
(MAX_CONFIDENCE beats the best fixed slot on 73 examples, loses on 8). The only non-positive region is
S2 [77,115] (−0.15), where the frozen BEST_FIXED = slot 115 is by construction the optimal snapshot,
so adaptive selection cannot improve on it there. Everywhere else the adaptive edge is large (S0 +0.52,
S1 +0.23, S3 +0.75).

**Reconciliation with 3A.** In the narrow band [8,64] (3A), a single fixed slot (76) was near-optimal,
so `ADAPTIVE_SELECTOR_ADVANTAGE` was only DIRECTIONAL. Broadening the target support so that no fixed
snapshot can see every target reveals a **decisively qualified** adaptive advantage: parameter-free
confidence selection is genuinely valuable precisely when the target location is unknown and variable —
the realistic setting. This is the qualified positive that the 06D audit's `ADAPTIVE_SELECTOR_
INCREMENTAL_ADVANTAGE = NOT_QUALIFIED` correctly said 06D had not established.

## Consequence

Recovery + adaptive selection both QUALIFIED in the wide-target regime ⇒ proceed to Section 4
end-to-end economics (and the optional non-synthetic scout). No prior artifact modified; nothing
pushed.
