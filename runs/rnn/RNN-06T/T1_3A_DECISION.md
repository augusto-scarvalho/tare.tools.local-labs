# RNN-06T — 3A DECISION (exact-contract transportability)

## Verdicts (reported separately, not collapsed)

- **`HISTORICAL_RECOVERY_TRANSPORT = QUALIFIED`**
- **`ADAPTIVE_SELECTOR_ADVANTAGE = DIRECTIONAL`**
- **`OFFICIAL_MAMBA_SYNTHETIC_TRANSPORT = PARTIAL`** ⇒ 3B OPEN (recovery positive).

Official `state-spaces/mamba2-1.3b` @ `c5b59d00`, mamba_ssm 2.2.4 fast path (T0-qualified single-pass
capture), fresh disjoint set `qualificationSetSha256_3A = 5e47408e…` (N=192, overlap 0 vs all 9 prior
sets). MAX_CONFIDENCE frozen (unchanged from 06D). Executed-source PROVEN (runner blob `56700c0c…`,
dirty ∅). fast_path_active=True (446,400 kernel calls, 0 fallback); 12 single-pass runs, 960
snapshots captured/restored, 0 boundary failures; weights immutable. Runtime 226 s, peak VRAM 7.7 GB.

## Arm accuracies (N=192, chance 1/256)

| arm | acc |
|---|---:|
| FINAL (same-run final state) | 0.219 |
| FIXED_SLOT_76 (non-adaptive control) | 0.771 |
| MAX_CONFIDENCE (frozen adaptive) | 0.823 |
| ORACLE_TARGET_PROXIMAL (diag) | ~0.90 |
| ORACLE_BEST_GOLD (diag upper bound) | 0.911 |
| MATCHED_NO_HISTORY (compute control) | 0.219 (== FINAL) |
| pool per-slot | 38:0.500 · 76:0.771 · 115:0.469 · 153:0.260 |

The per-slot profile matches RNN-06D (transformers) almost exactly (06D: 38:0.49, 76:0.77, 115:0.46,
153:0.29), and MAX_CONFIDENCE 0.823 ≈ 06D's 0.833 — the phenomenon transports across implementation,
tokenizer wrapper, and capture mechanism (re-prefill → genuine single-pass).

## CLAIM 1 — historical recovery transport: QUALIFIED

- `FIXED_SLOT_76 − FINAL = +0.552`, 95% CI [0.479, 0.620] (recovered 106, harmed 0).
- `MAX_CONFIDENCE − FINAL = +0.604`, 95% CI [0.531, 0.677] (recovered 118, harmed 2, net 116).
- Both ≫ SESOI_RECOVERY 0.15, CI lower bounds ≫ 0.05, robust 3/3. ⇒ **QUALIFIED.** Historical-state
  recovery is real on the official fast-path substrate: even a fixed early/middle snapshot recovers
  the large majority of FINAL's forgetting failures.

## CLAIM 2 — adaptive selector incremental value: DIRECTIONAL

- `MAX_CONFIDENCE − FIXED_SLOT_76 = +0.0521`, paired 95% CI **[−0.0156, 0.1147]** — includes 0.
  Point estimate meets SESOI_ADAPTIVE (0.05) and is robust 3/3, but the CI lower bound is negative.
  ⇒ **DIRECTIONAL, not QUALIFIED.** MAX_CONFIDENCE recovers 12 more examples than FIXED_SLOT_76 (118
  vs 106) at the cost of 2 harms; the net edge is small and not separated from zero.

This **prospectively confirms the RNN-06D audit** (`ADAPTIVE_SELECTOR_INCREMENTAL_ADVANTAGE =
NOT_QUALIFIED`): on 06D data the adaptive-vs-fixed increment was a never-preregistered +0.0625; here,
preregistered on fresh official-substrate data with a paired CI, it is +0.052 [−0.016, 0.115] —
directional but unqualified. The headline recovery is overwhelmingly attributable to *using a fixed
early snapshot at all*, not to the adaptive confidence selection.

## Consequence

`OFFICIAL_MAMBA_SYNTHETIC_TRANSPORT = PARTIAL` (recovery QUALIFIED, adaptive DIRECTIONAL) — a valid,
non-binary outcome per pre-registration. Recovery transport is positive ⇒ **3B (wide-target) OPEN**,
where a fixed control is not guaranteed to have seen every target and the adaptive selector faces a
genuinely harder selection problem. No 06A/06D artifact modified; nothing pushed.
