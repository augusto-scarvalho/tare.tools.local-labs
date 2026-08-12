# RNN-06D0 — CALIBRATION DECISION (EXPLORATORY; no qualification outcome)

Calibration chooses ONLY the snapshot-schedule configuration. No threshold/SESOI was changed; those
were frozen in `TRAIN_PROTOCOL.md` before any calibration. Qualification runs on a fresh disjoint
set. No seed screening.

## AMENDMENT 1 (construction, pre-outcome) — v1 → v2

The first calibration used construction **v1** (every non-target slot = unique DS load). It produced
a strongly-degraded FINAL (acc 0.042) but the ORACLE arms were also weak — even ORACLE_TARGET_PROXIMAL
(the snapshot right after the target write) retrieved the target only 0.15 (K=2) → 0.40 (K=8), far
below `TAU_PROX = 0.75`. Cause: v1 loaded the state with unique bindings **before** the target too,
so the target was written into a near-saturated state and was never cleanly encoded — there was
nothing for even an oracle to recover, and the gate would (correctly) have returned `NOT_TESTABLE`
on a badly-posed construction.

**v2** (this train) isolates the quantity recovery must exploit: slots `[0, t-1]` = REPEAT1 sentinel
(low-info padding, target cleanly encoded), slot `t` = TARGET, slots `[t+1, M-1]` = unique DS load
(subsequent interference). This is a calibration-time configuration choice made BEFORE any
qualification outcome; append-only; thresholds unchanged; qualification set fresh + disjoint. Lib
`GENERATOR_VERSION = rnn06d_anti_oracle_random_target_v2_sentinel_pre_load_post`.

## v2 calibration sweep (fresh calib set; seed 20260901; N=48, 3×16; M=192; band [8,64])

`calibrationSetSha256 = 0bf7d2613c4054f602e01662c66eb97260adce59806b10ca33a6d5b75601a936`
FINAL acc = **0.146** (degraded ≤ 0.75 ✓; well above chance 1/256).

| K | schedule slots | ORACLE_BEST | ORACLE_PROXIMAL | OB−FINAL | recoverable (final-wrong) | robust | adequate |
|---:|---|---:|---:|---:|---|---:|:--:|
| 2 | [64,128] | 0.833 | 0.812 | 0.688 | 33/41 (0.80) | 3/3 | yes |
| 4 | [38,76,115,153] | 0.833 | 0.833 | 0.688 | 33/41 (0.80) | 3/3 | yes |
| 8 | [21,42,64,85,106,128,149,170] | 0.938 | 0.938 | 0.792 | 38/41 (0.93) | 3/3 | yes |

Adequacy signals (all frozen in the protocol, used here only as go/no-go, not tuned): OB−FINAL ≥
0.15, FINAL ≤ 0.75, PROXIMAL ≥ 0.75, recoverable frac ≥ 0.30, robust ≥ 2/3. All three K pass.

## Frozen configuration: **K = 4**, band **[8, 64]**, M = 192

Raw parsimony (smallest adequate K) = **K=2**, recorded as `chosen_K` in `D0_CALIBRATION.json`. It is
**superseded** here for **anti-oracle validity**: at K=2 the schedule [64,128] places *every*
snapshot at/after the target (band tops at 64), so there are **no pre-target distractor snapshots** —
the D1 target-agnostic "which snapshot, not knowing t" selection is trivial (every snapshot has seen
the target). **K=4** is the *smallest* adequate K whose schedule also contains a snapshot **before**
the target for part of the band (slot 38 < t for t∈(38,64]), so the anti-oracle selection is genuinely
non-trivial. The recovery ceiling at K=4 is **identical** to K=2 (OB−FINAL = 0.688); K=4 is chosen for
experimental validity, not effect size. K=8 (larger ceiling) is available but not selected, to avoid
post-hoc enrichment.

Consequence: at K=4, for a target at slot t, snapshot slot 38 is a pre-target distractor when t>38
(state has not seen the target → the recovery method must avoid it without knowing t), and slots
{76,115,153} straddle/follow the target with graded post-target load. This is the frozen substrate for
D0 qualification and (if QUALIFIED) D1.

Batch = 2, chunk_size = 32, peak VRAM 8.2 GB, calibration runtime 350 s.
