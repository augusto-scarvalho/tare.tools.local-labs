# RNN-07A-BRIDGE-R1 — TRUE IN-RUN HISTORICAL RECOVERY — DECISION

Recovery-only corrective on the NoLiMa `ONLYDirect` SEMI_SYNTHETIC_CONTROLLED_BRIDGE. Subject unchanged
(`state-spaces/mamba2-1.3b` @ `c5b59d00…`, official `mamba_ssm` fast path; `fallbackPathCalls=0`,
`selective_state_update` fired; fast path proven). MAX_CONFIDENCE frozen; no training/tournament. Historical
bridge + LongBench results preserved unchanged (`AUDIT_RECONCILIATION_BRIDGE.md`). Prereg:
`R1_PRE_REGISTRATION.md`. Nothing pushed.

## Mints

| mint | verdict |
|---|---|
| `HISTORICAL_INFORMATION_PRESENCE_R1` | **NOT_DETECTED** |
| `TRUE_IN_RUN_COARSE_HISTORICAL_RECOVERY_R1` | **NO_NET_SIGNAL** |
| `TRUE_IN_RUN_MAX_CONFIDENCE_R1` | **NO_SIGNAL** |

## The load-bearing correction (vs the historical bridge recovery)

- **True single-trajectory in-run capture.** States are captured from ONE canonical qualified
  `run_trajectory` pass (WARMUP prefill + single-pass token stepping on the official fast path) at
  boundaries 25/50/75/90/FINAL, then the SAME trajectory continues to FINAL — NOT independent prefix
  re-prefills. This is the exact prefix-reprefill-vs-in-run distinction RNN-06T/T2 qualified.
- **Fresh, disjoint, declared-stratified population.** New seeds (pool 20261500, filler base
  20261510000, sample 20261501); `R1QualificationSetSha256` over 64 selected UIDs; **disjoint from the
  historical first-48 recovery set (overlap 0)**. Declared cap N=64 drawn by deterministic stratified
  random sample across needle_id × reasoning_type — not first-N order. Eligibility = already-frozen
  512-tok SHORT-correct rule (150/168 short-correct; not a new competence mint).

## Result (N = 64 in-run recovery examples)

| arm | accuracy |
|---|---|
| FINAL | 0.328 |
| SNAP_25 | 0.312 |
| SNAP_50 | 0.312 |
| SNAP_75 | 0.328 |
| SNAP_90 | 0.328 |
| MAX_CONFIDENCE (frozen) | 0.312 |
| ORACLE_HISTORICAL_ONLY (diagnostic) | 0.344 |
| ORACLE_ALL (diagnostic) | 0.344 |

- **`TRUE_IN_RUN_COARSE_HISTORICAL_RECOVERY_R1 = NO_NET_SIGNAL`.** No fixed historical arm beats FINAL:
  SNAP_25/50 Δ −0.016 (net −1), SNAP_75/90 Δ 0.000 (net 0). Every CI includes/sits ≤ 0.
- **`TRUE_IN_RUN_MAX_CONFIDENCE_R1 = NO_SIGNAL`.** MAX_CONFIDENCE 0.312 vs FINAL 0.328: Δ −0.016, CI
  [−0.047, 0.000], net −1. Not POSITIVE and not HARMFUL (Δ > −0.05, CI UB = 0). Selector spreads
  `{SNAP_25:14, SNAP_50:10, SNAP_75:17, SNAP_90:11, FINAL:12}`.
- **`HISTORICAL_INFORMATION_PRESENCE_R1 = NOT_DETECTED`.** `ORACLE_HISTORICAL_ONLY − FINAL = +0.016`, CI
  [0.000, 0.047] (LB not > 0); among 43 FINAL-wrong, **only 1** is historically recoverable by any
  25/50/75/90 snapshot (rate 0.023, Wilson LB 0.004) — far below the 0.20 presence rule. Both PRESENT
  conditions fail.

## Headline finding — the prefix-reprefill probe OVERESTIMATED historical recoverability

Under **true in-run capture**, historical snapshots carry essentially **no recoverable information beyond
FINAL** (1/43 FINAL-wrong recoverable; ORACLE_HISTORICAL_ONLY barely above FINAL). The historical bridge's
**prefix-reprefill** probe had reported 13/32 FINAL-wrong recoverable and ORACLE_HISTORICAL_ONLY 0.4375 —
an apparent "information presence." That signal was largely an **execution-semantics artifact of prefix
re-prefill** (chunked-prefill-of-prefix states differ numerically from the true stepped in-run states,
per RNN-06A Claim B). This is precisely why the audit demanded true in-run capture, and it materially
changes the interpretation: the coarse 25/50/75/90 in-run history does not hold the needle.

## Confidence diagnostics (non-gating)

Mean selected confidence: correct 0.859 vs incorrect 0.813; Pearson corr(selected confidence,
correctness) = **+0.176** (weakly positive here — NOT the anti-correlation the historical decision
inferred from the prefix-reprefill subset). But weak positive calibration is irrelevant when there is no
historical information to select: MAX_CONFIDENCE cannot recover what the snapshots do not contain.

## Temporal identity / replay authority — ESTABLISHED (in-process same-shape)

**In-process same-shape replay reproduces every boundary state hash BIT_EXACT (40/40)** — two
`run_trajectory` passes on the exact capture batch #1 (shape `(32, 32000)`) within one process match on all
first-8-row × 5-boundary hashes. The qualified capture path is therefore **deterministic at the capture
shape**, so the recovery-outcome states (all captured in-process at batch 32) are internally reproducible:
**recovery authority established.**

Two earlier replay attempts were **mis-specified** and are retained as honest negative evidence, each
explained by an already-documented RNN-06T2 boundary condition (NOT a capture defect):
- **(a) in-runner replay = 0/40**: re-ran the first-8 UIDs as a **batch of 8** vs the **batch-32** capture
  → `BATCH_SHAPE_NUMERICAL_PORTABILITY = OUT_OF_SCOPE` (batch-size states are not bit-portable).
- **(b) standalone same-shape replay = 0/40**: ran in a **separate process** → bf16 kernel-autotuning
  makes state bytes diverge across process starts.
- Consistently, this verification's **cross-process** match to the original capture process is also 0/40
  (`CROSS_PROCESS_MATCH_TO_CAPTURE = False`) — the documented bf16 cross-process boundary. The recovery
  arithmetic is computed entirely in-process at one consistent batch-32 shape and is unaffected.
Evidence: `R1_REPLAY_INPROCESS.json`.

## Scope / non-claims

Conditional on this frozen checkpoint and controlled ONLYDirect construction. SEMI_SYNTHETIC_CONTROLLED
BRIDGE — not a natural-workload qualification; does not change the natural-workload negative
(`REALISTIC_TASK_COMPETENCE = INSUFFICIENT`). The historical `BRIDGE_DECISION.md` claim "forgotten before
any captured snapshot" is NOT repeated; the true reading is that coarse in-run snapshots carry no
recoverable signal, and the earlier apparent signal was prefix-reprefill-specific. No goalpost moved, no
seed screening, MAX_CONFIDENCE unchanged, no new selector.

## Exactly one next recommendation (NOT executed) — CASE A

`HISTORICAL_INFORMATION_PRESENCE_R1 = NOT_DETECTED` on the coarse 25/50/75/90 grid ⇒ recommend a fresh,
separately preregistered **finer / adaptive snapshot-spacing** experiment (denser capture between the 15%
needle depth and the 25% boundary, where a still-retaining state may exist) under the SAME true in-run
capture semantics — base model frozen, no new selector, no training. Do NOT combine finer spacing with a
new retention selector in the same packet. Deferred; opens only after audit accepts RNN-07A-BRIDGE-R1.
