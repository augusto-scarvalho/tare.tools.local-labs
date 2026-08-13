# HANDOFF — RNN-07A-BRIDGE-R1 (TRUE In-Run Historical Recovery Requalification)

Recovery-only corrective completing RNN-07A-BRIDGE: replaces the prefix-reprefill recovery probe with a
true single-trajectory in-run capture, on a fresh disjoint declared-stratified set. **Nothing pushed.**

## Git / run identity

- **START HEAD:** `bcdb6d7` (RNN-07A-BRIDGE tip)
- **FINAL HEAD:** `5d97ae4` (branch `master`; bundle/handoff commits follow)
- **Subject (unchanged):** `state-spaces/mamba2-1.3b` @ `c5b59d00ec85d313adea86a08cad2a43c962dd3b`,
  official `mamba_ssm` 2.2.4 fast path (proven firing; `fallbackPathCalls=0`). MAX_CONFIDENCE frozen; no
  training/tournament; no Qwen/DART/StateX/SDM/GDN-2/INT8/ReplaySSM; no host-policy change.
- **Bridge workload (unchanged):** NoLiMa `ONLYDirect` `amodaresi/NoLiMa` @ `378115b1…` —
  **SEMI_SYNTHETIC_CONTROLLED_BRIDGE**, never a natural-workload qualification. Provenance in
  `EXTERNAL_WORKLOAD_PROVENANCE.json`; datasets excluded from the bundle.
- **Budget:** R1 GPU ≈ short-eligibility (~2 min) + 2× batch-32 32K in-run capture (~27 min) + readouts +
  in-process replay (2× batch-32 32K, ~24 min). Cumulative train well within the 3-hour ceiling.

## What R1 corrected (load-bearing)

1. **True in-run capture.** ONE canonical qualified `run_trajectory` per capture-batch (WARMUP prefill +
   single-pass token stepping on the fast path) captures the actual recurrent states in-run at
   25/50/75/90/FINAL and continues the SAME trajectory to FINAL — NOT independent `prefill_state(ctx[:cut])`
   re-prefills (the historical bridge's defect).
2. **Population governance.** Fresh set (new seeds), `R1QualificationSetSha256` over 64 UIDs, **disjoint
   from the historical first-48 (overlap 0)**; declared cap N=64; deterministic stratified random draw
   across needle_id × reasoning_type (not first-N). Eligibility = frozen 512-tok SHORT-correct rule
   (150/168 short-correct).

## Verdicts (N = 64)

| mint | verdict |
|---|---|
| `HISTORICAL_INFORMATION_PRESENCE_R1` | **NOT_DETECTED** |
| `TRUE_IN_RUN_COARSE_HISTORICAL_RECOVERY_R1` | **NO_NET_SIGNAL** |
| `TRUE_IN_RUN_MAX_CONFIDENCE_R1` | **NO_SIGNAL** |

Arm accuracy: FINAL 0.328; SNAP_25 0.312, SNAP_50 0.312, SNAP_75 0.328, SNAP_90 0.328; MAX_CONF 0.312;
ORACLE_HISTORICAL_ONLY 0.344; ORACLE_ALL 0.344. Fixed arms vs FINAL: Δ ∈ {−0.016, 0.000}, net ≤ 0.
MAX_CONF vs FINAL: Δ −0.016, CI [−0.047, 0.000], net −1 (NO_SIGNAL — not HARMFUL). Among 43 FINAL-wrong,
**only 1** is historically recoverable (rate 0.023, Wilson LB 0.004); ORACLE_HISTORICAL_ONLY − FINAL
+0.016 with CI LB = 0. Confidence corr(selected, correct) = +0.176 (weakly positive here).

## Headline finding

Under **true in-run capture**, coarse 25/50/75/90 historical states carry **no recoverable information
beyond FINAL** (1/43 FINAL-wrong recoverable). The historical bridge's **prefix-reprefill** probe had
reported 13/32 recoverable (ORACLE_HISTORICAL_ONLY 0.4375) — that "presence" was largely an
**execution-semantics artifact of prefix re-prefill**, not a property of the model's real in-run
trajectory. This vindicates the audit's insistence on true in-run capture and corrects the earlier
"forgotten before any snapshot" wording (the true reading: coarse in-run history simply carries no signal
here).

## Replay authority — ESTABLISHED (in-process, same shape)

In-process same-shape replay reproduces every boundary state hash **BIT_EXACT (40/40)** (two
`run_trajectory` passes on the exact capture batch, shape (32,32000), in one process) →
`R1_REPLAY_INPROCESS.json`. Two earlier replay attempts were mis-specified and are retained as negative
evidence, each a documented RNN-06T2 boundary condition: (a) in-runner replay used batch 8 vs batch-32
capture → batch-shape non-portability (0/40); (b) a standalone same-shape replay ran cross-process → bf16
kernel-autotuning divergence (0/40). The recovery arithmetic is computed entirely in-process at one
consistent batch-32 shape and is unaffected.

## Executed source

`ops/rnn_07a_bridge_r1.py` (in-run capture + readout + gates + provenance; stable seeds), 
`ops/rnn_07a_bridge_r1_replay.py` (in-process determinism verification), reuses
`ops/rnn_07a_bridge_lib.py`, `ops/rnn_07a_lib.py` (frozen readout), `ops/rnn_06t_lib.py` (qualified
`run_trajectory` / state hashing).

## Committed diffs (append-only; nothing pushed)

`4c6fdd7` R1 prereg+audit+runner · `abb13fc` pre-outcome fix · `ccec173` R1 outcome ·
`1611bb2`/`a7dc569` replay tooling · `5d97ae4` R1 decision + in-process replay authority.

## Authority / effect status

Record + true-in-run recovery requalification on the controlled bridge. Establishes that coarse in-run
history is not recoverable here and that the prior positive was prefix-reprefill-specific. Does not change
the natural-workload negative. No production/deploy effect. Nothing pushed.

## Exactly one next recommendation (NOT executed) — CASE A

`HISTORICAL_INFORMATION_PRESENCE_R1 = NOT_DETECTED` on the coarse grid ⇒ a fresh, separately preregistered
**finer / adaptive snapshot-spacing** experiment (denser capture between the 15% needle depth and the 25%
boundary) under the SAME true in-run capture semantics — base frozen, no new selector, no training. Do not
combine finer spacing with a new retention selector in one packet. Opens only after audit accepts
RNN-07A-BRIDGE-R1.
