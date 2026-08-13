# RNN-07A-BRIDGE-R1 — TRUE IN-RUN HISTORICAL RECOVERY — PRE-REGISTRATION

Frozen BEFORE any R1 recovery outcome. Recovery-only corrective. Subject/workload/task/schedule
UNCHANGED; this changes **execution correctness + population governance only**. All historical bridge
results are preserved append-only. Nothing pushed. See `AUDIT_RECONCILIATION_BRIDGE.md`.

## Subject / workload (unchanged)

`state-spaces/mamba2-1.3b` @ `c5b59d00ec85d313adea86a08cad2a43c962dd3b`, official `mamba_ssm` 2.2.4 fast
path, bf16, RTX 3090, same qualified fixed-batch lifecycle semantics. NoLiMa `ONLYDirect` controlled
bridge (`amodaresi/NoLiMa` @ `378115b1…`) — **SEMI_SYNTHETIC_CONTROLLED_BRIDGE**, never a natural-workload
qualification. Task: DIRECT literal association, 4-way teacher-forced length-normalized option-likelihood,
natural-language book filler, needle depth 15%, recovery cell **32000 tokens**. Snapshot schedule
UNCHANGED: **25% / 50% / 75% / 90% / FINAL**. MAX_CONFIDENCE frozen exactly as before. No selector
tournament, no training, no Qwen, no DART/StateX/SDM/GDN-2/INT8/ReplaySSM, no host-policy change.

## Fresh R1 recovery qualification set (disjoint)

- **New seeds:** R1 pool seed `20261500`; R1 filler-offset base `20261510000` (disjoint family from the
  historical recovery filler seeds `20463400+k`); sample seed `20261501`; bootstrap seed `20261502`.
- **Source/task identical**, only seeds change. Pool built as before (ONLYDirect × tests × N_CHAR seeded
  char assignments), N_CHAR chosen so ≥ 64 short-correct examples are available.
- **Per-example identity (UID)** = sha256 over `needle_id | test | char | option_order |
  needle_text | question | filler_seed | context_token_sha256`. `R1QualificationSetSha256` = sha256 over
  the sorted list of selected UIDs.
- **Disjointness proof:** regenerate the historical recovery set (old pool seed `20261400`, old short
  correctness, `eligible[:90][:48]`, old 32K filler seeds) and compute its UIDs; assert the R1 selected
  UID set is **disjoint** (empty intersection) at the full-identity level. No outcome-based replacement.

### Competence eligibility (already-frozen rule; not a new mint)

Evaluate the already-frozen 512-token SHORT competence construction on the fresh examples. An example is
**recovery-eligible iff SHORT-correct** (same rule as the historical bridge). Thresholds are NOT tuned;
this only defines the population. Persist: N generated, N short-correct, reasoning-type distribution,
needle/template distribution.

### Population size + declared cap (governance fix)

- Target **N_RECOVERY = 64** (declared here, before outcomes; ≥ 48 required, ≥ 64 preferred).
- If ≥ 64 short-correct exist, draw **64** by a **deterministic stratified random** sample (seed
  `20261501`) stratified across **needle_id × reasoning_type** (proportional, remainder by seeded
  shuffle) — NOT first-N file/template order. If < 64 short-correct, use **all** short-correct (still
  ≥ 48 expected).
- Freeze and persist: sample seed, selected UID list, `selectedSetSha256`, per-stratum counts.

## True single-trajectory in-run capture (load-bearing)

For every selected example: construct the full **32000-token** context ONCE (needle at 15% depth). Batch
examples (capture batch `B_CAP = 32`, uniform length ⇒ uniform boundaries) and execute **ONE canonical
qualified trajectory** per batch via the already-qualified `ops/rnn_06t_lib.py::run_trajectory` (WARMUP
prefill + single-pass token stepping on the official fast path). Capture the actual recurrent states
**in-run** at boundaries `{round(0.25·L), round(0.50·L), round(0.75·L), round(0.90·L), L=32000}` and
continue the SAME trajectory to FINAL. **No independent prefix re-prefill** (`A.prefill_state(ctx[:cut])`
is prohibited for capture). No new/faster capture path is invented.

### Temporal identity + replay

Persist per (example, snapshot): packet ID, `R1QualificationSetSha256`, example UID, run ID, snapshot
role, token boundary, seqlen position, per-row state hash (SHA-256 of bf16 conv+ssm bytes), model
revision, backend identity. `runId` binds to the full R1 run/example identity (not a short common
prefix). **Replay:** re-run the trajectory for a preregistered subset (`REPLAY_SUBSET_N = 8`, the first 8
selected UIDs sorted) and require every captured boundary state hash to reproduce **exactly**
(in-process same-path replay). Any boundary mismatch ⇒ recovery authority invalidated (mint records the
mismatch).

## Readout + arms (frozen)

From each captured state, the SAME frozen NoLiMa readout (teacher-forced length-normalized 4-way
option-likelihood) evaluates the SAME question/options. Arms:
`FINAL`, `SNAP_25`, `SNAP_50`, `SNAP_75`, `SNAP_90`, `MAX_CONFIDENCE` (frozen: argmax option-confidence
over the 5 states), `ORACLE_HISTORICAL_ONLY` (diagnostic: correct iff ANY of SNAP_25/50/75/90 correct,
FINAL excluded), `ORACLE_ALL` (diagnostic: historical snapshots + FINAL). Oracles are diagnostic only; no
gold is used by deployable arms.

## Primary metrics (full R1 population)

Accuracy of every arm. For each fixed historical arm vs FINAL and for MAX_CONFIDENCE vs FINAL:
`n_final_wrong, n_recovered, n_final_correct, n_harmed, net_recovery, accuracy_delta,
paired_bootstrap_CI` (N_BOOT=2000, seed 20261502). Also: `ORACLE_HISTORICAL_ONLY` accuracy;
`historically_recoverable_final_wrong = count(FINAL wrong ∧ ≥1 historical snapshot correct)`;
`historical_recoverability_rate_over_final_wrong` (+ Wilson CI); `ORACLE_ALL` accuracy; selector
histogram; per-needle/template and reasoning-type strata; raw denominators. Persist per-example rows.

## Gates (frozen definitions)

- **`TRUE_IN_RUN_COARSE_HISTORICAL_RECOVERY_R1`** ∈ {POSITIVE_SIGNAL, NO_NET_SIGNAL, INCONCLUSIVE}.
  POSITIVE_SIGNAL iff **at least one** fixed historical arm (SNAP_25/50/75/90) beats FINAL with paired
  bootstrap 95% CI lower bound `> 0` AND `net_recovery > 0` AND `accuracy_delta ≥ REC_EFFECT_MIN = 0.05`.
  INCONCLUSIVE if `N_RECOVERY < 48`. Else NO_NET_SIGNAL.
- **`TRUE_IN_RUN_MAX_CONFIDENCE_R1`** ∈ {POSITIVE_SIGNAL, NO_SIGNAL, HARMFUL, INCONCLUSIVE}.
  POSITIVE_SIGNAL iff MAX_CONFIDENCE beats FINAL with paired CI LB `> 0` AND `delta ≥ 0.05`.
  HARMFUL iff `delta ≤ -0.05` AND paired CI **upper** bound `< 0`. INCONCLUSIVE if `N < 48`. Else
  NO_SIGNAL.
- **`HISTORICAL_INFORMATION_PRESENCE_R1`** ∈ {PRESENT, NOT_DETECTED, INCONCLUSIVE} (information-presence
  diagnostic — may be PRESENT even when no fixed snapshot has positive net utility and MAX_CONF fails).
  PRESENT iff **either** (a) `ORACLE_HISTORICAL_ONLY − FINAL ≥ 0.05` with paired bootstrap 95% CI LB
  `> 0`, **or** (b) `historical_recoverability_rate_over_final_wrong ≥ 0.20` with Wilson 95% LB `> 0`.
  NOT_DETECTED if both clearly fail. INCONCLUSIVE if `N < 48` or both borderline.

If the qualified single-pass 32K path cannot finish the frozen sample within the remaining 3-hour
ceiling, mint `TRUE_IN_RUN_RECOVERY_R1 = BLOCKED_BY_RUNTIME_BUDGET`, preserve evidence, and STOP — no
silent fallback to prefix re-prefill.

## Descriptive selector-calibration diagnostics (NON-GATING)

Selected-state confidence distribution; mean selected confidence for correct vs incorrect; Pearson
correlation(selected confidence, correctness); accuracy by coarse confidence bins (if N permits). NO
temperature scaling / calibration training / new retention score in R1 (that is a later hypothesis).

## Self-recorded provenance (in the R1 result artifact)

runner SHA-256 + git blob, HEAD, dirty state, R1 protocol SHA-256, `R1QualificationSetSha256`,
`selectedSetSha256`, external-workload-provenance SHA-256, model/revision/backend identity, kernel
counters (fast-path firing). If runner source changes after any R1 outcome: preserve old outcome, package
diff, do not reuse the same qualification set for a scientifically-affecting change.

## Next-recommendation cases (report exactly one, NOT executed)

A: history NOT_DETECTED → fresh finer-snapshot-spacing experiment. B: history PRESENT but fixed/MAX_CONF
recovery fails → fresh retention-signal / selector-calibration experiment (base frozen). C: a fixed arm
or MAX_CONF becomes positive → fresh confirmatory bridge before any natural-workload claim. Do not combine
finer spacing and a new selector in one next packet unless evidence makes them inseparable.
