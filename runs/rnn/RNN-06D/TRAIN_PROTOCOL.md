# RNN-06D — Historical-State Recovery Ceiling + Parameter-Free Utility Train — PROTOCOL

**Author:** Claude (Opus 4.8) via Claude Code. **Date:** 2026-08-12. **Branch:** `master`
(no upstream). **Pushed:** NO. Two backlog items, one hard dependency gate.

Thresholds and SESOI in this document are frozen **before any calibration or qualification
outcome**. Calibration may choose ONLY the bounded snapshot-schedule configuration (K, target band,
batching). It may NOT tune any threshold. No threshold is changed after outcomes.

## Frozen subject (verified live, must match)

`AntonV/mamba2-1.3b-hf` @ `703e19a43f397c70315244a3424d79456b54fb34`; transformers-native
`Mamba2ForCausalLM`; transformers 4.48.3; torch 2.6.0+cu124; bf16; no quantization; no
mamba_ssm/causal_conv1d (`is_fast_path_available=False`, naive `torch_forward`); `chunk_size=32`.
State = `Mamba2Cache{conv_states, ssm_states}` = 52,002,816 bytes/sequence (remeasured, not assumed).

## Prior scientific state (reconstructed from repo; NOT rewritten)

- RNN-06A2 continuation lifecycle = QUALIFIED (cs=32).
- RNN-06B3 controlled unique-binding-load forgetting = QUALIFIED (`TRANSITION_SHAPE = GRADED`).
- RNN-06C historical-state information PRESENCE = QUALIFIED.
- Automatic recovery utility = NOT TESTED (this train).
- Synthetic dense post-hoc Memory Caching = PARKED. Qwen GDN transplant = DEFER.

## Anti-oracle construction (new; `ops/rnn_06d_lib.py`)

Differs from B3/06C (which fixed the target at slot 0). Here the **target-write slot t is
randomized** in a preregistered valid band [T_MIN, T_MAX] (chosen at calibration). Every non-target
slot carries a **unique DS (disjoint filler-space) load binding** — general recurrent-state load
(the B3 primary arm), no same-scored-space competition. Fixed total length 4·M+2 (M=192 → 770
tokens). Scored-space target ⇒ constrained argmax over 256 scored values, chance 1/256, gold =
target value. Query `target_key =` appended.

**Snapshot schedule (target-agnostic).** K interior slot boundaries at fixed fractions
`floor(M·(k+1)/(K+1))`, k=0..K-1, all strictly before FINAL. The schedule depends on neither t nor
the gold answer. A snapshot at slot s = the recurrent state from an **independent prefill of the
prefix [0:4(s+1)]** (transformers-native Mamba2 has no multi-token mid-sequence forward; the decode
path is single-token, prefill requires `cache_position[0]==0`). This makes each snapshot a
well-defined, reproducible object; a boundary self-check re-prefills and matches conv/ssm state
hashes (as RNN-06C validated). No BIT_EXACT-vs-full-prefill claim is needed or made.

## Snapshot temporal identity (HARD; per 06C)

Every snapshot persists: exampleId, snapshotOrdinal, sequenceTokenPosition, associationSlotPosition,
recurrenceBoundaryId, cachePosition, prefixTokenSha256, convStateSha256, ssmStateSha256,
combinedStateSha256, model/revision/backend/source identity, chunkSize, dtype.
**`cachePosition == len(prefix)` and boundary self-check reproduction are HARD: any boundary
mismatch = invalid run / STOP.**

---

## BACKLOG 1 — RNN-06D0 — Recovery Ceiling & Snapshot Schedule Qualification

**Question:** does the target-agnostic historical snapshot pool contain enough *recoverable* target
information to justify building a recovery mechanism? (Not yet a deployed selector.)

**Arms.** FINAL (full-sequence state). FIXED_HISTORICAL_POOL (all K target-agnostic snapshots;
substrate for D1). ORACLE_TARGET_PROXIMAL (first pool snapshot at/after t; **diagnostic**, uses t).
ORACLE_BEST_GOLD (any pool snapshot correct vs gold; **upper bound**, uses gold). Optional
MATCHED_NO_HISTORY compute control (K readouts all from FINAL state; same compute, no history).

**Calibration/qualification separation.** Fresh calibration set chooses K∈{2,4,8}, target band
[T_MIN,T_MAX], batching. Then freeze `calibrationSetSha256`, `snapshotScheduleSha256`,
qualificationSet generator/version, resource ceiling, schedule-selection rule, source/model/backend
identities. Generate a **fresh disjoint** qualification set. No seed screening. Family not enlarged
after results.

**Frozen thresholds (set now, before any outcome):**
- `CEILING_SESOI = 0.15` — ORACLE_BEST_GOLD acc − FINAL acc.
- `FINAL_ACC_MAX = 0.75` — FINAL must be at/below this (a genuine forgetting regime).
- `TAU_PROX = 0.75` — ORACLE_PROXIMAL competence (confirms the target IS historically retrievable).
- Robustness: ORACLE_BEST − FINAL ≥ `CEILING_SESOI` in ≥ 2 of 3 strata.
- Paired ORACLE_BEST − FINAL stratified-bootstrap 95% CI lower bound > 0.05.
- Recoverable substrate for D1: among FINAL-wrong examples, recovery ceiling (fraction correct by
  any pool snapshot, gold-aware) ≥ `RECOV_FRAC_MIN = 0.30` **and** absolute
  `n_recoverable ≥ RECOV_N_MIN = 20`.

**Gate — mint exactly `RECOVERY_CEILING ∈ {QUALIFIED | TOO_SMALL | NOT_TESTABLE}`:**
1. boundary/identity/counter checks fail ⇒ `NOT_TESTABLE` (invalid machinery);
2. else FINAL not degraded (acc > FINAL_ACC_MAX) **or** proximal not competent (< TAU_PROX) ⇒
   `NOT_TESTABLE` (forgetting regime / historical retrievability absent);
3. else all of {ORACLE_BEST−FINAL ≥ CEILING_SESOI, CI_lb > 0.05, robust ≥ 2/3, recoverable ≥ frac
   & N} ⇒ `QUALIFIED`;
4. else ⇒ `TOO_SMALL`.

If `TOO_SMALL`/`NOT_TESTABLE`: persist all negative evidence, set `RNN-06D1 = BLOCKED_BY_D0`, do
NOT run D1, finish packaging, STOP. One next recommendation: PARK historical-snapshot recovery on
this substrate and open the alternative current-state-memory line.

---

## BACKLOG 2 — RNN-06D1 — Target-Agnostic Parameter-Free Recovery Utility (only if D0 QUALIFIED)

**Question:** without target-write position and without the gold answer, can a simple target-agnostic
mechanism exploit the qualified historical pool with recovery materially exceeding harm? No trained
reader, no DART, no imported Memory Caching.

**Candidate arms (frozen family; no post-outcome additions).** RECENCY; MAX_CONFIDENCE; MIN_ENTROPY;
MAX_TOP1_TOP2_MARGIN; CONFIDENCE_X_RECENCY (score = top1_prob · (rank_recency+1)/K); LOGIT_ENSEMBLE
(mean constrained logits over pool); FINAL_PLUS_HISTORICAL (mean constrained logits of FINAL + pool).
None may use gold, actual target-write position, or oracle-best identity. D1 independently
re-captures the pool (its own mechanism-activation counters + boundary checks) — a method that does
not actually restore/score historical states is not evidence.

**Primary paired outcomes per method:** n_final_wrong, n_recovered, recovery_rate; n_final_correct,
n_harmed, harm_rate; net_recovery_count = recovered − harmed; net_recovery_rate; accuracy_delta =
acc_method − acc_final; stratified interval; per-stratum; oracle gap (ORACLE_BEST − method); selection
regret vs ORACLE_BEST; selectedSnapshotHistogram. Queries/conditions are NOT flattened into fake
independent replications.

**Mechanism activation counters (persist):** snapshotsCreated, snapshotsRestored,
candidateSnapshotsScored, historicalSelections, finalSelections, ensembleCalls,
selectedSnapshotHistogram, oracleCalls (diagnostic, separate), snapshotBoundaryChecks,
snapshotBoundaryFailures, queriesEvaluated.

**Economics (persist, split compile / cold / warm):** #historical snapshots exposed, state bytes
(remeasured), peak CPU RAM, peak VRAM, snapshot capture time, GPU→host transfer, restore time,
readout time, total added latency, recovery per MiB, net recovery per added ms. The re-prefill
capture cost is a **naive-backend artifact** (a fast-path/incremental capture folds it into the
single forward pass) and is reported separately from the intrinsic mechanism cost.

**Frozen utility thresholds (set now, before any outcome):**
- `UTILITY_SESOI = 0.05` on accuracy_delta.
- QUALIFIED_PARAMETER_FREE requires a preregistered arm with accuracy_delta ≥ UTILITY_SESOI,
  net_recovery_count > 0, accuracy_delta stratified-bootstrap 95% CI lower bound > 0, robust
  (accuracy_delta ≥ 0) in ≥ 2/3 strata, and COST_OK.
- `COST_OK`: intrinsic added memory ≤ K × 52,002,816 bytes **and** intrinsic added per-query latency
  (restore + readout + selection, warm steady state) ≤ 100 ms. (Re-prefill capture reported but
  excluded from this bound as a substrate artifact.)

**Gate — mint exactly `RECOVERY_UTILITY ∈ {QUALIFIED_PARAMETER_FREE | SEMANTIC_GAIN_COST_FAIL |
ORACLE_GAP_REMAINS | NOT_USEFUL}`:**
- best arm meets semantic + cost ⇒ `QUALIFIED_PARAMETER_FREE`;
- best arm meets semantic gain but not COST_OK ⇒ `SEMANTIC_GAIN_COST_FAIL`;
- best arm has accuracy_delta > 0 and net_recovery_count > 0 but fails the SESOI/CI/robustness bar ⇒
  `ORACLE_GAP_REMAINS` (captures some, not a useful fraction of the D0 ceiling);
- best arm accuracy_delta ≤ 0 (no positive value) ⇒ `NOT_USEFUL`.

**Final recommendation mapping (exactly one; NOT executed here):**
`QUALIFIED_PARAMETER_FREE` → open official-Mamba transportability replication (new session).
`ORACLE_GAP_REMAINS` → open ONE trained-selector / DART-like historical-state experiment (new
session). `SEMANTIC_GAIN_COST_FAIL` → evaluate whether a bounded state-compression/replay experiment
could cross the measured Pareto gap, else pivot to current-state alternatives. `NOT_USEFUL` → PARK
historical-state retrieval on this substrate; open a current-state-memory comparison (StateX-like).

---

## Permanent boundaries

Do NOT: modify/reinterpret RNN-06B3/06C results; use target position/gold in primary recovery; train
a reader; implement DART; resurrect synthetic dense Memory Caching; implement StateX / Sparse Delta
Memory / GDN-2; repair old GDN checkpoints; run Qwen; change serving; optimize INT8/ReplaySSM; alter
Windows Update / host policy; push. Engineering amendments allowed only BEFORE affected outcomes,
append-only and explicit.

## Git / evidence discipline

Append-only commit boundaries: train protocol → D0 calibration → D0 prereg/identities → D0 impl →
D0 results → D0 decision → (if QUALIFIED) D1 prereg → D1 impl → D1 results → D1 decision → evidence/
handoff. No amend/rebase of outcome history. Nothing pushed. Executed-source proven per run
(runner blob == committed, dirty ∅) before outcomes. Bundle invariant: archive payload == manifest
payload == SHA256SUMS payload (excluding only the two metadata files); raises on any unmanifested
payload.
