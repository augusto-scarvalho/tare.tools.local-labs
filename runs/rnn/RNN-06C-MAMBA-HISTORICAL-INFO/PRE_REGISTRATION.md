# RNN-06C-MAMBA-HISTORICAL-INFO — PRE-REGISTRATION

**Written and committed BEFORE any 06C outcome-bearing execution.** Executes ONLY if
`STATE_LOAD_FORGETTING_PERTURBATION = QUALIFIED` (RNN-06B3). Tests INFORMATION PRESENCE only.
**No recovery mechanism, no trained reader, no Memory Caching.** Mints
`HISTORICAL_STATE_INFORMATION ∈ {QUALIFIED | NOT_DETECTED | BLOCKED}`.

## 1. Scientific question

Under the qualified state-load perturbation, is target information functionally accessible from
an EARLIER recurrent state while being materially LESS accessible from the later high-load
recurrent state? This is presence, not exploitation/recovery.

## 2. Frozen subject & construction

Identical frozen subject/backend/cs=32 as RNN-06A2/06B3. Uses the B3 order-stable construction
(`rnn_06b3_lib.py`, M=192, REPEAT1 sentinel, target@slot0) and the frozen B3→06C dose rule:
**HIGH = the B3 qualified dose with maximum paired loss** (recorded in `B3_RESULTS.json →
c06_dose_selection.HIGH_U`). Readouts use the qualified RNN-06A2 continuation/restore semantics
(checkpoint/restore proven BIT_EXACT).

## 3. Fresh held-out 06C set — `historicalInfoSetSha256`

`ops/rnn_06c_challenges.py` (`generator_version = rnn06c_historical_info_v1`, `master_seed =
20260818`) emits deterministic examples (same schema as B3), disjoint from P0, 06B, B2-calib,
B2-qual, B3-calib, B3-qual (distinct seed + generator + example-level checks + distinct SHA).
All example/boundary identities frozen before outcomes. **No seed screening.**

## 4. Three state conditions (§16) — all from the SAME target-history prefix

For each example, the target occupies slot 0; the "same H" premise is guaranteed because H, N,
and L share the byte-identical target-slot-0 prefix (asserted). Each condition's readout is
produced by capturing a COMPLETE recurrent snapshot, RESTORING it (06A2 semantics), and decoding
the identical query tokens `target_key =`:

- **H — historical-direct:** snapshot = state after `prefill(target)` (4 tokens, boundary right
  after the target write). Restore → decode query → readout.
- **N — same-aged neutral control:** snapshot = state after `prefill(target + neutral body)`
  where the body is all REPEAT1 sentinel (U=1 arrangement), 768 tokens. Restore → decode query.
- **L — high-load final:** snapshot = state after `prefill(target + high-load body)` at U=HIGH
  (order-stable), 768 tokens. Restore → decode query.

N and L have identical continuation token count (764 body tokens), identical final query
position (768), identical query tokens, identical model/backend, and the identical H target
prefix. Only the body content differs (neutral sentinel vs unique high-load).

## 5. Snapshot temporal identity — HARD invariant (§15)

Every snapshot carries a machine-verifiable identity record:
`exampleId, snapshotRole ∈ {H,N,L}, sequenceTokenPosition, associationSlotPosition,
recurrenceBoundaryId, cachePosition, prefixTokenSha256, convStateSha256, ssmStateSha256,
combinedStateSha256, modelRevision, modelWeightsIdentity, backendSemanticsId, chunkSize, dtype,
runnerSourceSha256, runnerGitBlob, gitHead`.

**HARD invariant:** the prefix whose hash is recorded MUST end at the exact token boundary
represented by the captured recurrent state, i.e. `cachePosition == len(prefix_tokens)` AND
re-prefilling the recorded prefix reproduces the recorded state hashes. A deterministic
self-check runs BEFORE substantive outcomes on a preregistered audit sample (≥8 examples, all
roles): re-prefill prefix → assert `convStateSha256/ssmStateSha256` match; assert H equals the
post-target state of N and L (branch-from-same-H). **Any boundary mismatch ⇒ STOP / invalid run**
(`snapshotBoundaryFailures > 0` ⇒ BLOCKED).

## 6. Mechanism activation counters (§17)

Persist: `snapshotsCreated, snapshotsHashed, snapshotsRestored, historicalDirectReadouts,
neutralAgedReadouts, highLoadReadouts, branchPairsCompleted, snapshotBoundaryChecks,
snapshotBoundaryFailures, stateHashChecks, restoreChecks, queriesEvaluated`. Unexpected zero
counts invalidate the corresponding interpretation. `historical_state_enabled=true` is NOT
evidence — only nonzero executed counters are.

## 7. Primary causal endpoint & channels (§18) — predeclared

- **PRIMARY contrast: `neutral_minus_load = neutral_aged_accuracy − high_load_accuracy` (N − L)**,
  paired per example (both begin from the same H and reach the same query boundary). H is the
  positive anchor.
- Report (pooled population, denominators exposed): `historical_direct_accuracy`,
  `neutral_aged_accuracy`, `high_load_accuracy`, `neutral_minus_load`, `historical_minus_load`;
  and paired transition counts: `N_correct→L_wrong`, `N_wrong→L_correct`, `H_correct→L_wrong`,
  `H_wrong→L_correct`. Population metrics are primary; conditional analyses (e.g. among L-failures)
  may be added transparently but do NOT replace the pooled headline.

## 8. SESOI & gate (§19)

- **SESOI (N−L) = 0.15** absolute paired accuracy gap. Justification: the minimum
  neutral-over-load retention gap for target information to be meaningfully "present earlier but
  degraded later" rather than trivially noisy; trivial region `[−0.05, +0.05]`.
- Mint `HISTORICAL_STATE_INFORMATION`:
  - **QUALIFIED** requires ALL: (a) H competent (`historical_direct_accuracy ≥ 0.75`, target
    info accessible near the write); (b) N retains materially more than L (`N−L ≥ SESOI`) with
    stratified-bootstrap 95% CI lower bound `> 0.05`; (c) L reproduces the B3 degradation
    (`high_load_accuracy` within a tolerance band of the B3 U=HIGH DS accuracy, ±0.10); (d)
    robust across the 3 strata (`N−L ≥ SESOI` in ≥2/3); (e) all snapshot temporal-identity
    checks pass (`snapshotBoundaryFailures == 0`); (f) mechanism counters prove all required
    paths ran (all required counters > 0).
  - **NOT_DETECTED** — H competent and machinery valid but `N−L` below SESOI / CI includes trivial.
  - **BLOCKED** — machinery invalid (boundary failures, zero counters) or H not competent (info
    not even accessible near the write ⇒ the contrast is uninterpretable).
- No trained reader; no recovery.

## 9. Snapshot economics (§20)

Stream: capture → hash → CPU/in-memory restore+readout → release. Persist only a small
preregistered raw-snapshot audit sample (≤4 snapshots) if needed. Record peak CPU RAM, peak
VRAM, state bytes/seq (52,002,816), snapshot/copy time, restore time, total runtime.

## 10. Executed-source identity (§21) — before outcomes

Runner + lib SHA-256 + git blobs + HEAD + dirty; protocol hash; `historicalInfoSetSha256`;
model/revision; backend source hashes; chunk_size; dtype; versions. Assert
`is_fast_path_available is False`.

## 11. Invariants

No recovery, no reader training, no Memory Caching, no GDN repair, no Qwen, no RNN-06D, no seed
screening, no threshold change after outcomes, frozen model, nothing pushed. STOP after minting.
