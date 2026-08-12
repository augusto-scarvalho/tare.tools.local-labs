# RNN-06T2-T1R — PRE-REGISTRATION (frozen before substantive recovery/economics outcomes)

Executes because T0R qualified both gates (`OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE = QUALIFIED`,
`SINGLE_PASS_HISTORICAL_CAPTURE_T0R = QUALIFIED`). Fresh, disjoint calibration and qualification
data. `MAX_CONFIDENCE` is a **frozen** selector — no selector tournament.

## 0. Frozen scientific interpretation (BEFORE outcomes)

Schedule `[38,76,115,153]` (06D slot boundaries). Wide target band `[8,144]`. **slot153 is AFTER
every possible target in `[8,144]`.** Therefore the adaptive problem is **NOT** "no fixed snapshot
has observed every target." The real adaptive problem is choosing a temporal state under the
tradeoff:

```
NOT_YET_WRITTEN   (snapshot too early — target not yet encoded)
      vs
SEEN_AND_RETAINED (snapshot after write, before it is overwritten by subsequent load)
      vs
ALREADY_FORGOTTEN (snapshot too late — target encoded then overwritten)
```

A fixed late snapshot (e.g. slot153) has *seen* every target but has *forgotten* early ones; a
fixed early snapshot has *retained* early targets but *not yet written* late ones. The adaptive
selector's job is to pick, per example, a snapshot in the SEEN_AND_RETAINED band. This is the
hypothesis the wide-target test evaluates.

## 1. Substrate / identity (unchanged from T0R)

`state-spaces/mamba2-1.3b` @ `c5b59d00ec85d313adea86a08cad2a43c962dd3b`, official `mamba_ssm` 2.2.4
fast path, chunk_size 256, bf16, RTX 3090. Fixed batch size throughout (BATCH=16). Qualification is
conditional on this exact frozen checkpoint.

## 2. Construction (06D v2 anti-oracle) + fresh disjoint seeds

Target at a random slot; pre-target sentinel padding; post-target unique-DS load (forgetting).
Constrained scoring over the scored-value vocabulary; gold = target value; chance = 1/|scored vals|.

| set | seed | band | purpose |
|---|---|---|---|
| narrow qualification | 20261110 | [8,64] | formal transport replication vs RNN-06D |
| wide calibration | 20261120 | [8,144] | freeze fixed control(s) only |
| wide qualification | 20261121 | [8,144] | primary confirmatory test (fresh, disjoint) |

All disjoint from every historical seed and from each other; the challenge generator emits a
`disjointness_proof` (signature-overlap counts). No seed screening. No regeneration for poor results.

## 3. Arms

**Narrow (Section 7):** `FINAL`, `FIXED_SLOT_76`, `MAX_CONFIDENCE`, `ORACLE_BEST_GOLD` (diagnostic),
`ORACLE_TARGET_PROXIMAL` (diagnostic, retained prospectively). Report `HISTORICAL_RECOVERY_NARROW`
and `ADAPTIVE_SELECTION_NARROW` separately. Adaptive selection is **not required** to qualify in the
narrow band (the historical hypothesis is that the fixed checkpoint itself captures most of the
recovery value here).

**Wide (Sections 8–11):** `FINAL`, all four fixed snapshots `{slot38,slot76,slot115,slot153}`,
`MAX_CONFIDENCE`, `ORACLE_BEST_GOLD` (diag), `ORACLE_TARGET_PROXIMAL` (diag).

## 4. Fixed-control tie policy (Section 9; frozen BEFORE calibration outcomes)

Calibration accuracy is computed for every fixed snapshot. **Carry ALL fixed snapshots whose
calibration accuracy is within `TAU_TIE = 0.02` of the best** into qualification as
`CARRIED_FIXED_CONTROLS`. The primary adaptive gate requires `MAX_CONFIDENCE` to beat the
**STRONGEST** carried fixed control (evaluated on qualification, paired). Tie-break for
reporting a single "best" is deterministic: highest calibration accuracy, ties broken by **smallest
slot index** (NOT Python dict insertion order). `MAX_CONFIDENCE` is additionally reported against
**every** fixed snapshot. Any "MAX_CONFIDENCE − best fixed snapshot observed on qualification"
comparator is labeled **POST_HOC_DESCRIPTIVE**, not confirmatory.

## 5. Primary claims + SESOI (frozen)

Paired examples; stratified (region) bootstrap, 2000 resamples.

- **`WIDE_TARGET_RECOVERY_T1R`** — `MAX_CONFIDENCE` vs `FINAL`:
  `QUALIFIED` iff Δacc ≥ `SESOI_RECOVERY = 0.15` AND bootstrap CI lower bound > `0.05` AND robust in
  ≥ 3/4 region strata. `PARTIAL` iff Δacc ≥ 0.15 but CI/robustness not met. Else `NOT_REPLICATED`.
- **`ADAPTIVE_SELECTION_T1R`** — `MAX_CONFIDENCE` vs **strongest carried fixed control**:
  `QUALIFIED` iff Δacc ≥ `SESOI_ADAPTIVE = 0.05` AND CI lower bound > 0 AND robust ≥ 3/4 strata.
  `DIRECTIONAL` iff Δacc > 0 but thresholds not met. Else `NOT_QUALIFIED`.

These are **independent** mints — historical recovery may qualify even if adaptive incremental value
does not. They are NOT collapsed into a single "Mamba PASS".

## 6. Recovery / harm accounting (Section 11)

For every primary fixed control AND `MAX_CONFIDENCE`, persist: `N`, `n_final_wrong`, `n_recovered`,
`recovery_rate`, `n_final_correct`, `n_harmed`, `harm_rate`, `net_recovery_count =
n_recovered − n_harmed`, `net_recovery_rate`, accuracy delta, paired/stratified bootstrap CI,
per-stratum effect. Denominators exposed. Per-example correctness persisted. The bundle contains the
**scored-value token mapping** (`vset`: gold-column index ↔ token id) needed to reconstruct
gold-column indices from raw logits.

## 7. Mechanism activation evidence (Section 12)

Persist actual counters: `singlePassRuns`, `snapshotsCapturedInRun`, `snapshotsRestored`,
`candidateSnapshotsScored`, `historicalSelections`, `fixedSelections`, `finalSelections`,
`fastPathPrefillCalls`, `fastPathStepCalls`, `fallbackPathCalls`, `queriesEvaluated`,
`snapshotBoundaryChecks`, `snapshotBoundaryFailures`. `snapshotBoundaryChecks` counts **actual
performed assertions**. For a preregistered sample persist runId/exampleId/boundary/captured-hash/
replay-hash/match/restore-result.

## 8. Apples-to-apples economics (Section 13; envelope frozen BEFORE timing outcomes)

Every arm executes the **same semantic task**: same context + same target query + same constrained
scored-answer readout, returning the same type of scored answer. Timed arms:

- `FINAL_FUSED_EQUIVALENT_WORK` — fused/chunked prefill of the context **then the query readout**.
- `FINAL_STEP_EQUIVALENT_WORK` — single-pass step trajectory to FINAL **then the query readout**.
- `RECOVERY_ENABLED_EQUIVALENT_WORK` — step trajectory + in-run capture at K boundaries + K+1
  restore/readouts + `MAX_CONFIDENCE` selection → scored answer.

Measured separately: compile/build/autotune, cold, warm steady state, snapshot capture, state copy,
GPU→CPU transfer, restore, historical readout, selection, total wall-clock, CPU RAM, VRAM, snapshot
bytes, throughput impact. Persist RAW warm latency samples; report median/p25/p75/p95/n.

**Primary utility comparator (frozen):** `RECOVERY_ENABLED − FINAL_STEP_EQUIVALENT_WORK`. Rationale:
in-run mid-sequence capture is *only* available on the step path (the fused prefill kernel cannot
expose mid-sequence states), so a deployment that wants recovery must already run the step path; the
marginal cost of *enabling recovery* is therefore exactly this difference, holding the step path
constant. The `RECOVERY_ENABLED − FINAL_FUSED` difference (the orthogonal cost of choosing a
capture-capable path at all) is reported **descriptively**.

**Envelope (frozen, with robust margin; NOT the reused 1000 ms):** `ENVELOPE_MS = 250` ms/query
added. Rationale: an interactive-serving budget tolerates a sub-quarter-second retrieval premium per
query when it recovers otherwise-lost answers; 250 ms is a conservative ceiling. **Robust-margin
rule:** the gate uses the **p95** warm added-latency statistic (not the median) — the conservative
upper-warm statistic must remain inside the envelope.

**Mint:** `END_TO_END_RECOVERY_UTILITY_T1R`:
- `QUALIFIED` iff net_recovery_count > 0 AND wide `MAX_CONFIDENCE − FINAL` Δacc ≥ 0.05 AND
  p95(`RECOVERY_ENABLED − FINAL_STEP`) ≤ 250 ms/query.
- `COST_FAIL` iff the quality/net conditions hold but the p95 cost exceeds 250 ms.
- `NOT_QUALIFIED` iff no net/quality gain.
- `NOT_COMPARABLE` iff the arms do not return the same scored-answer type (they do, by construction).

The envelope is NOT tuned after observing timings. No claim is made that a hypothetical
capture-exposing kernel would remove the cost unless measured.

## 9. Statistical discipline (Section 14)

All primary recovery comparisons paired; stratified resampling. Qualification conditional on this
exact frozen checkpoint. Token logits are NOT treated as independent model replications.
`MAX_CONFIDENCE` frozen before qualification; no reselection after outcomes. Every fixed-control
result exposed, including negative evidence.
