# RNN-06D1 — PRE-REGISTRATION (Target-Agnostic Parameter-Free Recovery Utility)

Frozen BEFORE the D1 run. Executes because `RECOVERY_CEILING = QUALIFIED` (D0). Utility thresholds
were frozen in `TRAIN_PROTOCOL.md` and are restated unchanged. Same frozen subject, construction,
qualification set, and K=4 schedule as D0. No trained reader, no DART, no Memory Caching. No method
uses the gold answer, the actual target-write position, or the oracle-best identity.

## Independent re-capture (honest mechanism activation)

D1 **independently re-captures** the K=4 target-agnostic pool + FINAL (its own snapshot/restore/score
counters + boundary self-check), and cross-checks bit-determinism against `D0_READOUTS.npz`. A method
that does not actually restore and score historical states is not evidence for historical recovery.

## Frozen candidate family (no additions after outcomes)

Over the per-snapshot constrained value distribution (256 scored values) of the K=4 pool:
1. **RECENCY** — select the most recent pool snapshot (largest slot).
2. **MAX_CONFIDENCE** — select the snapshot with the highest top-1 probability.
3. **MIN_ENTROPY** — select the snapshot with the lowest entropy.
4. **MAX_TOP1_TOP2_MARGIN** — select the snapshot with the largest (p1−p2) margin.
5. **CONFIDENCE_X_RECENCY** — select argmax of `top1_prob · (k+1)/K` (k = schedule order).
6. **LOGIT_ENSEMBLE** — argmax of the mean constrained logits over the pool.
7. **FINAL_PLUS_HISTORICAL** — argmax of the mean constrained logits of FINAL + pool.

Plus a **MATCHED_NO_HISTORY** compute control (K FINAL readouts ensembled: same compute, no history).

## Primary paired outcomes (per method)

n_final_wrong, n_recovered, recovery_rate; n_final_correct, n_harmed, harm_rate;
net_recovery_count = recovered − harmed; net_recovery_rate; accuracy_delta = acc_method − acc_final;
stratified-bootstrap 95% CI on accuracy_delta; per-stratum delta; oracle gap (ORACLE_BEST − method);
selection regret vs ORACLE_BEST; selectedSnapshotHistogram. No flattening of queries into fake
independent replications.

## Mechanism-activation counters (persist)

snapshotsCreated, snapshotsRestored, candidateSnapshotsScored, historicalSelections, finalSelections,
ensembleCalls, selectedSnapshotHistogram, oracleCalls (diagnostic, separate), snapshotBoundaryChecks,
snapshotBoundaryFailures, queriesEvaluated.

## Economics (persist; compile / cold / warm)

#historical snapshots exposed (K=4), state bytes (52,002,816; × K = 208,011,264), peak VRAM/CPU RAM,
snapshot capture time, GPU→host transfer, restore time, readout time, total added latency, recovery
per MiB, net recovery per added ms. Re-prefill capture flagged as a naive-backend artifact and
reported separately from the intrinsic mechanism cost.

## Frozen utility thresholds (unchanged)

`UTILITY_SESOI = 0.05` on accuracy_delta. `COST_OK` = intrinsic added memory ≤ K × 52,002,816 bytes
AND intrinsic per-query latency (restore + readout + selection, warm) ≤ 100 ms.

## Gate — mint exactly `RECOVERY_UTILITY ∈ {QUALIFIED_PARAMETER_FREE | SEMANTIC_GAIN_COST_FAIL | ORACLE_GAP_REMAINS | NOT_USEFUL}`

Best arm = the preregistered method with the highest accuracy_delta.
- best meets {accuracy_delta ≥ 0.05, net_recovery_count > 0, delta CI_lb > 0, robust (delta ≥ 0) in
  ≥ 2/3 strata} AND COST_OK ⇒ `QUALIFIED_PARAMETER_FREE`;
- meets the semantic bar but not COST_OK ⇒ `SEMANTIC_GAIN_COST_FAIL`;
- best accuracy_delta > 0 and net_recovery_count > 0 but fails the semantic bar ⇒ `ORACLE_GAP_REMAINS`;
- best accuracy_delta ≤ 0 ⇒ `NOT_USEFUL`.

## Final recommendation mapping (exactly one; NOT executed here)

`QUALIFIED_PARAMETER_FREE` → open official-Mamba transportability replication (new session).
`ORACLE_GAP_REMAINS` → open ONE trained-selector / DART-like historical-state experiment (new
session). `SEMANTIC_GAIN_COST_FAIL` → evaluate a bounded state-compression/replay experiment vs the
measured Pareto gap, else pivot to current-state alternatives. `NOT_USEFUL` → PARK historical-state
retrieval; open a current-state-memory comparison (StateX-like).
