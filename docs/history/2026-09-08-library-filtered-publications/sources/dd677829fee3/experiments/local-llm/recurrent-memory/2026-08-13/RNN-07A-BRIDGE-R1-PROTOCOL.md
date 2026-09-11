# RNN-07A-BRIDGE-R1 — True In-Run Historical Recovery Requalification

**Document ID:** `experiment.local-llm.rnn-07a-bridge-r1.protocol-2026-08-13`  
**Document type:** `experiment`  
**Status:** `PROPOSED`  
**Execution state:** `NOT_EXECUTED_AT_PUBLICATION`  
**Authority:** prospective experimental protocol only.

## Purpose

Close only the recovery-semantics defects found in the RNN-07A NoLiMa bridge audit.

Do not rerun or rewrite the already accepted short-context competence and long-context degradation results.

## Frozen subject and bridge

- `state-spaces/mamba2-1.3b` @ `c5b59d00ec85d313adea86a08cad2a43c962dd3b`;
- official Mamba fast path / already-qualified fixed-batch lifecycle semantics;
- NoLiMa `ONLYDirect` controlled bridge @ pinned source identity;
- 32K recovery cell;
- needle depth 15%;
- same four-choice teacher-forced option-likelihood readout;
- snapshot schedule remains 25/50/75/90% + FINAL;
- `MAX_CONFIDENCE` remains frozen.

No new selector, training, model change, Qwen or systems optimization.

## Fresh qualification population

Recovery may not reuse the historical first-48 subset as load-bearing evidence.

Before outcomes:

1. generate a fresh deterministic example/filler-offset family;
2. materialize a full generated-example identity hash;
3. prove disjointness from the historical recovery set;
4. evaluate the existing 512-token short-competence rule to define recovery eligibility;
5. use all eligible examples when feasible, or preregister a deterministic random/stratified cap before outcomes.

A cap may not be hidden only in source and may not be implemented as first-N template order.

## Required execution semantics

Every recovery example must run its complete 32K context as one canonical trajectory.

Capture actual intermediate recurrent states in-run at the declared boundaries and continue that same trajectory to FINAL.

Independent fresh prefix prefills such as `prefill_state(ctx[:cut])` are not valid for the load-bearing R1 claim.

For a preregistered audit subset, independent same-path replay must reproduce the captured boundary hashes.

## Arms and diagnostics

Load-bearing arms:

- FINAL
- SNAP_25
- SNAP_50
- SNAP_75
- SNAP_90
- frozen MAX_CONFIDENCE

Diagnostic only:

- `ORACLE_HISTORICAL_ONLY`: any of 25/50/75/90 correct, FINAL excluded;
- `ORACLE_ALL`: history + FINAL.

Report recovery/harm, net recovery, paired intervals, selector histogram, needle/template strata and reasoning-type strata.

## Separate mints

```text
HISTORICAL_INFORMATION_PRESENCE_R1 =
PRESENT | NOT_DETECTED | INCONCLUSIVE

TRUE_IN_RUN_COARSE_HISTORICAL_RECOVERY_R1 =
POSITIVE_SIGNAL | NO_NET_SIGNAL | INCONCLUSIVE

TRUE_IN_RUN_MAX_CONFIDENCE_R1 =
POSITIVE_SIGNAL | NO_SIGNAL | HARMFUL | INCONCLUSIVE
```

These answer different questions:

```text
historical information exists
!=
a fixed state has positive net utility
!=
MAX_CONFIDENCE can select the useful state
```

## Provenance

Every result must self-record:

- runner SHA-256 and Git blob;
- HEAD / dirty state;
- protocol SHA-256;
- qualification-set SHA-256;
- external-workload provenance SHA-256;
- exact model/revision/backend identity;
- temporal/run identity per snapshot.

Any source modification after outcome exposure must preserve the old result, package the exact diff and use fresh qualification data when the change can affect scientific computation.

## Decision routing

- If historical information is not detected: next research hypothesis may test finer snapshot spacing.
- If historical information is present but fixed/MAX_CONF recovery fails: next hypothesis may test a calibrated retention/selection signal.
- If a fixed state or MAX_CONF becomes positive: require a fresh bridge confirmation before natural-workload claims.

No combination of finer spacing and a new selector is authorized merely because the historical bridge recovery was negative.
