# Published Independent Audit — RNN-07A-BRIDGE / NoLiMa Controlled Bridge

**Document ID:** `experiment.local-llm.rnn-07a-bridge.audit-reconciliation-2026-08-13`  
**Document type:** `experiment`  
**Status:** `EXPERIMENTAL`  
**Authority:** independent audit/reconciliation; no canonical tare.tools architecture authority.

---

RNN_07A_BRIDGE_TRAIN = ACCEPTED_WITH_RECOVERY_REQUALIFICATION_REQUIRED

BUNDLE_INTEGRITY = PASS

BRIDGE_SHORT_CONTEXT_COMPETENCE = SUFFICIENT
BRIDGE_LONG_CONTEXT_DEGRADATION = QUALIFIED_WITH_POPULATION_CAP_DEVIATION

BRIDGE_HISTORICAL_RECOVERY_SIGNAL_HISTORICAL_MINT = RECONCILED_NON_LOAD_BEARING
BRIDGE_ADAPTIVE_SELECTION_SIGNAL_HISTORICAL_MINT = RECONCILED_NON_LOAD_BEARING

PREFIX_REPREFILL_COARSE_SNAPSHOT_RECOVERY = EXPLORATORY_NO_NET_SIGNAL
PREFIX_REPREFILL_MAX_CONFIDENCE = EXPLORATORY_HARMFUL
TRUE_IN_RUN_HISTORICAL_RECOVERY_ON_NOLIMA_BRIDGE = NOT_TESTED

## Accepted bridge result

The controlled NoLiMa `ONLYDirect` bridge establishes short-context competence on the exact official Mamba-2 subject and a large conditional long-context degradation among short-correct examples:

- short: 97/112 = 0.8661; Wilson lower bound 0.7907;
- 8K: 23/90 = 0.2556;
- 16K: 23/90 = 0.2556;
- 32K: 20/90 = 0.2222.

The source truncates the 97 short-correct examples to `MAX_LONG_EVAL=90` without declaring that cap in the prose preregistration. This is a protocol deviation, but even an adversarial completion where all seven omitted examples remained correct cannot reverse the frozen degradation gate. The degradation result is therefore retained with the population-cap caveat.

This result is a `SEMI_SYNTHETIC_CONTROLLED_BRIDGE`: natural-language book filler plus a planted direct literal association. It is not a qualification of a natural workload or of NoLiMa's core beyond-literal-matching challenge.

## Major recovery reconciliation

The parent corrective required actual in-run snapshots from the already-qualified official-Mamba lifecycle.

The final bridge source instead implements `snapshot_eval()` by independently prefix-prefilling `ctx[:cut]` into a fresh cache at 25/50/75/90/100%.

Therefore the recovery experiment is a prefix-reprefill historical-state probe, not a single-trajectory in-run historical-state recovery test.

A second protocol deviation is `MAX_RECOVERY=48`, absent from the prose preregistration. Recovery uses the first 48 of the first 90 short-correct examples. This deterministic first-N subset contains only 7 of the 10 needle IDs and is not a random/stratified representation of the declared eligible population.

Consequently the historical `NO_SIGNAL` recovery/adaptive mints are preserved but downgraded to non-load-bearing observations.

## Recovery observations that remain useful

On the executed first-48 prefix-reprefill subset:

```text
FINAL          0.3333
SNAP_25        0.2292
SNAP_50        0.1250
SNAP_75        0.2500
SNAP_90        0.1875
MAX_CONFIDENCE 0.2083
```

No fixed state gives positive net utility versus FINAL and MAX_CONFIDENCE is harmful on this subset.

However, among the 32 FINAL-wrong examples, 13 (40.625%) are correct in at least one historical 25/50/75/90 prefix state.

Audit-only diagnostic:

```text
ORACLE_HISTORICAL_ONLY = 21/48 = 0.4375
FINAL                  = 16/48 = 0.3333
```

Thus the categorical explanation that the needle was already forgotten before every captured snapshot is not supported. Historical information exists for many examples; the open question is whether correct in-run states preserve the same signal and whether a selector can exploit it without net harm.

## Current experimental boundary

```text
NOLIMA_DIRECT_BRIDGE_SHORT_COMPETENCE = SUPPORTED
NOLIMA_DIRECT_BRIDGE_LONG_DEGRADATION = SUPPORTED_STRONGLY
TRUE_IN_RUN_HISTORICAL_RECOVERY_ON_BRIDGE = NOT_TESTED
COARSE_PREFIX_REPREFILL_HISTORY = EXPLORATORY_MIXED_SIGNAL
MAX_CONFIDENCE_ON_BRIDGE = EXPLORATORY_HARMFUL
NATURAL_WORKLOAD_GENERALIZATION = OPEN
QWEN = DEFER
```

The next corrective is deliberately narrow: fresh recovery examples, true single-trajectory in-run states, the same 25/50/75/90 schedule and frozen MAX_CONFIDENCE. Finer spacing or a new retention selector is not authorized until that semantic correction is tested.
