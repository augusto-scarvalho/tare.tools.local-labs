# RNN-07A-BRIDGE — AUDIT RECONCILIATION (append-only, R1 ingestion)

This file is APPEND-ONLY and does not edit any historical artifact. The historical bridge mints remain in
their original files (`BRIDGE_SHORT_RESULTS.json`, `BRIDGE_LONG_RESULTS.json`, `BRIDGE_DECISION.md`) and
are **preserved unchanged**. This records the independent audit reconciliation (2026-08-13) that
triggered the recovery-only corrective RNN-07A-BRIDGE-R1.

## Reconciled verdicts (carried into R1)

| item | reconciled state |
|---|---|
| `BRIDGE_SHORT_CONTEXT_COMPETENCE` | **SUFFICIENT** (load-bearing; unchanged) |
| `BRIDGE_LONG_CONTEXT_DEGRADATION` | **QUALIFIED_WITH_POPULATION_CAP_DEVIATION** (undeclared MAX_LONG_EVAL=90; even adversarial worst-case over the omitted 7 stays ≫ 0.15) |
| `BRIDGE_HISTORICAL_RECOVERY_SIGNAL` (historical mint) | **RECONCILED_NON_LOAD_BEARING** |
| `BRIDGE_ADAPTIVE_SELECTION_SIGNAL` (historical mint) | **RECONCILED_NON_LOAD_BEARING** |
| `PREFIX_REPREFILL_COARSE_SNAPSHOT_RECOVERY` | **EXPLORATORY_NO_NET_SIGNAL** |
| `PREFIX_REPREFILL_MAX_CONFIDENCE` | **EXPLORATORY_HARMFUL** |
| `TRUE_IN_RUN_HISTORICAL_RECOVERY_ON_NOLIMA_BRIDGE` | **NOT_TESTED** |
| `COARSE_HISTORICAL_INFORMATION_PRESENCE_ON_EXECUTED_SUBSET` | **EXPLORATORY_POSITIVE** (13/32 FINAL-wrong had ≥1 historical 25/50/75/90 prefix-state answer correct; ORACLE_HISTORICAL_ONLY 21/48=0.4375 vs FINAL 16/48=0.3333) |

## The two load-bearing defects R1 corrects

1. **Execution semantics (load-bearing).** The historical recovery used
   `ops/rnn_07a_bridge_lib.py::snapshot_eval`, which for each progress point independently re-prefilled
   `ctx[:cut]` in a **new cache** (`A.prefill_state`). Those five states are **independently
   reconstructed prefix states**, NOT captures from one canonical full-context trajectory. This is the
   exact prefix-reprefill-vs-in-run distinction RNN-06T/T2 qualified. The historical recovery therefore
   supports only a `PREFIX_REPREFILL_HISTORICAL_STATE_PROBE`, not
   `TRUE_IN_RUN_HISTORICAL_SNAPSHOT_RECOVERY`. R1 replaces this with a **single canonical trajectory**
   (qualified `L.run_trajectory`) that captures the actual in-run recurrent states at the boundaries and
   continues the SAME trajectory to FINAL.
2. **Population governance.** The historical recovery ran on an **undeclared first-48** subset
   (`MAX_RECOVERY=48`, `eligible[:48]`), which is order-dependent (only 7 needle IDs; 33 world_knowledge
   / 15 commonsense). R1 uses a **fresh** deterministic set, a **declared** cap, and a **stratified**
   (needle-id × reasoning-type) random draw frozen before outcomes.

## Correction to the historical decision wording

The historical `BRIDGE_DECISION.md` stated "the needle is forgotten before any captured snapshot." The
executed data (13/32 FINAL-wrong historically recoverable) **do not support** that categorical claim. R1
does not repeat it; the coarseness hypothesis remains *plausible* but is not established as the cause.

## R1 scope (recovery-only corrective)

- Do NOT rerun LongBench parent, NoLiMa short competence, or NoLiMa long-degradation qualification.
- Keep exact subject (`state-spaces/mamba2-1.3b` @ `c5b59d00…`, official `mamba_ssm` fast path) and the
  frozen 25/50/75/90/FINAL schedule and 32K recovery cell. This corrective changes **execution
  correctness / population governance only.**
- Fresh disjoint R1 recovery set; frozen 512-tok SHORT-correct eligibility rule (not a new competence
  mint); declared stratified cap N≥64; true single-trajectory in-run capture; temporal identity +
  same-path replay hash check; frozen `MAX_CONFIDENCE`; ORACLE_HISTORICAL_ONLY and ORACLE_ALL kept
  distinct.
- Mint separately: `TRUE_IN_RUN_COARSE_HISTORICAL_RECOVERY_R1`, `TRUE_IN_RUN_MAX_CONFIDENCE_R1`,
  `HISTORICAL_INFORMATION_PRESENCE_R1`. Remain within the original 3-hour ceiling; on runtime overrun
  mint `TRUE_IN_RUN_RECOVERY_R1 = BLOCKED_BY_RUNTIME_BUDGET` and STOP (no silent fallback to prefix
  re-prefill). Nothing pushed. Do not open RNN-07B.
