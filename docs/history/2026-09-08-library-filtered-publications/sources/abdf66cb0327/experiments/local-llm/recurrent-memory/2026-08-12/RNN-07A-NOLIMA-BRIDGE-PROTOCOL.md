# RNN-07A — NoLiMa Controlled-Bridge Corrective Protocol

**Document ID:** `experiment.local-llm.rnn-07a.nolima-bridge-protocol-2026-08-12`  
**Document type:** `experiment`  
**Status:** `PROPOSED`  
**Authority:** experimental protocol only; no result minted here.

## Purpose

Complete the parent RNN-07A fallback omitted by the first LongBench-v2 discovery arm.

The bridge is explicitly classified:

`SEMI_SYNTHETIC_CONTROLLED_BRIDGE`

It must never be published as a natural/realistic workload confirmation.

## Frozen subject

Keep the exact official Mamba-2 subject and fixed-batch lifecycle already qualified by RNN-06T2. No checkpoint/backend change, no training, no new selector.

## Source provenance before outcomes

Record immutable repository/dataset revision, license/research-use constraints, materialization command, local source-file SHA-256, normalization identity and selected-example-set SHA-256. If raw third-party text is excluded from the audit bundle, preserve per-example source IDs and hashes of question/context/answer objects.

Result artifacts must self-record runner SHA-256, Git blob, HEAD/dirty state, protocol hash, source manifest hash, selected-set hash and model/backend identity.

## Stage 1 — short-context competence

Establish a prospectively frozen short-context competence baseline before searching for long-context degradation. Use a bounded short grid (for example ~1K/~2K/~4K) and freeze the minimum competence threshold and sample-count requirement before qualification outcomes.

If short-context competence is insufficient:

`NOLIMA_TASK_COMPETENCE = INSUFFICIENT`

and the historical-recovery branch is blocked.

## Stage 2 — long-context degradation

Only if short competence is sufficient, use a bounded preregistered length/placement grid such as 8K/16K/32K, with 64K only when runtime and budget justify it.

Do not screen needles/examples by long-context failure. Preserve NoLiMa's non-literal retrieval/placement semantics.

Mint:

`NOLIMA_LONG_CONTEXT_DEGRADATION = FOUND | NOT_FOUND_WITHIN_BUDGET | BLOCKED`.

## Stage 3 — historical recovery, conditional

Only if competence and long-context degradation qualify.

Use actual in-run recurrent snapshots from target-agnostic normalized context positions, frozen before recovery outcomes. `MAX_CONFIDENCE` remains frozen; no selector tournament.

Compare FINAL, each fixed snapshot, frozen MAX_CONFIDENCE and ORACLE_BEST_GOLD diagnostic only. Report recovery/harm, selector histogram, length/placement strata and paired intervals.

Mint:

- `NOLIMA_HISTORICAL_RECOVERY_SIGNAL`
- `NOLIMA_ADAPTIVE_SELECTION_SIGNAL`

with `POSITIVE_SIGNAL | NO_SIGNAL | INCONCLUSIVE | N/A_NO_OPERATING_POINT` as applicable.

## Stop rules

- If short competence fails: stop; do not tune prompts until green.
- If competence passes but degradation is not found: stop; do not extend thresholds automatically.
- If recovery is positive: stop for independent audit; do not open RNN-07B in the same session.
- No Qwen, DART, StateX, SDM, GDN-2, INT8 archive or ReplaySSM.

## Publication boundary

This protocol exists to complete an experimental fallback. No result is inferred from its publication or from external implementation activity.
