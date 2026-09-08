# Published Independent Audit — RNN-07A

**Document ID:** `experiment.local-llm.rnn-07a.audit-reconciliation-2026-08-12`  
**Document type:** `experiment`  
**Status:** `EXPERIMENTAL`  
**Authority:** independent audit/reconciliation; no canonical architecture authority.

> The LongBench-v2 arm is preserved as a valid negative discovery, but the parent train is incomplete because the intended NoLiMa controlled bridge was not executed. The exact local audit artifact is retained outside the research repository and identified by SHA-256 in future publication receipts.

## Audited state

```text
RNN_07A_TRAIN
= ACCEPTED_PARTIAL_WITH_PROTOCOL_AND_PROVENANCE_RECONCILIATION

ECONOMICS_OUTPUT_COMPARABILITY_E1
= QUALIFIED

MARGINAL_RECOVERY_UTILITY_ON_STEP_PATH_E1
= QUALIFIED

RECOVERY_PATH_VS_FUSED_BASELINE_E1
= NOT_COMPETITIVE_WITH_FUSED

GENERAL_END_TO_END_DEPLOYMENT_UTILITY
= OPEN

RNN_07A_LONGBENCH_REALISTIC_ARM
= VALID_NEGATIVE_DISCOVERY

REALISTIC_TASK_COMPETENCE
= INSUFFICIENT_UNDER_TESTED_BM25_CONTROL

REALISTIC_FORGETTING_OPERATING_POINT
= BLOCKED

REALISTIC_HISTORICAL_RECOVERY_SIGNAL
= N/A_NO_OPERATING_POINT

REALISTIC_ADAPTIVE_SELECTION_SIGNAL
= N/A_NO_OPERATING_POINT

MODEL_TASK_COMPETENCE_ABSENCE
= NOT_ESTABLISHED

NATURAL_FORGETTING_PREMISE_FALSIFIED
= NO

NOLIMA_PARENT_FALLBACK_TRIGGER
= TRUE

NOLIMA_EXECUTED
= NO

RNN_07A_PARENT_TRAIN_COMPLETENESS
= INCOMPLETE

WORKLOAD_SOURCE_IDENTITY
= NOT_AUDIT_GRADE
```

## E1 economics closure

The RNN-06T2 economics output-domain defect was corrected prospectively. FINAL_FUSED, FINAL_STEP and RECOVERY now return scored value token IDs in the same frozen token-id domain.

Two clean process starts produced 80 pooled warm samples:

- FINAL_FUSED median 37.792 ms/query; p95 39.848.
- FINAL_STEP median 991.298; p95 1113.067.
- RECOVERY median 1014.626; p95 1159.144.
- RECOVERY−FINAL_STEP median +37.038; p95 +222.645 ms/query.
- RECOVERY−FINAL_FUSED median +974.964; p95 +1120.564.

With the frozen 250 ms/query marginal envelope:

`MARGINAL_RECOVERY_UTILITY_ON_STEP_PATH_E1 = QUALIFIED`.

This is not a claim that the recovery-capable step path is competitive with the fused baseline.

## LongBench-v2 natural arm

The discovery arm observed:

| Cell | N | control | full | eligible | forgotten |
|---|---:|---:|---:|---:|---:|
| ~16K | 8 | 0.125 | 0.250 | 1 | 0 |
| ~32K | 35 | 0.257 | 0.314 | 9 | 4 |

The frozen competence gate required control Wilson lower bound >0.35 and at least 20 competence-eligible examples. It did not pass, so recovery was correctly blocked.

The correct interpretation is narrower than “the model has no LongBench-v2 competence.” The competence control was a custom target-agnostic token-ID BM25 retrieval implementation, not the official LongBench-v2 retrieval path. Therefore the packet establishes insufficient competence **under the tested control**, not a model-level absence result.

Because competence did not qualify, the forgetting question is blocked rather than falsified.

## Parent-protocol deviation

The parent implementation packet requested the NoLiMa controlled bridge when realistic tasks produced too few competent examples for diagnosis. The local preregistration narrowed that condition to dataset unusability/loading failure. LongBench loaded, but only nine 32K examples were competence-eligible, so the parent fallback condition was met while NoLiMa remained unexecuted.

The LongBench negative remains valid; the parent RNN-07A train remains incomplete.

A second incompleteness is that the preregistered 64K cell (“if budget”) was omitted although the scout consumed only ~10.5 GPU minutes against the 90-minute scout target. This is secondary to the missing NoLiMa bridge.

## Provenance gap

The bundle includes benchmark name, local path, example IDs and derived predictions, but not an immutable external dataset revision/source hash, local `data.json` SHA-256, license record, or cryptographic identity for the selected source records. Thus the derived numerical artifacts are internally coherent but the external workload source identity is not audit-grade.

## Next corrective boundary

Continue the same RNN-07A research train with a bounded NoLiMa controlled bridge on the same exact official Mamba checkpoint. Repair external-source provenance before new outcomes. Preserve NoLiMa as `SEMI_SYNTHETIC_CONTROLLED_BRIDGE`, not a natural-workload result.

Do not open RNN-07B, Qwen or a new model subject until that bridge is audited.
