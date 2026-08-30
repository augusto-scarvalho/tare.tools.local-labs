# BACKLOG-SLX08-RELEVANCE-PREFILL-10 preregistration

Task: Resume relevance-selected prefill with complete abort receipt ledger
Evidence class: `serving_runtime`

## Hypothesis

Unchanged from R6-R9: on 64 frozen 4096-token Qwen3.8 retrieval fixtures,
relevance-selected 8/16-block prefill will retain every target, reach 90%
accuracy, remain non-inferior to dense, beat naive by 20 points, and improve
p50 TTFT by 1.10x without p95 regression.

## Frozen inputs

- `runs/autonomous/SLX08-RELEVANCE-R9-2026-08-30/FINAL.json`
  SHA-256 `f065df7920bfb7173db5912e08be3698d8a4045c2978e09f23682419c78d311f`
- `runs/autonomous/SLX08-RELEVANCE-R9-2026-08-30/WORKER_EXIT.json`
  SHA-256 `18bc271555a742fe3f9198d08f1ecf7de07f515131c8856eacf43e3a27546b24`
- `tools/research/run_slx08_relevance_prefill_r9.py`
  SHA-256 `451f4df230289ef4b436266cf4c2952d44e53db2e174a742fd9c227a4c48e5b5`
- `runs/autonomous/SLX08-RELEVANCE-R6-2026-08-30/FINAL.json`
  SHA-256 `d4968708056ee0ed301aae5070d9514cd62834aa759738cfc54c061327e3aef0`
- `runs/autonomous/SLX08-RELEVANCE-R6-2026-08-30/WORKER_EXIT.json`
  SHA-256 `275e8883fdbe34459c96c11fb94a0fc59d982200ca63df1bcd78b0039d8ccc75`

## Command

```powershell
python tools/research/run_slx08_relevance_prefill_r10.py --outdir runs/research/BACKLOG-SLX08-RELEVANCE-PREFILL-10
```

## Factors

All scientific factors remain unchanged. R9 reached and passed malformed-index
controls, then aborted before measured requests while writing their evidence.
R10 changes only the source ledger supplied to that evidence writer.

## Acceptance gates

- `dense_control`: `dense_requests ge 64`
- `naive_control`: `naive_requests ge 64`
- `relevance_treatment`: `relevance_requests ge 64`
- `route_observation`: `relevance_route_observation_rate eq 1.0`
- `evidence_retention`: `relevance_evidence_retention_rate eq 1.0`
- `retained_fraction`: `relevance_median_retained_fraction eq 0.5`
- `dense_semantic_floor`: `dense_accuracy ge 0.9`
- `relevance_semantic_floor`: `relevance_accuracy ge 0.9`
- `semantic_noninferiority`: `relevance_vs_dense_accuracy_delta_ci95_low ge -0.03`
- `selector_value`: `relevance_vs_naive_accuracy_delta ge 0.2`
- `ttft_gain`: `relevance_vs_dense_p50_ttft_speedup ge 1.1`
- `tail_safety`: `relevance_vs_dense_p95_ttft_speedup ge 1.0`
- `service_restore`: `original_service_restored eq 1`
- `embedding_integrity`: `embedding_health eq 200`

## Abort conditions

All R9 abort conditions remain. Operational aborts are not scientific results.

## Allowed claims

- `SLX08_RELEVANCE_SELECTED_PREFILL_QUALIFIED_R10`
- `SLX08_RELEVANCE_SELECTED_PREFILL_REJECTED_R10`

Claims outside these codes are forbidden even if a metric looks favorable.
No server-side selector, generic sparse-attention or production claim is allowed.
