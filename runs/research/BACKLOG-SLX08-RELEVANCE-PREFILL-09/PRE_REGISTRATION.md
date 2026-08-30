# BACKLOG-SLX08-RELEVANCE-PREFILL-09 preregistration

Task: Resume relevance-selected prefill with corrected source digest
Evidence class: `serving_runtime`

## Hypothesis

Unchanged from R6-R8: on 64 frozen 4096-token Qwen3.8 retrieval fixtures,
relevance-selected 8/16-block prefill will retain every target, reach at least
90% accuracy, remain non-inferior to dense, beat naive selection by at least
20 points, and improve p50 TTFT by at least 1.10x without p95 regression.

## Frozen inputs

- `runs/autonomous/SLX08-RELEVANCE-R8-2026-08-30/FINAL.json`
  SHA-256 `d0c4fe952604ebe0e8fef4a3a9649ac1b1aff10a54333b41b22e53a4c2e37358`
- `runs/autonomous/SLX08-RELEVANCE-R8-2026-08-30/WORKER_EXIT.json`
  SHA-256 `6c0b13eee349d2172310f96cdcac3251ddc4fd4b88b867686d29f1dc120cb3f6`
- `tools/research/run_slx08_relevance_prefill_r8.py`
  SHA-256 `4ec2628c52d182c11d796295e44fdfdcb5be4415cde29d79812e60243e8e216d`
- The corrected delegated R7 FINAL digest is
  `7e339315aa0feeef47cc5b6971a4892494f170802d5b944b3be9cc0461fa4ba0`.

## Command

```powershell
python tools/research/run_slx08_relevance_prefill_r9.py --outdir runs/research/BACKLOG-SLX08-RELEVANCE-PREFILL-09
```

## Factors

All scientific and runtime factors remain unchanged. R8 made zero measured
requests and stopped before maintenance because its frozen R7 digest omitted
one hexadecimal `e`. R9 changes that digest only.

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

All R8 abort conditions remain. Any identity mismatch is an operational abort,
never a scientific rejection.

## Allowed claims

- `SLX08_RELEVANCE_SELECTED_PREFILL_QUALIFIED_R9`
- `SLX08_RELEVANCE_SELECTED_PREFILL_REJECTED_R9`

Claims outside these codes are forbidden even if a metric looks favorable.
No server-side selector, generic sparse-attention or production claim is allowed.
