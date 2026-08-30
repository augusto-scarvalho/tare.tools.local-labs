# BACKLOG-SLX08-RELEVANCE-PREFILL-08 preregistration

Task: Resume relevance-selected prefill with direct-file import bootstrap
Evidence class: `serving_runtime`

## Hypothesis

Unchanged from R6/R7: across 64 frozen 4096-token Qwen3.8 retrieval fixtures,
relevance-selected 8/16-block prefill will retain every target, reach at least
90% accuracy, remain non-inferior to dense, beat naive alternating selection by
at least 20 points, and improve p50 TTFT by at least 1.10x without p95 regression.

## Frozen inputs

- `runs/autonomous/SLX08-RELEVANCE-R7-2026-08-30/FINAL.json`
  SHA-256 `7e339315aa0feef47cc5b6971a4892494f170802d5b944b3be9cc0461fa4ba0`
- `runs/autonomous/SLX08-RELEVANCE-R7-2026-08-30/WORKER_EXIT.json`
  SHA-256 `eb7caa299abc71fd960777821c4d4aae4afeb0139312ab76e4a5e4a20946f2fe`
- `tools/research/run_slx08_relevance_prefill_r7.py`
  SHA-256 `21edf4276875cbc6b36faae5f3109fb88c18a3a6ade92444fe70076cc905a196`
- Qwen3.8 GGUF SHA-256
  `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`

## Command

```powershell
python tools/research/run_slx08_relevance_prefill_r8.py --outdir runs/research/BACKLOG-SLX08-RELEVANCE-PREFILL-08
```

## Factors

All 64 fixtures, three arms, token budgets, selector, scorer, decode controls,
Latin arm rotation, gates and service lifecycle remain unchanged. R7 exited
before import, measurement or maintenance. R8 adds only repository-root
resolution before importing the frozen R7 wrapper.

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

All R7 abort conditions remain. Import or wrapper identity failure is an
operational abort, never a scientific rejection.

## Allowed claims

- `SLX08_RELEVANCE_SELECTED_PREFILL_QUALIFIED_R8`
- `SLX08_RELEVANCE_SELECTED_PREFILL_REJECTED_R8`

Claims outside these codes are forbidden even if a metric looks favorable.
No server-side selector, generic sparse-attention or production claim is allowed.
