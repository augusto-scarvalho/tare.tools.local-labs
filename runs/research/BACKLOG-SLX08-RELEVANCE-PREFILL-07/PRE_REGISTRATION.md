# BACKLOG-SLX08-RELEVANCE-PREFILL-07 preregistration

Task: Resume relevance-selected prefill after pre-measurement tokenizer ordering failure
Evidence class: `serving_runtime`

## Hypothesis

Unchanged from R6: on 64 frozen 4096-token Qwen3.8 retrieval prompts,
query-key-selected 8/16-block prefill will retain every target, achieve at
least 90% accuracy, remain non-inferior to dense, beat naive alternating
selection by at least 20 percentage points, and improve p50 streamed TTFT by
at least 1.10x without p95 regression.

## Frozen inputs

- `runs/research/BACKLOG-SLX08-PHYSICAL-PREFILL-05/raw/receipt.json`
  SHA-256 `ed54519837bf23f174fcf4fdeef451a5d8e776fc8d3857b91b5e61b5190eb4eb`
- `runs/research/BACKLOG-SLX08-PHYSICAL-PREFILL-05/REVIEW.json`
  SHA-256 `1415bc422459a095e93b0474fca0d262474a8c87480ac618f3402bf477f96de6`
- `runs/autonomous/SLX08-RELEVANCE-R6-2026-08-30/FINAL.json`
  SHA-256 `d4968708056ee0ed301aae5070d9514cd62834aa759738cfc54c061327e3aef0`
- `runs/autonomous/SLX08-RELEVANCE-R6-2026-08-30/WORKER_EXIT.json`
  SHA-256 `275e8883fdbe34459c96c11fb94a0fc59d982200ca63df1bcd78b0039d8ccc75`
- `tools/research/run_slx08_relevance_prefill_r6.py`
  SHA-256 `fccac2347d78c3307448fe30c4cdc25363863e01a935286263a86df034f847e2`
- Qwen3.8 GGUF SHA-256
  `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`

## Command

```powershell
python tools/research/run_slx08_relevance_prefill_r7.py --outdir runs/research/BACKLOG-SLX08-RELEVANCE-PREFILL-07
```

## Factors

- The 64-fixture, three-arm, 4096-token/16-block, 8-block treatment, greedy
  decode, Latin arm rotation, answer scorer and all thresholds are unchanged.
- R6 made zero measured requests and failed before service maintenance because
  it tried to use the experimental tokenizer before starting that server.
- R7 changes only ordering: generate token fixtures through the already healthy
  qwen38 backend first, freeze them in memory, then enter the unchanged R6
  maintenance and measurement path.
- Runtime: experimental slop.cpp build, Qwen3.8 GGUF, RTX 3090, one slot.

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

All R6 abort conditions remain. Additionally abort if the live tokenizer route
is not the healthy qwen38 backend or if fixture generation changes any frozen
shape/selector invariant. Invalid operational execution is not scientific
rejection.

## Allowed claims

- `SLX08_RELEVANCE_SELECTED_PREFILL_QUALIFIED_R7`
- `SLX08_RELEVANCE_SELECTED_PREFILL_REJECTED_R7`

Claims outside these codes are forbidden even if a metric looks favorable.
The treatment remains client-selected token compaction before dense prefill;
no server-side semantic selector or generic sparse-attention claim is allowed.
