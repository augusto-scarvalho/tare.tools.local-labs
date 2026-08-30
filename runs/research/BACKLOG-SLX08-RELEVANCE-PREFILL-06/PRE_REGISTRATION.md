# BACKLOG-SLX08-RELEVANCE-PREFILL-06 preregistration

Task: Test relevance-selected half-context prefill against dense and naive controls
Evidence class: `serving_runtime`

## Hypothesis

For Qwen3.8 on 64 frozen 4096-token retrieval prompts, retaining the 8 of 16
blocks selected by exact query-key overlap will preserve every target evidence
block, reach at least 90% answer accuracy, remain non-inferior to dense prefill,
beat a position-only alternating selector by at least 20 percentage points, and
improve median streamed TTFT by at least 1.10x without p95 regression.

## Frozen inputs

- `runs/research/BACKLOG-SLX08-PHYSICAL-PREFILL-05/raw/receipt.json`
  SHA-256 `ed54519837bf23f174fcf4fdeef451a5d8e776fc8d3857b91b5e61b5190eb4eb`
- `runs/research/BACKLOG-SLX08-PHYSICAL-PREFILL-05/REVIEW.json`
  SHA-256 `1415bc422459a095e93b0474fca0d262474a8c87480ac618f3402bf477f96de6`
- Qwen3.8 GGUF SHA-256
  `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`
- The implementation source, shared-library and runner identities are frozen
  when the packet advances to IMPLEMENTED, before any measured request.

## Command

```powershell
python tools/research/run_slx08_relevance_prefill_r6.py --outdir runs/research/BACKLOG-SLX08-RELEVANCE-PREFILL-06
```

## Factors

- 64 fixtures, each exactly 4096 tokens in 16 blocks of 256 tokens.
- One exact case key and four-digit answer is placed in one rotating middle
  block; the final block queries that key. Other middle blocks are decoys.
- Three arms receive identical prompt-token bytes: dense 16/16; naive 8/16
  using the existing alternating policy; relevance 8/16 using an explicit,
  deterministic query-key selector that must retain the evidence block.
- Arm order follows a three-period Latin rotation by case ID.
- Greedy streamed decode: `n_predict=16`, `temperature=0`, `top_k=1`,
  `seed=0`, `cache_prompt=false`.
- TTFT is host monotonic time from POST to first non-empty streamed content.
- Runtime: the experimental slop.cpp build, Qwen3.8 GGUF, RTX 3090, one slot.

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

Abort on source/model/library mismatch, non-identical prompt hashes between
arms, missing or incorrect route telemetry, invalid block indices being
accepted, failure to retain the declared relevance block, malformed streaming,
port collision, unhealthy embedding service, or failure to restore the exact
stable gateway command/model/health identity. Invalid measurements are not
converted into scientific rejection.

## Allowed claims

- `SLX08_RELEVANCE_SELECTED_PREFILL_QUALIFIED_R6`
- `SLX08_RELEVANCE_SELECTED_PREFILL_REJECTED_R6`

Claims outside these codes are forbidden even if a metric looks favorable.
The treatment is client-selected server token compaction before ordinary dense
prefill. It is neither a server-side semantic selector nor generic sparse
attention, and the result is bounded to this model, context size and panel.
