# BACKLOG-SLX08-RELEVANCE-PREFILL-11 preregistration

Task: Confirm relevance-selected prefill with exact noninferiority and balanced sample
Evidence class: `serving_runtime`

## Hypothesis

Across 126 fresh balanced 4096-token Qwen3.8 retrieval fixtures,
relevance-selected 8/16-block prefill will retain every target, achieve at
least 90% accuracy, beat naive position-only selection by at least 20 points,
and improve p50 TTFT by at least 1.10x without p95 regression. The one-sided
exact 95% upper bound on relevance failures among dense-success opportunities
must be at most 3%, expressed as a lower accuracy-delta bound of at least -0.03.

## Frozen inputs

- `runs/research/BACKLOG-SLX08-RELEVANCE-PREFILL-10/raw/receipt.json`
  SHA-256 `7f573c2ae762772aadf2368a1945adfcf29f45feef0f35153453538e3c87324a`
- `runs/research/BACKLOG-SLX08-RELEVANCE-PREFILL-10/REVIEW.json`
  SHA-256 `6d018172936bc13870b4a2e6dfe107fc497aebd7cdfcc2e77e5d09b276f3560c`
- `tools/research/run_slx08_physical_prefill_r4.py`
  SHA-256 `42141bffd6c51635f1b0ec6e1ff3b531f4c7ee2b8eca933ee3371b673b86bf6d`
- `tools/research/run_slx08_physical_prefill_r5.py`
  SHA-256 `42a0c405bedd7003a6d28b17152261ea690bf1fb5cd6ca6f3e18f6451fd94848`
- `tools/research/run_slx08_relevance_prefill_r6.py`
  SHA-256 `fccac2347d78c3307448fe30c4cdc25363863e01a935286263a86df034f847e2`

## Command

```powershell
python tools/research/run_slx08_relevance_prefill_r11.py --outdir runs/research/BACKLOG-SLX08-RELEVANCE-PREFILL-11
```

## Factors

- 126 fresh cases yield exactly nine observations at each of 14 evidence
  positions and 42 observations in each of three arm-order periods.
- Identical 4096-token prompts feed dense 16/16, naive alternating 8/16 and
  explicit relevance-selected 8/16 arms; R10 samples are not reused.
- Greedy streamed decode remains n_predict=16, temperature=0, top_k=1,
  seed=0 and cache_prompt=false on the same Qwen3.8/RTX 3090 runtime.
- Noninferiority uses a one-sided exact binomial upper confidence limit found
  from P(X <= observed failures | n, p)=0.05. With zero failures and n=126,
  the bound is 1-0.05^(1/126), below 3%.
- The active Python path is intentionally shortened to R11 -> frozen R6;
  R4, R5, R6 and R11 are all included in provenance/implementation evidence.

## Acceptance gates

- `dense_control`: `dense_requests ge 126`
- `naive_control`: `naive_requests ge 126`
- `relevance_treatment`: `relevance_requests ge 126`
- `route_observation`: `relevance_route_observation_rate eq 1.0`
- `evidence_retention`: `relevance_evidence_retention_rate eq 1.0`
- `retained_fraction`: `relevance_median_retained_fraction eq 0.5`
- `dense_semantic_floor`: `dense_accuracy ge 0.9`
- `relevance_semantic_floor`: `relevance_accuracy ge 0.9`
- `semantic_noninferiority`: `relevance_vs_dense_exact_delta_ci95_low ge -0.03`
- `selector_value`: `relevance_vs_naive_accuracy_delta ge 0.2`
- `ttft_gain`: `relevance_vs_dense_p50_ttft_speedup ge 1.1`
- `tail_safety`: `relevance_vs_dense_p95_ttft_speedup ge 1.0`
- `service_restore`: `original_service_restored eq 1`
- `embedding_integrity`: `embedding_health eq 200`

## Abort conditions

Abort on any R10 route/service/identity invariant, incomplete active-code
binding, incorrect 126-case balance, failure of invalid-index controls, or
restoration ambiguity. Never substitute the degenerate normal CI or reuse R10
rows as confirmatory samples.

## Allowed claims

- `SLX08_RELEVANCE_SELECTED_PREFILL_QUALIFIED_R11`
- `SLX08_RELEVANCE_SELECTED_PREFILL_REJECTED_R11`

Claims outside these codes are forbidden even if a metric looks favorable.
Claims remain bounded to exact-key client selection and server token compaction
on this model/panel; no server-side selector or generic sparse attention.
