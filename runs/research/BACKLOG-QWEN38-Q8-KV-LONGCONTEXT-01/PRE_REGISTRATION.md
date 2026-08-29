# BACKLOG-QWEN38-Q8-KV-LONGCONTEXT-01 preregistration

Task: Confirm Qwen3.8 Q8 KV utility at 8k and 16k context with associative decoys
Evidence class: `serving_runtime`

## Hypothesis

For the frozen Qwen3.8 artifact and single-slot runtime, Q8_0 K/V cache is noninferior to F16 for exact associative retrieval at physical prompt depths near 8k and 16k tokens, while saving at least 500 MiB and retaining at least 90% of median decode throughput. Noninferiority requires a paired-bootstrap 95% lower bound for Q8-minus-F16 exact recall of at least `-0.05`; each arm must also achieve at least 95% recall.

## Frozen inputs

- `runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-04/raw/receipt.json` — SHA-256 `f94153b21ab3196000b321d06fb79b0b59c3862146de519f05d69cb47d2fa9fe`
- `runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-04/REVIEW.json` — SHA-256 `d3d230ca8fe27b198ce2170d54f1c95feae6cb53b22d15e14ccce98e727d3e54`
- `config/qualified_model_fleet.json` — SHA-256 `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`
- Server `/home/augus/opt/slop.cpp/b10165-71676e46c/bin/llama-server` — bytes `17920`, SHA-256 `efb2f06c19d26605a1934c0a9ed5b65dd69034e8765f2d29d0426b7a011cfbe2`
- CUDA library `/home/augus/opt/slop.cpp/b10165-71676e46c/bin/libggml-cuda.so` — dereferenced bytes `75928784`, SHA-256 `ca18591576b67632bfb09eaee56b958ff951a2a0c558a36ea2232f15032a8c6b`
- Model `/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf` — bytes `17923394624`, SHA-256 `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`

## Command

```powershell
python tools/research/run_qwen38_q8_kv_longcontext.py
```

## Factors

- Four fresh-process blocks in crossover order `F16-Q8 | Q8-F16`; two adjacent process pairs control drift and treatment order.
- Each block runs 12 cases: target depths 8k and 16k, target positions start/middle/end, and two pair-specific replicates. Pair 0 uses replicates 0/1 and pair 1 uses 2/3, yielding 24 unique paired cases and 48 physical requests total.
- Each prompt contains 31 near-label/near-code decoys plus one exact `[ORION-DELTA]` target. Filler counts are frozen at 320 lines for 8k and 684 for 16k, calibrated before preregistration against the live qualified Qwen3.8 tokenizer. Every response must physically report `prompt_n` in `[7800,8200]` or `[15800,16200]` according to its target.
- Codes are deterministic and unique by target, position and replicate. Correctness is normalized exact match to the full code; incidental substrings or last-number extraction are not accepted.
- Server shape: context 32768, parallel 1, flash attention, all GPU layers, batch 2048, ubatch 512, 32 checkpoints, MTP draft maximum 3. The only changed arguments are both cache types, `f16` versus `q8_0`.
- Each process receives two excluded warmups. Recorded requests use `n_predict=32`, temperature 0, top_k 1, seed `2026082822`, cache_prompt false, slot 0 and non-streaming.
- GPU state is captured after warmup and after the recorded block. No failed request or outlier is replaced or trimmed.

## Analysis

- Exact recall is computed independently for each arm and target depth.
- Paired differences are `Q8 correct - F16 correct` for the same unique case. A deterministic 20,000-replicate paired bootstrap with seed `2026082823` resamples all 24 cases and reports the percentile 95% interval.
- Throughput uses the median server-reported predicted tokens/s over successful requests; the reported ratio is Q8/F16.
- VRAM saving is median F16 block memory minus median Q8 block memory from same-host global GPU snapshots. It is not per-process attribution.

## Acceptance gates

- `binary_model_identity`: `binary_and_model_identity_verified eq True`
- `cache_treatment_identity`: `explicit_cache_controls_verified eq True`
- `balanced_crossover`: `valid_crossover_blocks eq 4`
- `sample_size`: `recorded_requests eq 48`
- `physical_context`: `requests_within_target_token_bands eq 48`
- `request_integrity`: `successful_response_rate eq 1.0`
- `f16_retrieval`: `f16_exact_recall ge 0.95`
- `q8_retrieval`: `q8_exact_recall ge 0.95`
- `paired_noninferiority`: `paired_bootstrap_ci95_low_q8_minus_f16 ge -0.05`
- `throughput_nonregression`: `q8_vs_f16_median_tps_ratio ge 0.9`
- `memory_saving`: `median_vram_saving_mib ge 500.0`
- `service_recovery`: `service_gateway_embedding_restored eq True`

## Abort conditions

- Abort before requests on any host/WSL identity mismatch, unhealthy 8080/8081 baseline, occupied temporary port, wrong process executable or wrong cache arguments.
- Abort after three consecutive HTTP failures or any embedding-health failure between blocks; retain partial evidence and restore services in `finally`.
- Do not change filler counts, token bands, cases, order, scorer, thresholds or bootstrap after observing outputs.
- Stop the persistent gateway only through `llm-inference.service`; never stop 8081; restore the initial gateway model and verify both endpoints.
- Executor stops at `EXECUTED`; independent review is mandatory. The concurrency successor remains blocked until this packet is independently promoted.

## Allowed claims

- `QWEN38_Q8_KV_LONGCONTEXT_NONINFERIOR_R1`
- `QWEN38_Q8_KV_LONGCONTEXT_NOT_NONINFERIOR_R1`

No claim is allowed for concurrency, contexts beyond 16k, general reasoning, other models, production deployment or exact semantic equivalence outside this frozen construct.
