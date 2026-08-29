# BACKLOG-QWEN38-Q8-KV-LONGCONTEXT-02 preregistration

Task: Repeat Qwen3.8 Q8 KV long-context utility through the serving chat contract
Evidence class: `serving_runtime`

## Hypothesis

Through the actual chat-serving contract, Q8_0 K/V cache is noninferior to F16 for exact associative retrieval at physical prompt depths near 8k and 16k tokens, while saving at least 500 MiB and retaining at least 90% of median decode throughput. Noninferiority requires a paired-bootstrap 95% lower bound for Q8-minus-F16 exact recall of at least `-0.05`, and each arm must reach at least 95% recall.

R1 is a frozen harness negative: 40/48 raw `/completion` calls returned empty one-token EOS responses symmetrically, so it cannot answer this chat-serving hypothesis and fresh inference is required.

## Frozen inputs

- R1 receipt `runs/research/BACKLOG-QWEN38-Q8-KV-LONGCONTEXT-01/raw/receipt.json` — SHA-256 `289b17a34e4bb1c298f9e34f8d914c47ffec7641850cc29a58d1992f5a8f2093`
- R1 review `runs/research/BACKLOG-QWEN38-Q8-KV-LONGCONTEXT-01/REVIEW.json` — SHA-256 `c88f586556e10d9120cf6790f6514b5406d9b4a99ffff564899fd16ee34ef393`
- Q8 R4 receipt `runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-04/raw/receipt.json` — SHA-256 `f94153b21ab3196000b321d06fb79b0b59c3862146de519f05d69cb47d2fa9fe`
- Q8 R4 review `runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-04/REVIEW.json` — SHA-256 `d3d230ca8fe27b198ce2170d54f1c95feae6cb53b22d15e14ccce98e727d3e54`
- `config/qualified_model_fleet.json` — SHA-256 `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`
- R1 runner, used only for the frozen generator/crossover/scorer implementation — SHA-256 `4197afacf8c2bd9c4f3a96802612afcd41a4ca47ed4f1190284d170cf13419c8`
- Server `/home/augus/opt/slop.cpp/b10165-71676e46c/bin/llama-server` — bytes `17920`, SHA-256 `efb2f06c19d26605a1934c0a9ed5b65dd69034e8765f2d29d0426b7a011cfbe2`
- CUDA library `/home/augus/opt/slop.cpp/b10165-71676e46c/bin/libggml-cuda.so` — bytes `75928784`, SHA-256 `ca18591576b67632bfb09eaee56b958ff951a2a0c558a36ea2232f15032a8c6b`
- Qwen3.8 model — bytes `17923394624`, SHA-256 `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`

## Command

```powershell
python tools/research/run_qwen38_q8_kv_longcontext_r2.py
```

## Factors

- Preserve R1's four fresh-process crossover `F16-Q8 | Q8-F16`, cache args, 24 unique paired cases, 31 decoys, exact scorer, filler counts, target positions, seeds, two warmups, model/server shape and service restoration.
- Repeat all 48 requests; no R1 output is reused as a scientific observation.
- Change only the request contract to `POST /v1/chat/completions`, one `user` message containing the frozen prompt, `chat_template_kwargs.enable_thinking=false`, `max_tokens=32`, temperature 0, top_k 1, seed `2026082822`, cache_prompt false, slot 0, non-streaming.
- Read answer from `choices[0].message.content`, finish reason from `choices[0].finish_reason`, and physical prompt count from `usage.prompt_tokens`. Recalibrated accepted bands remain `[7800,8200]` and `[15800,16200]`; live diagnostic predicts about 20 chat-template tokens over R1's 8129/16137 raw counts.
- A retained response is nonempty when normalized content is nonempty. Timing is nondegenerate only when predicted_n >= 2, predicted_ms >= 1 ms and predicted throughput is between 1 and 1000 tokens/s. These are integrity gates, not performance wins.
- Repeat R1 paired bootstrap exactly: 20,000 replicates, seed `2026082823`, 24 paired exact-recall differences. No trimming/replacement.

## Acceptance gates

- `binary_model_identity`: `binary_and_model_identity_verified eq True`
- `cache_treatment_identity`: `explicit_cache_controls_verified eq True`
- `balanced_crossover`: `valid_crossover_blocks eq 4`
- `sample_size`: `recorded_requests eq 48`
- `physical_context`: `requests_within_target_token_bands eq 48`
- `request_integrity`: `successful_response_rate eq 1.0`
- `chat_content`: `nonempty_content_rate eq 1.0`
- `timing_integrity`: `nondegenerate_timing_rate eq 1.0`
- `f16_retrieval`: `f16_exact_recall ge 0.95`
- `q8_retrieval`: `q8_exact_recall ge 0.95`
- `paired_noninferiority`: `paired_bootstrap_ci95_low_q8_minus_f16 ge -0.05`
- `throughput_nonregression`: `q8_vs_f16_median_tps_ratio ge 0.9`
- `memory_saving`: `median_vram_saving_mib ge 500.0`
- `service_recovery`: `service_gateway_embedding_restored eq True`

## Abort conditions

- Abort before requests on any frozen identity mismatch, unhealthy service baseline, occupied temporary port, wrong executable or wrong cache args.
- Abort after three consecutive HTTP failures or an embedding failure. Preserve partial evidence and restore 8080/Qwen3.8 and 8081 in `finally`.
- Do not modify prompts, sample, scorer, treatment order, token bands, integrity rules or statistics after observing R2 outputs.
- Executor stops at `EXECUTED`; independent review is mandatory. Concurrency remains blocked until R2 promotion.

## Allowed claims

- `QWEN38_Q8_KV_LONGCONTEXT_NONINFERIOR_R2`
- `QWEN38_Q8_KV_LONGCONTEXT_NOT_NONINFERIOR_R2`

No claim is allowed for concurrency, contexts beyond 16k, broad reasoning, other models, production deployment or exact equivalence beyond this construct.
