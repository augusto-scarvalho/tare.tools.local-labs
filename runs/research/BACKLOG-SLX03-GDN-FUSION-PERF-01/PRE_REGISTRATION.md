# BACKLOG-SLX03-GDN-FUSION-PERF-01 preregistration

Task: Measure causal SLX-03 GDN fusion decode performance on the qualified Release build
Evidence class: `serving_runtime`

## Hypothesis

For the frozen Qwen3.8 27B Q4_K_XL model, single-slot server shape, fixed 64-token decode work and qualified non-instrumented Release artifact, enabling the SLX-03 GDN fused-cache route increases decode throughput. The hypothesis is accepted only when the lower bound of the preregistered 95% hierarchical cluster-bootstrap confidence interval for the paired ON/OFF throughput ratio is greater than `1.0`, with exact output parity and no material end-to-end wall-throughput regression.

Failure of that statistical gate authorizes only `SLX03_GDN_FUSION_DECODE_SPEEDUP_NOT_DEMONSTRATED_R1`; it does not prove slowdown, absence of write reduction, or behavior outside this request shape.

## Frozen inputs

- `runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-04/raw/receipt.json` — SHA-256 `a46690e67b723368328a2f996d8b0d4e05e36d4c03e89590724561046e814029`
- `runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-04/REVIEW.json` — SHA-256 `59bfad4ba63444b508b45908d772547846603accfeb9e0c7f3539280b79667ba`
- `runs/research/BACKLOG-SLX03-GDN-FUSION-RUNTIME-03/raw/receipt.json` — SHA-256 `bb1391f3fb13792b0c71a658da4eb55eeeb64ac277cde00f93ee37bd78a9a256`
- `runs/research/BACKLOG-SLX03-GDN-FUSION-RUNTIME-03/REVIEW.json` — SHA-256 `4ef6e60706cf87ef74052eebffdeafb7d3901a644cc0af6e6a3ef3a32925ceb0`
- `config/qualified_model_fleet.json` — SHA-256 `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`
- Release server `/home/augus/src/slop.cpp-main/build-slx03-gdn-audit-03/bin/llama-server` — dereferenced bytes `17920`, SHA-256 `0267affe48ff9d49a13dbe0891b33598ead1179edd5db85ecb3b2c86c7e1fd0b`
- Release CUDA library `/home/augus/src/slop.cpp-main/build-slx03-gdn-audit-03/bin/libggml-cuda.so` — dereferenced bytes `63364248`, SHA-256 `378d85d3a09ae61982b016b186166dbe88a8dedf4ff9337dddafbe75ce70c7ce`
- Model `/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf` — bytes `17923394624`, SHA-256 `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`

The R4 review establishes Release-build inclusion/callability; the R3 review establishes treatment-specific route execution on the matching instrumented lineage. Neither prior packet contributes performance observations.

## Command

```powershell
python tools/research/run_slx03_gdn_fusion_perf.py
```

All outputs are written under `runs/research/BACKLOG-SLX03-GDN-FUSION-PERF-01`.

## Factors and sampling

- Hardware/runtime: the installed RTX 3090 under WSL Ubuntu-24.04; one temporary `llama-server` slot at `127.0.0.1:18080`; persistent gateway on 8080 stopped only through `llm-inference.service`; embedding on 8081 remains running.
- Treatment: `GGML_CUDA_DISABLE_FUSION=0` (ON) versus `1` (OFF), verified from `/proc/<PID>/environ` for every block.
- Crossover: 12 fresh-process blocks arranged as six adjacent pairs with alternating order: `OFF-ON | ON-OFF | OFF-ON | ON-OFF | OFF-ON | ON-OFF`.
- Panel: 12 frozen short multilingual/arithmetic prompts. Each pair uses the same deterministic cyclic prompt order; rotation changes between pairs to distribute request-order drift.
- Per block: two excluded warmups followed by 12 recorded requests. Total recorded sample is 144 requests, 72 paired ON/OFF comparisons and six independent process pairs.
- Decode work: `n_predict=64`, `ignore_eos=true`, `temperature=0`, `top_k=1`, `seed=2026082820`, `cache_prompt=false`, slot 0, non-streaming. Every retained request must report exactly 64 predicted tokens.
- Server shape: Qwen3.8 model, context 32768, flash attention, all GPU layers, Q4_0 K/V cache, parallel 1, batch 2048, ubatch 512, 32 context checkpoints, MTP draft maximum 3.
- Warmup and drift control: two fixed 64-token warmups after health per fresh process; adjacent paired blocks; alternating treatment order; GPU temperature, utilization, power and memory captured at block boundaries.

The exact prompt panel is:

1. `Reply with only the result: 17 + 28 =`
2. `Reply with only the result: 9 times 13 =`
3. `Reply with only the result: 144 divided by 12 =`
4. `Reply with only the result: 81 minus 37 =`
5. `Responda apenas com o resultado: 23 + 19 =`
6. `Responda apenas com o resultado: 7 vezes 16 =`
7. `Reply with only the result: the next integer after 399 is`
8. `Reply with only the result: half of 86 is`
9. `Responda apenas com o resultado: 225 dividido por 15 =`
10. `Reply with only the result: 19 squared =`
11. `Responda apenas com o resultado: 1000 menos 457 =`
12. `Reply with only the result: three quarters of 200 =`

## Statistical analysis

- Primary per-request endpoint: server-reported `predicted_per_second`.
- Primary paired ratio: `ON predicted_per_second / OFF predicted_per_second` for the same pair and prompt.
- Secondary wall-throughput ratio: `OFF wall_ms / ON wall_ms`; values above one favor ON.
- Point estimate: geometric mean of the 72 paired ratios.
- Confidence interval: deterministic hierarchical cluster bootstrap with seed `2026082821` and 20,000 replicates. Each replicate resamples the six process pairs with replacement, then resamples the 12 prompts within each selected pair with replacement, and recomputes the geometric mean. The two-sided percentile interval uses 2.5% and 97.5% quantiles.
- No requests are trimmed as outliers. Warmups are excluded before observation. A malformed/failed/non-64-token request fails integrity rather than being silently dropped.

## Acceptance gates

- `binary_model_identity`: `binary_and_model_identity_verified eq True`
- `treatment_identity`: `explicit_fusion_controls_verified eq True`
- `balanced_crossover`: `valid_crossover_blocks eq 12`
- `sample_size`: `recorded_requests eq 144`
- `request_integrity`: `successful_response_rate eq 1.0`
- `fixed_decode_work`: `fixed_decode_token_rate eq 1.0`
- `semantic_parity`: `exact_output_parity_rate eq 1.0`
- `decode_speedup`: `cluster_bootstrap_ratio_ci95_low gt 1.0`
- `wall_non_regression`: `cluster_bootstrap_wall_ratio_ci95_low ge 0.98`
- `service_recovery`: `service_gateway_embedding_restored eq True`

## Abort conditions

- Abort before requests on any binary, CUDA library, model, preregistration or source-artifact identity mismatch.
- Abort if the persistent gateway or embedding service is unhealthy before maintenance, if the temporary port is occupied after the gateway stops, or if a temporary process has the wrong executable, library path or treatment environment.
- Abort a block after three consecutive HTTP failures; retain partial evidence and always execute restoration in `finally`.
- Do not replace a failed or malformed recorded request, remove an outlier, change block order, shorten the 12-block design, or alter bootstrap settings after observing results.
- Always stop temporary units, restart `llm-inference.service`, restore the initial gateway model and verify both 8080 and 8081. Restoration failure blocks every scientific claim.
- The executor stops at `EXECUTED`; independent review is mandatory for promotion or bounded rejection.

## Allowed claims

- `SLX03_GDN_FUSION_DECODE_SPEEDUP_CONFIRMED_R1`
- `SLX03_GDN_FUSION_DECODE_SPEEDUP_NOT_DEMONSTRATED_R1`

Claims about hardware write reduction, production deployment, other models/GPUs, concurrency, long context, or request shapes are forbidden even if a metric looks favorable.
