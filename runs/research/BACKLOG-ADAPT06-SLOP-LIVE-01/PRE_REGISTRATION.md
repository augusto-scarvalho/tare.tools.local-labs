# BACKLOG-ADAPT06-SLOP-LIVE-01 preregistration

Task: Materialize ADAPT-06 and SLOP multi-adapter routing in live serving
Evidence class: `serving_runtime`

## Hypothesis

Two real PEFT adapters converted to GGUF and loaded simultaneously will produce materially distinct behavior on at least 4/12 frozen prompts while 100% of 72 alternated per-request routes reproduce their isolated route baselines, including after same-slot cache reuse. Client-side affinity grouping will reduce requested route changes by at least 90% without changing outputs. Any isolation, cache, restoration, or parity failure rejects the historical promotion.

## Frozen inputs

- `runs/research/ADAPT-06-ADAPTER-CACHE-TAGGING-2026-08-25/PRE_REGISTRATION.md`
- `runs/research/ADAPT-06-ADAPTER-CACHE-TAGGING-2026-08-25/RESULT.md`
- `runs/research/ADAPT-06-ADAPTER-CACHE-TAGGING-2026-08-25/raw/receipt.json`
- `tools/analysis/adapter_cache_tagger.py`
- `runs/research/SLOP-L1-L7-MULTI-ADAPTER-2026-08-25/PRE_REGISTRATION.md`
- `runs/research/SLOP-L1-L7-MULTI-ADAPTER-2026-08-25/RESULT.md`
- `runs/research/SLOP-L1-L7-MULTI-ADAPTER-2026-08-25/raw/receipt.json`
- `tools/analysis/multi_adapter_router.py`
- `runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw/target_mlp_only/adapter/adapter_config.json`
- `runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw/target_mlp_only/adapter/adapter_model.safetensors`
- `runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw/target_attn_only/adapter/adapter_config.json`
- `runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw/target_attn_only/adapter/adapter_model.safetensors`

- ADAPT-06 prereg/result/receipt/tagger: `9551d6ba2f481ae11490159b530d305887463bd49ae160cf781cfbcb00fd244c`, `3ef1e46397b285a5ebaddd6924c383c6462de4d6cf03c1d8277292419653122a`, `f00dd6d31aa4d8970ef77aad7ccbfa68ca23bc31d26caf3f2bf8ca5e43665bb9`, `951395b9210ce86771cc4982c94ecd31db7641c0e5c2a53c47aabc7539765369`.
- SLOP prereg/result/receipt/router: `3c321c0a568ac4451de6cd5f998005e8e20e2fbef816df420637c245df956239`, `75f76dad7122e124bb115998e08155bfcb13eb002ab6f1471160e5bd0f901841`, `60d6f9e0f1a189a663a44ca6cc1979444b0b4653b5c2e708994311ebe287dd8d`, `1b4e6abf5d88ea26293d0ff55320631236e57de54472e74d6bfb52532b57a9b4`.
- MLP adapter config/weights: `45067f22d87e53ba56114cd0126c20d0591cefc5c9261a1de6c83b705f56e784`, `3fda4d2bae7c6388e97fc69c3c2e4de5d85a614e99f436d8c04373ced3b38966`.
- Attention adapter config/weights: `8516576a6d6a79f13bddd2388483d0638804d3367f661041be9c1b99aa0008fd`, `839d777b848ec202349266fc271aaacd4eff3078bb68e8a46f341dbc9b3194eb`.
- Active binary/model/converter/base-config identities are captured before conversion and must remain stable.

## Command

```powershell
python tools/research/run_adapt06_slop_live.py --outdir runs/research/BACKLOG-ADAPT06-SLOP-LIVE-01
```

## Factors

- Convert `target_mlp_only` and `target_attn_only` with the local llama.cpp converter and the cached Qwen3.5-0.8B config, output F16 GGUF under this packet.
- Stop `llm-inference.service`, launch the same frozen model/binary/arguments plus both LoRAs and `--lora-init-without-apply`, and require four idle slots.
- Freeze twelve deterministic arithmetic prompts. Establish isolated baselines for base, MLP and attention routes with `cache_prompt=false`.
- Run two alternated repetitions of all 36 route/prompt cells using explicit per-request `lora` scales. Every output must exactly match the corresponding isolated baseline; at least four prompts must distinguish routes.
- On fixed slots, execute same-route repeat and cross-route return sequences with long common prefixes and `cache_prompt=true`; require cache hits and baseline-consistent outputs.
- Run the same 30 route/prompt cells in alternating and route-grouped order; compare output maps exactly and count requested route changes. This qualifies client-side affinity only, not an internal scheduler.
- In `finally`, terminate the temporary server, restart the original systemd service, and verify original PID is replaced by a healthy service with its exact original `ExecStart`; port 8081 must stay healthy throughout.

## Acceptance gates

- `adapter_conversion`: `converted_adapters eq 2`
- `adapter_loading`: `loaded_adapters eq 2`
- `behavioral_materiality`: `prompts_with_distinct_route_outputs ge 4`
- `route_isolation`: `routed_exact_match_rate eq 1.0`
- `cross_route_isolation`: `cross_route_contamination_count eq 0`
- `cache_reuse`: `same_route_cache_hit_rate ge 0.75`
- `affinity_switch_reduction`: `requested_route_switch_reduction ge 0.9`
- `affinity_parity`: `schedule_semantic_parity eq 1.0`
- `service_restore`: `original_service_restored eq 1`
- `embedding_integrity`: `embedding_health eq 200`

## Abort conditions

- Hash/conversion/load failure, fewer than four idle slots, any request error, output nondeterminism during baseline, model/binary drift, embedding failure, or inability to restore the original service aborts.
- No prompt, adapter, scale, sample, threshold, or scheduling order may change after outputs are observed.

## Allowed claims

- `ADAPT06_LIVE_ISOLATION_QUALIFIED_SLOP_CLIENT_AFFINITY_R1`
- `ADAPT06_SLOP_FALSE_POSITIVE_CONFIRMED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.

No server-native affinity scheduler, fused GEMM, production deployment, internal cache-key identity, or out-of-panel claim is permitted.
