# BACKLOG-ADAPT06-SLOP-LIVE-04 preregistration

Task: Run multi-adapter audit on matching physical Qwen3.5 base
Evidence class: `serving_runtime`

## Hypothesis

Serving the exact Qwen3.5-0.8B base used by both LoRAs will eliminate R3's tensor mismatch and allow the unchanged physical routing protocol to execute. All ten gates remain binding.

## Frozen inputs

- `runs/research/BACKLOG-ADAPT06-SLOP-LIVE-03/PRE_REGISTRATION.md`
- `runs/research/BACKLOG-ADAPT06-SLOP-LIVE-03/ABORTED.md`
- `tools/research/run_adapt06_slop_live_r3.py`
- `runs/research/BACKLOG-ADAPT-TRAIN-01/raw/checkpoint_seed_20260824/adapter_config.json`
- `runs/research/BACKLOG-ADAPT-TRAIN-01/raw/checkpoint_seed_20260824/adapter_model.safetensors`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/checkpoints/seed_20260824_full_trace/adapter_config.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/checkpoints/seed_20260824_full_trace/adapter_model.safetensors`

- R3 preregistration/abort/implementation: `d84708bea312661ee398b08e078ef57695d49212141c99501195273b74b7ab3a`, `6e85386ad9dea830117fc8385fee1da35629fd65a83eca06721c515d39b1a514`, `8d23d446ef996c56a0a986c062ac493680ada8a86b8ffc917b0a2e1890f16901`.
- Qwen base config/weights/index: `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`, `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`, `ce9a885efdf27d3664fdef5d512ad365216f1074051ef840c7cd8e5431495d0a`.
- HF-to-GGUF converter: `8f1bed9466221e57e434caa7ee720abe1569deb6bc2fe5a65da950ea66c8e737`.
- The two LoRA identities remain exactly those frozen by R3.

## Command

```powershell
python tools/research/run_adapt06_slop_live_r4.py --outdir runs/research/BACKLOG-ADAPT06-SLOP-LIVE-04
```

## Factors

- Before service maintenance, convert `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe` to an F16 GGUF inside this packet and verify all frozen input hashes.
- Repeat the exact R3 panels and schedules. The only runtime deltas are the matching base GGUF and one comma-separated `--lora adapter0,adapter1` argument required by the active binary.

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

- All R3 abort/restoration conditions remain binding. Base conversion, adapter load, route, port handoff, or service restore failure aborts.

## Allowed claims

- `ADAPT06_LIVE_ISOLATION_QUALIFIED_SLOP_CLIENT_AFFINITY_R4`
- `ADAPT06_SLOP_FALSE_POSITIVE_CONFIRMED_R4`

Claims outside these codes are forbidden even if a metric looks favorable.

No server-native scheduler, fused GEMM, production, internal cache-key, or out-of-panel claim is allowed.
