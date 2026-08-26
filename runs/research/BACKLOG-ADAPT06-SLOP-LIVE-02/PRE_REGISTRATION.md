# BACKLOG-ADAPT06-SLOP-LIVE-02 preregistration

Task: Materialize live multi-adapter routing with two real LoRA checkpoints
Evidence class: `serving_runtime`

## Hypothesis

Replacing only R1's incompatible LoKr artifacts with two real LoRA checkpoints will permit the unchanged conversion/runtime protocol to execute. All ten R1 gates, prompts, route orders, cache controls, and restoration requirements remain binding.

## Frozen inputs

- `runs/research/BACKLOG-ADAPT06-SLOP-LIVE-01/PRE_REGISTRATION.md`
- `runs/research/BACKLOG-ADAPT06-SLOP-LIVE-01/ABORTED.md`
- `tools/research/run_adapt06_slop_live.py`
- `runs/research/BACKLOG-ADAPT-TRAIN-01/raw/checkpoint_seed_20260824/adapter_config.json`
- `runs/research/BACKLOG-ADAPT-TRAIN-01/raw/checkpoint_seed_20260824/adapter_model.safetensors`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/checkpoints/seed_20260824_full_trace/adapter_config.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/checkpoints/seed_20260824_full_trace/adapter_model.safetensors`

- R1 preregistration/abort/implementation: `4da2aaf993703e42bcce71a9cebf5931e5cf8fe8cdd0f940ae021f5311e06885`, `2d64f4178def4793d96cb197be20f229f6bac5a3e06ad14dae204623825772dc`, `850d2a30ce8ae636931644837035261dbf92cc6c43c4ee545a208037294c2586`.
- Behavioral LoRA config/weights: `4872214e344ce266b0b10a1f70db33775224441ff1d78dbe0c7fb4dd70aeac84`, `05b80090d2d1ba751d48a5032cddec82819a79b9bfb5bd8b05306b85d6ef0122`.
- Trace LoRA config/weights: `4872214e344ce266b0b10a1f70db33775224441ff1d78dbe0c7fb4dd70aeac84`, `174832aa1bd25cbc5ed7f0ff717ad253ec94e2c23edc82e6f828ceadeed566b7`.

## Command

```powershell
python tools/research/run_adapt06_slop_live_r2.py --outdir runs/research/BACKLOG-ADAPT06-SLOP-LIVE-02
```

## Factors

- Same 12 prompts, three routes, 36 isolated baselines, 72 alternated route cells, cache sequences, and 30-cell alternating/grouped schedules as R1.
- Only artifact identities change: adapter 0 is the reproduced behavioral LoRA seed 20260824; adapter 1 is the full-trace LoRA seed 20260824. Both target the same Qwen3.5 MLP modules.
- Temporary runtime and fail-safe service restoration are identical to R1.

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

- All R1 abort conditions remain binding. No prompt, scale, route, threshold, order, or service setting may change after observation.

## Allowed claims

- `ADAPT06_LIVE_ISOLATION_QUALIFIED_SLOP_CLIENT_AFFINITY_R2`
- `ADAPT06_SLOP_FALSE_POSITIVE_CONFIRMED_R2`

Claims outside these codes are forbidden even if a metric looks favorable.

No server-native scheduler, fused GEMM, production, internal cache-key, or out-of-panel claim is allowed.
