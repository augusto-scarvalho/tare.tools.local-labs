# BACKLOG-ADAPT06-SLOP-LIVE-05 preregistration

Task: Rerun live multi-adapter audit with normalized restoration identity
Evidence class: `serving_runtime`

## Hypothesis

Comparing only immutable systemd executable path and `argv[]` will accept a correctly restored original service while leaving the complete R4 scientific protocol and ten gates unchanged.

## Frozen inputs

- `runs/research/BACKLOG-ADAPT06-SLOP-LIVE-04/PRE_REGISTRATION.md`
- `runs/research/BACKLOG-ADAPT06-SLOP-LIVE-04/ABORTED.md`
- `tools/research/run_adapt06_slop_live_r4.py`
- `runs/research/BACKLOG-ADAPT-TRAIN-01/raw/checkpoint_seed_20260824/adapter_config.json`
- `runs/research/BACKLOG-ADAPT-TRAIN-01/raw/checkpoint_seed_20260824/adapter_model.safetensors`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/checkpoints/seed_20260824_full_trace/adapter_config.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/checkpoints/seed_20260824_full_trace/adapter_model.safetensors`

- R4 preregistration/abort/implementation: `93dc07c53d8742e3b2b9f8cafc2ed96c989dc8e6da48e9b85341407af45a3b16`, `68d9b58e5daab2df6809ba4816bf377b4c6e96fa115a6d1b0f4faa944b4573b8`, `1ea84a9e1d256efc6bd7edd0f34ceca4464c3f022dd941eb3bb68e347c454f25`.
- The Qwen base and two LoRA input identities remain exactly those frozen by R4.

## Command

```powershell
python tools/research/run_adapt06_slop_live_r5.py --outdir runs/research/BACKLOG-ADAPT06-SLOP-LIVE-05
```

## Factors

- Identical to R4: same base conversion, two LoRAs, 12 prompts, 36 baselines, 72 routed cells, cache sequences, scheduling orders, and root service handoff.
- Sole delta: normalize systemd `ExecStart` to its substring through the terminating `argv[]` command, excluding volatile start/stop time, PID, code and status fields.

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

- All R4 abort conditions remain binding. Normalized path/argv mismatch, health failure, or restore failure aborts.

## Allowed claims

- `ADAPT06_LIVE_ISOLATION_QUALIFIED_SLOP_CLIENT_AFFINITY_R5`
- `ADAPT06_SLOP_FALSE_POSITIVE_CONFIRMED_R5`

Claims outside these codes are forbidden even if a metric looks favorable.

No server-native scheduler, fused GEMM, production, internal cache-key, or out-of-panel claim is allowed.
