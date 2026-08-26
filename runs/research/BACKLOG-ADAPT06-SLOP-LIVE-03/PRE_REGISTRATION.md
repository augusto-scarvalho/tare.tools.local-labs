# BACKLOG-ADAPT06-SLOP-LIVE-03 preregistration

Task: Run live multi-adapter audit with privileged verified service handoff
Evidence class: `serving_runtime`

## Hypothesis

Using WSL root for the systemd handoff and requiring 8080 to become unavailable will route the unchanged R2 protocol to the temporary two-LoRA server. All ten R2 gates and scientific factors remain binding.

## Frozen inputs

- `runs/research/BACKLOG-ADAPT06-SLOP-LIVE-02/PRE_REGISTRATION.md`
- `runs/research/BACKLOG-ADAPT06-SLOP-LIVE-02/ABORTED.md`
- `tools/research/run_adapt06_slop_live_r2.py`
- `runs/research/BACKLOG-ADAPT-TRAIN-01/raw/checkpoint_seed_20260824/adapter_config.json`
- `runs/research/BACKLOG-ADAPT-TRAIN-01/raw/checkpoint_seed_20260824/adapter_model.safetensors`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/checkpoints/seed_20260824_full_trace/adapter_config.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/checkpoints/seed_20260824_full_trace/adapter_model.safetensors`

- R2 preregistration/abort/implementation: `0b5135d8f3c2cd26bed6b88ea1ea1d251ed0903aaea113caa609a084d97ff254`, `efa3ccec36f4eb99392a6e7dac4896d78d2699079fcec760bfccbd695a301314`, `2242b3e5e424297739828b115c53ed2f36e2c452218dc059f6eb57bfe718f860`.
- Behavioral LoRA config/weights: `4872214e344ce266b0b10a1f70db33775224441ff1d78dbe0c7fb4dd70aeac84`, `05b80090d2d1ba751d48a5032cddec82819a79b9bfb5bd8b05306b85d6ef0122`.
- Trace LoRA config/weights: `4872214e344ce266b0b10a1f70db33775224441ff1d78dbe0c7fb4dd70aeac84`, `174832aa1bd25cbc5ed7f0ff717ad253ec94e2c23edc82e6f828ceadeed566b7`.

## Command

```powershell
python tools/research/run_adapt06_slop_live_r3.py --outdir runs/research/BACKLOG-ADAPT06-SLOP-LIVE-03
```

## Factors

- Identical to R2: same two LoRAs, conversion, 12 prompts, 36 baselines, 72 alternated cells, cache sequences, and paired scheduling orders.
- Sole delta: invoke systemd stop/start through `wsl -u root` and abort unless 8080 is actually unavailable before launching the temporary process.

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

- Every R2 abort condition remains binding. Any failure to stop or restore the exact original unit aborts; 8081 must remain HTTP 200 throughout.

## Allowed claims

- `ADAPT06_LIVE_ISOLATION_QUALIFIED_SLOP_CLIENT_AFFINITY_R3`
- `ADAPT06_SLOP_FALSE_POSITIVE_CONFIRMED_R3`

Claims outside these codes are forbidden even if a metric looks favorable.

No server-native scheduler, fused GEMM, production, internal cache-key, or out-of-panel claim is allowed.
