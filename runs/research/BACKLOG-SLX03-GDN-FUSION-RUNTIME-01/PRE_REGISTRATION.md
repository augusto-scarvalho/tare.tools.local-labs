# BACKLOG-SLX03-GDN-FUSION-RUNTIME-01 preregistration

Task: Observe SLX-03 GDN fusion in live Qwen3.8 runtime with an explicit OFF control
Evidence class: `serving_runtime`

## Hypothesis

On the frozen Qwen3.8 27B Q4_K_XL artifact, the promoted instrumented build emits the exact GDN fused-cache marker only when `GGML_CUDA_DISABLE_FUSION=0`, never when it is `1`, while producing byte-identical deterministic outputs across a balanced OFF-ON-ON-OFF crossover.

## Frozen inputs

- `runs/research/BACKLOG-SLX03-GDN-FUSION-INSTRUMENTED-01/raw/receipt.json`
- `runs/research/BACKLOG-SLX03-GDN-FUSION-INSTRUMENTED-01/REVIEW.json`
- `config/qualified_model_fleet.json`

The runner freezes this admission/preregistration, the promoted instrumented-build receipt/review and `config/qualified_model_fleet.json`. It independently verifies WSL SHA-256 and dereferenced sizes for the instrumented server, CUDA library and Qwen3.8 GGUF before stopping the persistent gateway.

## Command

```powershell
python tools/research/run_slx03_gdn_fusion_runtime.py
```

## Factors

Four fresh systemd transient blocks in order OFF-ON-ON-OFF, each with two discarded warmups and eight frozen deterministic prompts (`temperature=0`, `top_k=1`, seed `2026082816`, `n_predict=64`). All other server arguments are identical, including Q4_0 KV, one slot, SM86 CUDA build and MTP draft depth 3. The effective environment is read from `/proc/<pid>/environ`; journal marker counts are captured per block. Pair 0 compares blocks 1/2 and pair 1 blocks 4/3.

## Acceptance gates

- `binary_model_identity`: `binary_and_model_identity_verified eq True`
- `treatment_identity`: `explicit_fusion_controls_verified eq True`
- `balanced_crossover`: `valid_abba_blocks eq 4`
- `request_integrity`: `successful_response_rate eq 1.0`
- `runtime_route`: `on_blocks_with_marker eq 2`
- `negative_control`: `off_blocks_without_marker eq 2`
- `semantic_parity`: `exact_output_parity_rate eq 1.0`
- `service_recovery`: `service_gateway_embedding_restored eq True`

## Abort conditions

Abort on identity mismatch, occupied temporary port, unhealthy baseline/embedding, three consecutive request failures, incorrect treatment environment, missing ON marker, any OFF marker, output mismatch, incomplete block, or provenance failure. Stop transient units in `finally`, restart `llm-inference.service`, restore the initial gateway alias and verify embedding. Do not stop 8081. No performance/write-reduction/deployment claim is allowed.

## Allowed claims

- `SLX03_GDN_FUSION_RUNTIME_ROUTE_CONFIRMED_R1`
- `SLX03_GDN_FUSION_RUNTIME_ROUTE_NOT_CONFIRMED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.
