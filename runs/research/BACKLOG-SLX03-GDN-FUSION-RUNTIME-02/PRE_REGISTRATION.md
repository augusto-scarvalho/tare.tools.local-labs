# BACKLOG-SLX03-GDN-FUSION-RUNTIME-02 preregistration

Task: Repeat SLX-03 runtime route observation at trace verbosity
Evidence class: `serving_runtime`

## Hypothesis

With server verbosity explicitly set to trace (`-lv 4`), the promoted instrumented build emits the exact GDN fused-cache marker only when `GGML_CUDA_DISABLE_FUSION=0`, never when it is `1`, while retaining byte-identical outputs across the same frozen OFF-ON-ON-OFF crossover.

## Frozen inputs

- `runs/research/BACKLOG-SLX03-GDN-FUSION-RUNTIME-01/raw/receipt.json`
- `runs/research/BACKLOG-SLX03-GDN-FUSION-RUNTIME-01/REVIEW.json`
- `runs/research/BACKLOG-SLX03-GDN-FUSION-INSTRUMENTED-01/raw/receipt.json`
- `runs/research/BACKLOG-SLX03-GDN-FUSION-INSTRUMENTED-01/REVIEW.json`
- `config/qualified_model_fleet.json`

The runner freezes this admission/preregistration, R1 receipt/review, promoted instrumented receipt/review and fleet config. It reuses the exact R1 model, binary, library, prompts, seed, order and server configuration, changing only `-lv 4`. WSL identities are independently checked before maintenance.

## Command

```powershell
python tools/research/run_slx03_gdn_fusion_runtime_r2.py
```

## Factors

Four fresh blocks OFF-ON-ON-OFF, two discarded warmups and eight fixed prompts per block, seed `2026082816`, `temperature=0`, `top_k=1`, `n_predict=64`. The treatment variable remains only `GGML_CUDA_DISABLE_FUSION`; `-lv 4` is common to every block. Every journal must contain the startup declaration `verbosity = 4`, serving as a positive logging-channel control. Artifact ledgers use exact file SHA-256 values rather than ambiguous semantic reserialization hashes.

## Acceptance gates

- `binary_model_identity`: `binary_and_model_identity_verified eq True`
- `trace_logging`: `trace_verbosity_verified eq True`
- `treatment_identity`: `explicit_fusion_controls_verified eq True`
- `balanced_crossover`: `valid_abba_blocks eq 4`
- `request_integrity`: `successful_response_rate eq 1.0`
- `runtime_route`: `on_blocks_with_marker eq 2`
- `negative_control`: `off_blocks_without_marker eq 2`
- `semantic_parity`: `exact_output_parity_rate eq 1.0`
- `service_recovery`: `service_gateway_embedding_restored eq True`

## Abort conditions

Abort on identity mismatch, occupied port, unhealthy baseline/embedding, incorrect process environment, absent trace-verbosity declaration, three consecutive HTTP failures, incomplete provenance, or failed restoration. Scientific gate failures still produce a receipt. Stop transient units in `finally`, restore the initial gateway alias and never stop 8081. No performance, write-reduction or deployment claim is allowed.

## Allowed claims

- `SLX03_GDN_FUSION_RUNTIME_ROUTE_CONFIRMED_R2`
- `SLX03_GDN_FUSION_RUNTIME_ROUTE_NOT_CONFIRMED_R2`

Claims outside these codes are forbidden even if a metric looks favorable.
