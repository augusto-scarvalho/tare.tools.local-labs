# BACKLOG-SLX03-GDN-FUSION-RUNTIME-03 preregistration

Task: Finalize SLX-03 runtime route observation with PID-bound startup capture
Evidence class: `serving_runtime`

## Hypothesis

With trace verbosity captured from each current process before warmups, the promoted instrumented build emits the exact GDN fused-cache marker only for `GGML_CUDA_DISABLE_FUSION=0`, never for `1`, while retaining byte-identical outputs across the frozen OFF-ON-ON-OFF crossover.

## Frozen inputs

- `runs/research/BACKLOG-SLX03-GDN-FUSION-RUNTIME-02/raw/receipt.json`
- `runs/research/BACKLOG-SLX03-GDN-FUSION-RUNTIME-02/REVIEW.json`
- `runs/research/BACKLOG-SLX03-GDN-FUSION-INSTRUMENTED-01/raw/receipt.json`
- `runs/research/BACKLOG-SLX03-GDN-FUSION-INSTRUMENTED-01/REVIEW.json`
- `config/qualified_model_fleet.json`

The runner freezes this admission/preregistration, R2 receipt/review, promoted instrumented receipt/review and fleet config. It verifies the same WSL model/binary/library hashes as R2. R3 changes only evidence collection: startup is queried by current PID before warmups; request journals are queried by that PID after requests without a tail cap.

## Command

```powershell
python tools/research/run_slx03_gdn_fusion_runtime_r3.py
```

## Factors

Identical R2 ABBA order, prompts, seed, generation parameters, model, binary, cache, MTP and `-lv 4`. For each block, the runner seals a PID-bound startup journal before warmups and requires `verbosity = 4`; it then captures the full PID-bound request journal separately. Four startup and four request logs are required. Artifact ledgers are exact byte SHA-256.

## Acceptance gates

- `binary_model_identity`: `binary_and_model_identity_verified eq True`
- `trace_logging`: `pid_bound_startup_verbosity_verified eq True`
- `treatment_identity`: `explicit_fusion_controls_verified eq True`
- `balanced_crossover`: `valid_abba_blocks eq 4`
- `request_integrity`: `successful_response_rate eq 1.0`
- `runtime_route`: `on_blocks_with_marker eq 2`
- `negative_control`: `off_blocks_without_marker eq 2`
- `semantic_parity`: `exact_output_parity_rate eq 1.0`
- `service_recovery`: `service_gateway_embedding_restored eq True`

## Abort conditions

Abort before requests if current-PID startup capture lacks `verbosity = 4`, treatment env is wrong, or identity/health fails. Abort on three consecutive HTTP failures, incomplete provenance, or failed restoration. Scientific gate failures still produce a receipt. Stop transient units in `finally`; restore the initial gateway and never stop 8081. No performance, write-reduction or deployment claim is allowed.

## Allowed claims

- `SLX03_GDN_FUSION_RUNTIME_ROUTE_CONFIRMED_R3`
- `SLX03_GDN_FUSION_RUNTIME_ROUTE_NOT_CONFIRMED_R3`

Claims outside these codes are forbidden even if a metric looks favorable.
