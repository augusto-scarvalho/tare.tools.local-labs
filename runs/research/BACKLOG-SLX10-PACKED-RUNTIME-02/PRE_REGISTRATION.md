# BACKLOG-SLX10-PACKED-RUNTIME-02 preregistration

Task: Materialize supported Q2_K packed GGUF and live runtime
Evidence class: `serving_runtime`

## Hypothesis

A supported immutable Q2_K GGUF will be at most 25% of F16 size, reduce net live VRAM at least 50%, retain at least 95% throughput, lose at most 2pp frozen-panel accuracy, and preserve at least 80% exact outputs. Failure rejects low-bit physical qualification at this tested scale.

## Frozen inputs

- `runs/research/BACKLOG-SLX10-PACKED-RUNTIME-01/PRE_REGISTRATION.md`
- `runs/research/BACKLOG-SLX10-PACKED-RUNTIME-01/ABORTED.md`
- `tools/research/run_slx10_packed_runtime.py`
- `runs/research/BACKLOG-ADAPT06-SLOP-LIVE-05/raw/qwen3.5-0.8b-base-f16.gguf`

- R1 preregistration/abort/implementation: `f1fdf716671ea11108c7628d5f12984c00c2231c4f2a64d6e81fcd039b12afcf`, `a7ccdd1a4cf152dfcc207cab7eb954ba94281b25c119713abd38d653e6057ff7`, `f8561917bd2172ad5d003638f3cf7996b1411b694c022e6a28b67ba02956fe87`.
- Physical F16 source GGUF remains SHA-256 `514133770c0e30367721334fb86a76a8647bf8ab4d51fedc62980ce86dda1ac1`.

## Command

```powershell
python tools/research/run_slx10_packed_runtime_r2.py --outdir runs/research/BACKLOG-SLX10-PACKED-RUNTIME-02
```

## Factors

- Identical physical F16 baseline, frozen 32-prompt panel, serving settings, metrics and root service handoff as R1.
- Sole codec delta is deployed quantizer type `Q2_K`; the packed file gate is explicitly 25% rather than IQ2's 18%.

## Acceptance gates

- `packed_artifact`: `q2k_file_ratio le 0.25`
- `physical_load`: `loaded_arms eq 2`
- `memory_reduction`: `vram_reduction ge 0.5`
- `throughput`: `throughput_ratio ge 0.95`
- `quality`: `accuracy_regression le 0.02`
- `semantic_stability`: `exact_output_rate ge 0.8`
- `service_restore`: `original_service_restored eq 1`
- `embedding_integrity`: `embedding_health eq 200`

## Abort conditions

- All R1 runtime, hash, panel, request, VRAM and restoration abort conditions remain binding. No importance matrix is synthesized.

## Allowed claims

- `SLX10_Q2K_PHYSICAL_QUALIFIED_R2`
- `SLX10_FALSE_POSITIVE_CONFIRMED_R2`

Claims outside these codes are forbidden even if a metric looks favorable.

No IQ2_XXS, 27B/35B, production, or out-of-panel quality claim is allowed.
