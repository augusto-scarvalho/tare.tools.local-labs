# BACKLOG-FLEET-CONTEXT-ENVELOPE-03 preregistration

Task: Complete per-slot context retrieval through the verified backend tokenizer endpoint
Evidence class: `serving_runtime`

## Hypothesis

Each qualified text route will recover at least 90% of exact access-code
needles across its configured per-slot context envelope, with no position
bucket below 80%. The R2 scientific matrix is unchanged; only tokenizer calls
move from the non-proxying gateway path to the active backend `/tokenize` path.

## Frozen inputs

- `runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-02/PRE_REGISTRATION.md`
- `runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-02/PIPELINE.json`
- `tools/research/run_fleet_context_envelope.py`
- `runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-02/runner.stderr.log`
- `runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-02/raw/recovery_state.json`
- `runs/autonomous/EXPERIMENT-WATCH-2026-08-27-FLEET-CONTEXT-R2/FINAL.json`
- `config/qualified_model_fleet.json`

- Admission: `439ff053d46c1f0099300ec695d7ce23cd5b413926b40da70593f0ba153904d0`.
- R2 preregistration: `e851482e89abd1bc417db405921a1b0d4a4195f17452689eebb41b6f23e845ae`.
- R2 blocked pipeline: `5d7ee0aef3b9ed30ef6b15fc1cbca9257d8d23bf01b83932e35b2206ffaf37a6`.
- R2 runner: `1ebb0c07145edd48f1fcc7d8f97b248a954ce4faa554e4d6f220c6d693eb857b`.
- R2 stderr: `54640b924eb5820fd719db98ab60b2de9920771e81c8f2681058152ca2aff25a`.
- R2 recovery: `23cc4f5f154dc1f0fbd215c9acde15e710388a46e5cc6afe43930507d91f12ab`.
- R2 watcher final: `3bbdb733bf8e9930b5d1086467de452869bc7298e06ad9084174d4df20fa6426`.
- Fleet registry: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`.

## Command

```powershell
python tools/research/run_fleet_context_envelope_r3.py --outdir runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03
```

## Factors

- Exactly the R2 72-request matrix: 4 routes x 3 per-slot context targets x 3
  needle positions x 2 deterministic codes.
- Qwen3.8/HauhauCS targets `4k/16k/28k`; Fable `2k/6k/7.6k`;
  Qwen3.6-MoE `4k/12k/17k` under its 18432-token per-slot limit.
- The sole implementation change is `/tokenize` at the active backend port
  reported by `/fleet/status`; generation remains through gateway port 8080.
- Temperature 0, top-k 1, seed 20260827, prompt cache off, thinking off and
  32-token output cap remain frozen.

## Acceptance gates

- `artifact_identity`: `verified_model_artifacts eq 4`
- `request_coverage`: `recorded_requests eq 72`
- `request_integrity`: `successful_response_rate eq 1.0`
- `context_fit`: `requests_within_route_slot_context eq 72`
- `qwen38_recall`: `qwen38_exact_recall ge 0.9`
- `hauhaucs_recall`: `hauhaucs_exact_recall ge 0.9`
- `fable_recall`: `fable_tc_exact_recall ge 0.9`
- `qwen36_moe_recall`: `qwen36_moe_exact_recall ge 0.9`
- `position_robustness`: `minimum_position_bucket_recall ge 0.8`
- `service_recovery`: `initial_route_and_services_restored eq True`

## Abort conditions

- Abort on frozen identity mismatch, backend-port mismatch, tokenizer failure,
  context overflow, three consecutive request failures, unhealthy embedding,
  or inability to restore the initial route and services.
- Retrieval misses remain evidence and do not abort.

## Allowed claims

- `QUALIFIED_TEXT_FLEET_SLOT_CONTEXT_ENVELOPES_MEASURED_R3`
- `QUALIFIED_TEXT_FLEET_SLOT_CONTEXT_ENVELOPES_NOT_CONFIRMED_R3`

Claims outside these codes are forbidden even if a metric looks favorable.
