# BACKLOG-FLEET-CONTEXT-ENVELOPE-02 preregistration

Task: Measure bounded per-slot context retrieval envelopes for every qualified text route
Evidence class: `serving_runtime`

## Hypothesis

Each qualified text route will recover at least 90% of exact access-code
needles across its configured per-slot context envelope, with no
start/middle/end position bucket below 80%. This is bounded retrieval, not a
claim about general long-context reasoning or production RAG quality.

## Frozen inputs

- `runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-01/PRE_REGISTRATION.md`
- `runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-01/PIPELINE.json`
- `config/qualified_model_fleet.json`
- `tools/research/run_fleet_regression_screen.py`
- `runs/research/BACKLOG-GATEWAY-ROUTE-STRESS-01/raw/receipt.json`

- Admission: `3bee678e36d8016e782a586ec3b10b5f88dafb167ee1b2d48005a3a4789543b3`.
- Blocked R1 preregistration: `58496fcb6d374d2f08cc945404c1ed7cd1f68a067df70a90be3f9bc6fe3ae642`.
- Blocked R1 pipeline record: `a8faf0c57dc74a4b46a074b73dedefbc737e0acfd365c07d54febaeec4b5caa8`.
- Fleet registry: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`.
- Fleet helper: `7cbf942375c120970ac78fb40f7f15050ebddb7ec5372cb40bd721009faf1de3`.
- Gateway stress receipt: `527e308b2aa54fe96bb641f1d5380b04b42e7871245d173ee107cec0dabbfe41`.
- Every physical GGUF is rehashed against the registry before generation.

## Command

```powershell
python tools/research/run_fleet_context_envelope.py --outdir runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-02
```

## Factors

- Four routes, 18 cases each: three route-specific token targets, three needle
  positions and two deterministic codes per cell (72 physical requests).
- Per-slot targets: Qwen3.8 and HauhauCS `4k/16k/28k`; Fable-TC
  `2k/6k/7.6k`; Qwen3.6-MoE `4k/12k/17k` because its aggregate 73728-token
  context is divided across four configured slots (18432 per slot).
- The active route tokenizer binary-searches deterministic filler count; final
  tokenizer and server `prompt_n` counts plus context margins are recorded.
- Filler, codes, placement, instruction, temperature 0, top-k 1, seed
  20260827, disabled prompt cache and 32-token output cap are frozen in code.
- Exact recall requires returning only the expected code. Results are grouped
  by route, target and position; timing is descriptive only.

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

- Abort on source/artifact mismatch, unhealthy gateway or embedding service,
  route identity mismatch, tokenizer failure, any prompt exceeding per-slot
  context minus 64 tokens, three consecutive request failures, or inability to
  restore the initial route and services.
- Retrieval misses are evidence and do not abort remaining safe cases.

## Allowed claims

- `QUALIFIED_TEXT_FLEET_SLOT_CONTEXT_ENVELOPES_MEASURED_R2`
- `QUALIFIED_TEXT_FLEET_SLOT_CONTEXT_ENVELOPES_NOT_CONFIRMED_R2`

Claims outside these codes are forbidden even if a metric looks favorable.
