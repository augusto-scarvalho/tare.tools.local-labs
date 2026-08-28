# BACKLOG-FLEET-CONTEXT-ENVELOPE-01 preregistration

Task: Measure bounded long-context retrieval envelopes for every qualified text route
Evidence class: `serving_runtime`

## Hypothesis

Each qualified text route will recover at least 90% of exact access-code
needles across its own configured context envelope, with no start/middle/end
position bucket below 80%. This is a bounded retrieval test, not a claim about
general long-context reasoning or production RAG quality.

## Frozen inputs

- `config/qualified_model_fleet.json`
- `tools/research/run_fleet_regression_screen.py`
- `runs/research/BACKLOG-GATEWAY-ROUTE-STRESS-01/raw/receipt.json`

- Admission: `05f75bc1d0d279291f62bcef9425df2d9b2ee1ea6cb8712eb9519fc4e685b405`.
- Fleet registry: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`.
- Fleet runtime helper: `7cbf942375c120970ac78fb40f7f15050ebddb7ec5372cb40bd721009faf1de3`.
- Gateway stress receipt: `527e308b2aa54fe96bb641f1d5380b04b42e7871245d173ee107cec0dabbfe41`.
- Every physical GGUF is rehashed against the fleet registry before generation.

## Command

```powershell
python tools/research/run_fleet_context_envelope.py --outdir runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-01
```

## Factors

- Four routes, 18 cases each: three route-specific token targets, three needle
  positions and two independent deterministic codes per cell (72 requests).
- Targets: Qwen3.8 and HauhauCS `4k/16k/28k`; Fable-TC `2k/6k/7.6k` within
  its 8192 profile; Qwen3.6-MoE `8k/32k/64k` within its 73728 profile.
- The active route tokenizer binary-searches the deterministic filler count;
  final prompt token counts and context-fit margins are recorded physically.
- Filler records, needle codes, placement, instruction, temperature 0, top-k 1,
  seed 20260827, cache disabled and 32-token response cap are frozen in code.
- Exact recall requires returning the unique expected code. Results are grouped
  by model, target length and start/middle/end position; timing is descriptive.

## Acceptance gates

- `artifact_identity`: `verified_model_artifacts eq 4`
- `request_coverage`: `recorded_requests eq 72`
- `request_integrity`: `successful_response_rate eq 1.0`
- `context_fit`: `requests_within_route_context eq 72`
- `qwen38_recall`: `qwen38_exact_recall ge 0.9`
- `hauhaucs_recall`: `hauhaucs_exact_recall ge 0.9`
- `fable_recall`: `fable_tc_exact_recall ge 0.9`
- `qwen36_moe_recall`: `qwen36_moe_exact_recall ge 0.9`
- `position_robustness`: `minimum_position_bucket_recall ge 0.8`
- `service_recovery`: `initial_route_and_services_restored eq True`

## Abort conditions

- Abort on source/artifact mismatch, unhealthy gateway or embedding service,
  route identity mismatch, tokenizer failure, any prompt exceeding the route's
  configured context minus 64 tokens, three consecutive request failures, or
  inability to restore the initial route and service health.
- Retrieval misses are evidence and do not abort remaining safe cases.

## Allowed claims

- `QUALIFIED_TEXT_FLEET_CONTEXT_ENVELOPES_MEASURED_R1`
- `QUALIFIED_TEXT_FLEET_CONTEXT_ENVELOPES_NOT_CONFIRMED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.
