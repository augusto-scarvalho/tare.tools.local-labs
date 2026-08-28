# BACKLOG-FLEET-SEEDED-STABILITY-01 preregistration

Task: Measure fixed-seed sampling stability across the qualified text fleet
Evidence class: `serving_runtime`
Executor: Codex executor
Date: 2026-08-26

## Hypothesis

With temperature 0.2 and a fixed seed, at least 90% of semantic responses will
remain identical across three complete passes for each qualified text alias.

## Frozen inputs

- Fleet registry SHA-256 `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`.
- GSM8K source SHA-256 `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`.
- QA source SHA-256 `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`.
- Fleet harness SHA-256 `7cbf942375c120970ac78fb40f7f15050ebddb7ec5372cb40bd721009faf1de3`.

## Command

```powershell
python tools/research/run_fleet_seeded_stability.py --outdir runs/research/BACKLOG-FLEET-SEEDED-STABILITY-01
```

## Factors

- Four aliases: qwen38, hauhaucs, fable-tc and qwen36-moe.
- Three passes of 16 math plus eight protected QA cases: 288 requests.
- Temperature 0.2, top-p 0.95, fixed seed 20260826, cache disabled.
- Semantic comparison excludes timing, request IDs and usage counters.
- Route identity, service restart count, embedding health and initial route are
  invariants; quality is descriptive only.

## Acceptance gates

- `route_coverage`: `routes_completed eq 4`
- `request_coverage`: `recorded_requests eq 288`
- `request_integrity`: `successful_response_rate eq 1.0`
- `seeded_stability`: `exact_seeded_repeat_rate ge 0.9`
- `service_integrity`: `service_restarts eq 0`
- `service_recovery`: `initial_model_restored eq True`

## Abort conditions

- Frozen input or artifact identity mismatch.
- Gateway/embedding unhealthy, route misbinding or three consecutive errors.
- GPU free memory below 512 MiB or initial route cannot be restored.

## Allowed claims

- `QUALIFIED_TEXT_FLEET_SEEDED_STABLE_R1`
- `QUALIFIED_TEXT_FLEET_SEEDED_UNSTABLE_R1`

Claims outside these codes are forbidden even if a metric looks favorable.
The executor stops at `EXECUTED`; independent review is required.
