# BACKLOG-GATEWAY-ROUTE-STRESS-01 preregistration

Task: Stress all qualified gateway routes across repeated cold model switches
Evidence class: `serving_runtime`
Executor: Codex executor
Date: 2026-08-26

## Hypothesis

The single-resident gateway will complete 30 cold switches across all six
qualified aliases without route misbinding, request failure, embedding outage,
service restart or deterministic response drift, and will restore the initial
resident model. Any failed identity or recovery invariant rejects the claim.

## Frozen inputs

- `config/qualified_model_fleet.json`, SHA-256 `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`.
- `tools/serving/qualified_model_gateway.py`, SHA-256 `b4afb885e1b23ed0bc227620f4140038eeb6bcfc2b066d645666ce07e10a3ccb`.
- `docs/HANDOFF_2026-08-26_CONSOLIDATED_RESEARCH_BACKLOG.md`, SHA-256 `895fec3ac345bdf26350b4a97f513bf4f4b3bad9898d09701db07a985f8b7d55`.
- Every model artifact, runtime binary and argv tuple is transitively frozen by
  the fleet registry and verified by the runner before its first switch.

## Command

```powershell
python tools/research/run_gateway_route_stress.py --outdir runs/research/BACKLOG-GATEWAY-ROUTE-STRESS-01
```

## Factors

- Aliases: `qwen38`, `hauhaucs`, `fable-tc`, `qwen36-moe`, `gemma-vision`,
  `muse-vision`.
- Five cycles use a rotated alias order, producing 30 physical route switches.
- Four deterministic text-only probes follow each verified switch: exact token,
  arithmetic, JSON formatting and short context recall; 120 recorded requests.
- Temperature 0, seed 20260826, cache disabled, non-streaming. Vision capability
  is deliberately outside scope.
- Route status, backend PID, GPU state and embedding health are captured at every
  boundary. The gateway service itself must retain PID and restart count.

## Acceptance gates

- `switch_coverage`: `verified_switches eq 30`
- `request_coverage`: `recorded_requests eq 120`
- `request_integrity`: `successful_response_rate eq 1.0`
- `route_identity`: `route_identity_rate eq 1.0`
- `cycle_repeatability`: `exact_cycle_repeat_rate ge 0.95`
- `service_integrity`: `service_restarts eq 0`
- `embedding_integrity`: `embedding_boundary_successes eq 30`
- `service_recovery`: `initial_model_restored eq True`

## Abort conditions

- Any frozen source or registered artifact identity differs.
- Gateway or embedding health is not 200 before the campaign.
- A switch canary returns non-200 or status reports a different current model.
- Three consecutive probe errors occur, GPU free memory falls below 512 MiB, or
  embedding health fails at a route boundary.
- The initial resident route cannot be restored.

## Allowed claims

- `QUALIFIED_GATEWAY_ROUTE_STRESS_PASSED_R1`
- `QUALIFIED_GATEWAY_ROUTE_STRESS_REJECTED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.
The executor stops at `EXECUTED`; independent review is required.
