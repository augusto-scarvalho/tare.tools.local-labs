# BACKLOG-FLEET-REGRESSION-SCREEN-02 preregistration

Task: Replicate the large fleet regression screen with immutable request and terminal-state evidence
Evidence class: `serving_runtime`

## Hypothesis

Across the four qualified text routes, two fresh greedy passes over the frozen 56-case panel will retain 448/448 successful requests, preserve exact semantic projections in at least 95% of paired cases, and restore the initial service route without restart. Failure of any gate yields the negative claim.

## Frozen inputs

- `config/qualified_model_fleet.json`
- `workloads/gsm8k.jsonl`
- `runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl`
- `tools/benchmarks/agent_suite_v2.py`
- `tools/research/run_fleet_regression_screen.py`
- `runs/research/BACKLOG-FLEET-REGRESSION-SCREEN-01/raw/receipt.json`
- `docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md`

- admission: `04d4d41a6aca177eebd5cff44adf0892f6675b07eb665531f2da4d745fbaee03`
- fleet registry: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`
- GSM8K: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`
- protected QA: `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`
- agent suite: `14d0a1b76d4d729228678f215ecefa3254aef214eb65ac9d8d7061bccc0dc59e`
- predecessor runner: `7cbf942375c120970ac78fb40f7f15050ebddb7ec5372cb40bd721009faf1de3`
- predecessor receipt: `d303490b152babcc8f590b0b840fb14c4da02c865430f8398bdca7023a0eeb94`
- audit ledger: `a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04`

## Command

```powershell
python tools/research/run_fleet_regression_screen_r2.py --outdir runs/research/BACKLOG-FLEET-REGRESSION-SCREEN-02
```

## Factors

Four qualified text aliases; 32 GSM8K, 16 protected-QA and eight agent-tool cases; two complete repeats; 448 requests total. Frozen controls are temperature 0, seed 20260826, streaming off, prompt cache off and semantic projections excluding timing/IDs. Execution is sequential through the live gateway on the RTX 3090. Every request payload and response is retained. The initial alias, systemd PID/restart count, embedding health and physical model hashes are checked and restored.

## Acceptance gates

- `route_coverage`: `route_models_completed eq 4`
- `request_coverage`: `recorded_requests eq 448`
- `request_integrity`: `successful_response_rate eq 1.0`
- `route_identity`: `route_identity_verified eq True`
- `repeatability`: `exact_repeat_rate ge 0.95`
- `request_retention`: `retained_request_payloads eq 448`
- `terminal_binding`: `final_runner_state_bound eq True`
- `service_integrity`: `service_restarts eq 0`
- `embedding_integrity`: `embedding_health eq 200`
- `service_recovery`: `initial_model_restored eq True`

## Abort conditions

Abort on any frozen host or WSL artifact mismatch, preregistration mismatch, unhealthy gateway/embedding boundary, fewer than 512 MiB free after a route load, route-identity mismatch, three consecutive request errors, restoration failure, incomplete provenance, missing request payload, or receipt/state binding failure. Partial JSONL progress remains restart evidence but cannot satisfy coverage gates.

## Allowed claims

- `QUALIFIED_TEXT_FLEET_SCREEN_COMPLETE_R2`
- `QUALIFIED_TEXT_FLEET_SCREEN_REJECTED_R2`

Claims outside these codes are forbidden even if a metric looks favorable.
