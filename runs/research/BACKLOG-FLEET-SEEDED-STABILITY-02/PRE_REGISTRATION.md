# BACKLOG-FLEET-SEEDED-STABILITY-02 preregistration

Task: Replicate seeded fleet stability with retained requests and immutable terminal state
Evidence class: `serving_runtime`

## Hypothesis

Across the four qualified text routes, three fresh passes with identical seed 20260826, temperature 0.2 and top-p 0.95 will produce identical semantic projections in at least 90% of 192 repeat-to-baseline pairs while all 288 requests succeed and the initial route is restored.

## Frozen inputs

- `config/qualified_model_fleet.json`
- `workloads/gsm8k.jsonl`
- `runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl`
- `tools/research/run_fleet_regression_screen.py`
- `tools/research/run_fleet_seeded_stability.py`
- `runs/research/BACKLOG-FLEET-SEEDED-STABILITY-01/raw/receipt.json`
- `docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md`

- admission: `0c407d66472c63c2a9b78cdd071f85eda7e291a91aea9bbb8e6de8e62713eea4`
- fleet registry: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`
- GSM8K: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`
- protected QA: `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`
- fleet base runner: `7cbf942375c120970ac78fb40f7f15050ebddb7ec5372cb40bd721009faf1de3`
- predecessor seeded runner: `71189428e1d1c7aff8a3ddb55d56c2f2034cd89c9df516f1a6490e7a2888676c`
- predecessor receipt: `0d7bc9a65f6243cd0e0e1e83d670e76a2eb97a4c6ba27a5f9cf73f65a5620e5c`
- audit ledger: `a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04`

## Command

```powershell
python tools/research/run_fleet_seeded_stability_r2.py --outdir runs/research/BACKLOG-FLEET-SEEDED-STABILITY-02
```

## Factors

Four qualified text aliases; 16 GSM8K and eight protected-QA cases; three passes; 288 requests total. The treatment is temperature 0.2/top-p 0.95 with identical seed 20260826. Streaming and prompt caching remain off. Repeat 0 is paired with repeats 1 and 2 by exact semantic projection. Execution is sequential on the RTX 3090 through the live gateway, with exact request reconstruction retained before provenance sealing and terminal runner state bound before receipt.

## Acceptance gates

- `route_coverage`: `routes_completed eq 4`
- `request_coverage`: `recorded_requests eq 288`
- `request_integrity`: `successful_response_rate eq 1.0`
- `seeded_stability`: `exact_seeded_repeat_rate ge 0.9`
- `request_retention`: `retained_request_payloads eq 288`
- `terminal_binding`: `final_runner_state_bound eq True`
- `service_integrity`: `service_restarts eq 0`
- `service_recovery`: `initial_model_restored eq True`

## Abort conditions

Abort on a frozen input or preregistration mismatch, unhealthy systemd/gateway/embedding boundary, route identity failure, three consecutive request errors, missing pair baseline, request reconstruction below 288, service restart/restoration failure, incomplete provenance, or terminal-state binding failure. Partial rows remain evidence only and cannot qualify the task.

## Allowed claims

- `QUALIFIED_TEXT_FLEET_SEEDED_STABLE_R2`
- `QUALIFIED_TEXT_FLEET_SEEDED_UNSTABLE_R2`

Claims outside these codes are forbidden even if a metric looks favorable.
