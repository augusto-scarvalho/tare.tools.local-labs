# BACKLOG-FLEET-SEEDED-STABILITY-03 preregistration

Task: Replicate seeded fleet stability with live physical route identity
Evidence class: `serving_runtime`

## Hypothesis

Across the four qualified text routes, three fresh passes with identical seed
20260826, temperature 0.2 and top-p 0.95 will produce identical semantic
projections in at least 90% of 192 repeat-to-baseline pairs. Every route must
also be attributed at request time to a live backend PID whose `/proc` command
line names the expected GGUF and whose executable and GGUF SHA-256 values match
the frozen fleet registry. A result without that physical binding is a hold
regardless of the observed repeat rate.

## Frozen inputs

- `config/qualified_model_fleet.json`
- `workloads/gsm8k.jsonl`
- `runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl`
- `tools/research/run_fleet_regression_screen.py`
- `tools/research/run_fleet_seeded_stability.py`
- `tools/research/run_fleet_seeded_stability_r2.py`
- `runs/research/BACKLOG-FLEET-SEEDED-STABILITY-02/raw/receipt.json`
- `runs/research/BACKLOG-FLEET-SEEDED-STABILITY-02/REVIEW.json`

- admission: `5a68266de6f5dc97e4bdced3a43d510e298397fc10d6a84b77440170ba543349`
- fleet registry: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`
- GSM8K: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`
- protected QA: `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`
- fleet base runner: `7cbf942375c120970ac78fb40f7f15050ebddb7ec5372cb40bd721009faf1de3`
- seeded base runner: `71189428e1d1c7aff8a3ddb55d56c2f2034cd89c9df516f1a6490e7a2888676c`
- R2 successor runner: `1c6e0f1123acfef0a5756017b8f18e59cb7c626d72fdfa02b71093220e37f3c3`
- R2 receipt: `efbd36ba4d8178dcddf9554953d553d2aea6168c04bad11b2f6d9ccff29d5311`
- R2 hold review: `be288bc446d1a047b794512c3a9ae471213a8e9a51f0a5c48ca43b2982af6b70`

## Command

```powershell
python tools/research/run_fleet_seeded_stability_r3.py --outdir runs/research/BACKLOG-FLEET-SEEDED-STABILITY-03
```

## Factors

Four qualified text aliases; 16 GSM8K and eight protected-QA cases; three
passes; 288 fresh requests and 192 repeat-to-baseline comparisons. The sampling
contract is temperature 0.2, top-p 0.95, seed 20260826, with streaming and
prompt caching disabled. Execution is sequential through the live gateway on
the RTX 3090. Immediately after each route switch and before its first request,
the runner records the gateway backend PID, reads `/proc/<pid>/exe` and
`/proc/<pid>/cmdline` inside WSL, resolves the `-m` GGUF, and computes SHA-256
and byte size for the executable and model. Those observations must match the
registry entry and are included in provenance before receipt creation. Request
payloads are retained in every sample and terminal state is bound before the
receipt.

## Acceptance gates

- `route_coverage`: `routes_completed eq 4`
- `request_coverage`: `recorded_requests eq 288`
- `request_integrity`: `successful_response_rate eq 1.0`
- `seeded_stability`: `exact_seeded_repeat_rate ge 0.9`
- `request_retention`: `retained_request_payloads eq 288`
- `terminal_binding`: `final_runner_state_bound eq True`
- `physical_route_coverage`: `physical_routes_bound eq 4`
- `physical_model_identity`: `physical_model_hash_match_rate eq 1.0`
- `physical_binary_identity`: `physical_binary_hash_match_rate eq 1.0`
- `process_command_identity`: `process_command_identity_rate eq 1.0`
- `service_integrity`: `service_restarts eq 0`
- `service_recovery`: `initial_model_restored eq True`

## Abort conditions

Abort on a frozen-input or preregistration mismatch, unhealthy systemd/gateway
or embedding boundary, absent/nonpositive backend PID, inaccessible `/proc`
identity, alias mismatch, command line missing the expected `-m` GGUF, resolved
binary or model path mismatch, binary/model SHA-256 mismatch, model byte-count
mismatch, three consecutive request failures, fewer than 288 retained requests,
missing repeat pairs, service restart/restoration failure, incomplete
provenance, or terminal-state binding failure. Partial rows and identity
snapshots remain evidence only and cannot qualify the task. Identity inferred
only from the gateway alias or static registry is insufficient.

## Allowed claims

- `QUALIFIED_TEXT_FLEET_SEEDED_PHYSICALLY_BOUND_STABLE_R3`
- `QUALIFIED_TEXT_FLEET_SEEDED_PHYSICALLY_BOUND_UNSTABLE_R3`

Claims outside these codes are forbidden even if a metric looks favorable.
