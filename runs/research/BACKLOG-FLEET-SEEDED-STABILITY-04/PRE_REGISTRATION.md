# BACKLOG-FLEET-SEEDED-STABILITY-04 preregistration

Task: Replicate seeded fleet stability with atomic request and frozen binary identity
Evidence class: `serving_runtime`

## Hypothesis

Across four qualified text routes, three fresh passes with seed 20260826,
temperature 0.2 and top-p 0.95 will produce identical semantic projections in
at least 90% of 192 repeat-to-baseline pairs. All 288 experimental calls must
be atomically paired to their response IDs with self-consistent request hashes;
route canaries must be excluded and the pending-call queue must finish empty.
Each route must also match a live PID, command, GGUF hash/bytes and executable
hash/bytes frozen before implementation. Any binding failure is a hold
regardless of repeatability.

## Frozen inputs

- `config/qualified_model_fleet.json`
- `config/fleet_runtime_binary_identities_2026-08-29.json`
- `workloads/gsm8k.jsonl`
- `runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl`
- `tools/research/run_fleet_regression_screen.py`
- `tools/research/run_fleet_seeded_stability.py`
- `tools/research/run_fleet_seeded_stability_r2.py`
- `tools/research/run_fleet_seeded_stability_r3.py`
- `runs/research/BACKLOG-FLEET-SEEDED-STABILITY-03/raw/receipt.json`
- `runs/research/BACKLOG-FLEET-SEEDED-STABILITY-03/REVIEW.json`

- admission: `4cdb2bc941d5287968b235cdde8f041d71794b37fe0bd4831361479a67bb30c0`
- fleet registry: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`
- binary identity ledger: `b1142241dc28556d407821b5d663da2e71e88e1d365d0e93a8a82f3918aaa7dd`
- GSM8K: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`
- protected QA: `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`
- fleet base runner: `7cbf942375c120970ac78fb40f7f15050ebddb7ec5372cb40bd721009faf1de3`
- seeded base runner: `71189428e1d1c7aff8a3ddb55d56c2f2034cd89c9df516f1a6490e7a2888676c`
- R2 runner: `1c6e0f1123acfef0a5756017b8f18e59cb7c626d72fdfa02b71093220e37f3c3`
- R3 runner: `c9e04f5557f07ba4cde146e50ad403ed081f577f74ce85e86cdcb63fbe4a79b1`
- R3 receipt: `fcb390b241a9d8b369e58c08d938e03782e71ed6c99efeaecffe1ec14b404e7c`
- R3 hold review: `897d6e94113f66f916180c29172ac23e17aba6d0346753cd53e3b788f348e3cb`

## Command

```powershell
python tools/research/run_fleet_seeded_stability_r4.py --outdir runs/research/BACKLOG-FLEET-SEEDED-STABILITY-04
```

## Factors

Four qualified text aliases; 16 GSM8K and eight protected-QA cases; three
passes; 288 requests and 192 repeat-to-baseline comparisons. Sampling is fixed
at temperature 0.2, top-p 0.95 and seed 20260826, with streaming and prompt
caching disabled. Only calls matching that complete experimental contract enter
the capture path. The HTTP wrapper stores the actual argument together with the
returned response ID; sample append requires the same ID and immediately writes
the request plus canonical hash. The inherited reconstruction hook becomes a
validator and cannot overwrite requests. Before receipt creation every row hash
must match, every response ID must bind, and the call queue must be empty.
Physical identity is captured after each route canary but before its first
experimental request and compared against the frozen GGUF registry plus the
versioned executable identity ledger.

## Acceptance gates

- `route_coverage`: `routes_completed eq 4`
- `request_coverage`: `recorded_requests eq 288`
- `request_integrity`: `successful_response_rate eq 1.0`
- `seeded_stability`: `exact_seeded_repeat_rate ge 0.9`
- `request_retention`: `retained_request_payloads eq 288`
- `request_hash_integrity`: `captured_request_hash_match_rate eq 1.0`
- `response_binding`: `captured_response_id_match_rate eq 1.0`
- `capture_queue_drained`: `pending_captured_calls eq 0`
- `terminal_binding`: `final_runner_state_bound eq True`
- `physical_route_coverage`: `physical_routes_bound eq 4`
- `physical_model_identity`: `physical_model_hash_match_rate eq 1.0`
- `physical_binary_identity`: `physical_binary_hash_match_rate eq 1.0`
- `process_command_identity`: `process_command_identity_rate eq 1.0`
- `service_integrity`: `service_restarts eq 0`
- `service_recovery`: `initial_model_restored eq True`

## Abort conditions

Abort on frozen-input/preregistration mismatch, unhealthy service/gateway/
embedding boundary, missing backend PID or `/proc` identity, route or command
mismatch, GGUF hash/bytes mismatch, executable hash/bytes mismatch, captured
canary, response-ID mismatch, request-hash mismatch, nonempty pending-call
queue, three consecutive request errors, incomplete 288-row/192-pair panel,
service restart/restoration failure, incomplete provenance or terminal-state
binding failure. Partial evidence remains nonqualifying and predecessors remain
immutable.

## Allowed claims

- `QUALIFIED_TEXT_FLEET_SEEDED_ATOMIC_PHYSICAL_STABLE_R4`
- `QUALIFIED_TEXT_FLEET_SEEDED_ATOMIC_PHYSICAL_UNSTABLE_R4`

Claims outside these codes are forbidden even if a metric looks favorable.
