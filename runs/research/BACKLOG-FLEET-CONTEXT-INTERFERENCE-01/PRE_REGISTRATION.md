# BACKLOG-FLEET-CONTEXT-INTERFERENCE-01 preregistration

Task: Stress qualified text routes with long-context associative retrieval under hard decoys
Evidence class: `serving_runtime`

## Hypothesis

Each qualified text route will retain at least 90% exact associative recall
when the requested labeled access record is embedded among 31 near-name and
near-code secure-record distractors, with no target-position bucket below 80%
inside the same verified per-slot context envelopes.

## Frozen inputs

- `runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/raw/receipt.json`
- `runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/raw/case_manifest.json`
- `runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/RESULT.md`
- `tools/research/run_fleet_context_envelope.py`
- `tools/research/run_fleet_context_envelope_r3.py`
- `config/qualified_model_fleet.json`

- Admission: `8f25d5d0d023b4bd1809ade207edf27137ec261a9408befbb2ef8464a3cbc112`.
- Context-envelope R3 receipt: `17f0ec8b541f6d769dd5909ca4a44bc3f7c2813d13cc51ec8b206d74090d15d6`.
- R3 case manifest: `63936f59148535a54ca54221d29dd669c387c87e07f0f28361899f18ee914111`.
- R3 result: `4ebc023dae15da6be96ea2ab2c62da34d9d6471a83c7be05d6f66c47ee0783a0`.
- Context core runner: `1ebb0c07145edd48f1fcc7d8f97b248a954ce4faa554e4d6f220c6d693eb857b`.
- Backend-tokenizer successor: `a674174eeeb4aebb2a5cd871ed6211085b5e1c28040d1e8041e519100ba857cf`.
- Fleet registry: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`.

## Command

```powershell
python tools/research/run_fleet_context_interference.py --outdir runs/research/BACKLOG-FLEET-CONTEXT-INTERFERENCE-01
```

## Factors

- The successful 72-cell R3 matrix, context targets, route order, token fitting,
  decoding and exact scorer remain unchanged.
- Each archive now contains 32 `SECURE ACCESS RECORD` entries. The target has
  exact label `ORION-DELTA`; 31 distractors use labels
  `ORION-DELTA-01..31` and codes sharing the same route/length prefix.
- The target record occupies start, middle or end; distractors are distributed
  through the archive. Two independent target codes are used per cell.
- Generation remains on gateway 8080; token fitting uses the verified active
  backend `/tokenize` endpoint. All prompts stay below per-slot context minus
  64 tokens.

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

- Abort on frozen identity mismatch, wrong backend identity, tokenizer failure,
  context overflow, incomplete matrix, three consecutive request failures,
  embedding failure or inability to restore the initial route and services.
- Retrieval misses remain evidence. No easier retry or post-result prompt
  wording change is allowed.

## Allowed claims

- `QUALIFIED_TEXT_FLEET_CONTEXT_INTERFERENCE_MEASURED_R1`
- `QUALIFIED_TEXT_FLEET_CONTEXT_INTERFERENCE_NOT_CONFIRMED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.
