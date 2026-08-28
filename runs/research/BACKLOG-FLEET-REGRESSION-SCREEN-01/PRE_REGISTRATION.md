# BACKLOG-FLEET-REGRESSION-SCREEN-01 preregistration

Task: Run a large repeatability and regression screen across the qualified text fleet  
Evidence class: `serving_runtime`  
Executor: Codex executor  
Date: 2026-08-26

## Hypothesis

Every frozen qualified text route can complete two greedy passes over the same
56-case panel without an HTTP failure, route-identity mismatch, service restart,
or loss of the embedding endpoint. At least 95% of the 224 model-case pairs will
produce byte-identical semantic output across the two passes, and the route that
was resident before the screen will be restored at the end.

Math, protected-QA and tool-dispatch scores are descriptive regression signals.
They do not authorize cross-model ranking or change any model card.

## Frozen inputs

- Admission specification: `config/research_backlog_admissions/BACKLOG-FLEET-REGRESSION-SCREEN-01.json`, 2,609 bytes, SHA-256 `31a9ba96a9272a6301fa73454c134b9f16216502de31c36faec68ea9f71e0991`.
- Fleet registry: `config/qualified_model_fleet.json`, 9,783 bytes, SHA-256 `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`.
- GSM8K source: `workloads/gsm8k.jsonl`, 389,701 bytes, SHA-256 `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`; freeze the first 32 records in file order.
- Protected-QA source: `runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl`, 11,016 bytes, SHA-256 `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`; freeze the first 16 records in file order.
- Agent-tool suite: `tools/benchmarks/agent_suite_v2.py`, 13,035 bytes, SHA-256 `14d0a1b76d4d729228678f215ecefa3254aef214eb65ac9d8d7061bccc0dc59e`; freeze all eight `CASES`.
- Backlog handoff: `docs/HANDOFF_2026-08-26_CONSOLIDATED_RESEARCH_BACKLOG.md`, 20,043 bytes, SHA-256 `895fec3ac345bdf26350b4a97f513bf4f4b3bad9898d09701db07a985f8b7d55`.
- Routes, in frozen order: `qwen38`, `hauhaucs`, `fable-tc`, `qwen36-moe`.
- Gateway: `http://127.0.0.1:8080`; embedding health endpoint: `http://127.0.0.1:8081/health`.
- GPU: NVIDIA RTX 3090, 24,576 MiB. Maximum resident generation models: one.

## Command

```powershell
python tools/research/run_fleet_regression_screen.py --outdir runs/research/BACKLOG-FLEET-REGRESSION-SCREEN-01
```

The runner is restartable. It appends one immutable JSON line per completed
cell, skips an already complete `(model, repeat, suite, case)` key, and writes a
receipt only after the full matrix and restoration checks finish.

## Factors

- Models: four frozen text aliases in the order above.
- Repeats: exactly two complete passes per model before moving to the next model.
- Panel per repeat: 32 GSM8K cases, 16 protected-QA cases and eight agent-tool cases.
- Total recorded requests: `4 x 2 x (32 + 16 + 8) = 448`.
- Decode: temperature `0`, seed `20260826`, streaming disabled, prompt cache disabled for math and QA, and model-specific thinking disabled through `enable_thinking=false`.
- Math budget: 256 completion tokens. QA budget: 64 completion tokens. Agent-tool budget: 384 completion tokens.
- The gateway performs single-model swapping. Route identity is sampled after the first request of each model and at every model boundary.
- Semantic repeatability hashes content, reasoning content, finish reason and normalized tool calls; timing and generated request IDs are excluded.
- Full raw responses, wall latency, server timings, draft counts, scores, process identity and GPU snapshots are retained.

## Acceptance gates

- `route_coverage`: `route_models_completed eq 4`
- `request_coverage`: `recorded_requests eq 448`
- `request_integrity`: `successful_response_rate eq 1.0`
- `route_identity`: `route_identity_verified eq True`
- `repeatability`: `exact_repeat_rate ge 0.95`
- `service_integrity`: `service_restarts eq 0`
- `embedding_integrity`: `embedding_health eq 200`
- `service_recovery`: `initial_model_restored eq True`

## Abort conditions

- Any frozen source hash or declared fleet artifact identity differs before the first request.
- The gateway or embedding endpoint is unhealthy before execution.
- The active systemd service changes PID or executable unexpectedly.
- The gateway reports a route other than the requested canonical alias after loading.
- Three consecutive request errors occur within one route.
- Port 8081 becomes unhealthy at a model boundary.
- GPU free memory falls below 512 MiB after a route becomes ready.
- The initially resident route cannot be restored. Restoration is attempted in a `finally` block even after another abort.

A wrong answer or non-identical repeat is evidence, not an abort. It fails or
lowers the corresponding metric and the remaining safe cells continue.

## Allowed claims

- `QUALIFIED_TEXT_FLEET_SCREEN_COMPLETE_R1`
- `QUALIFIED_TEXT_FLEET_SCREEN_REJECTED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.
The executor stops at `EXECUTED`; an independent actor must review the packet.
