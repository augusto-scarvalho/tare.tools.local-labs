# BACKLOG-CUDAGRAPH-SERVING-01 preregistration

Task: Validate CUDA Graph replay inside the serving runtime
Evidence class: `serving_runtime`

## Hypothesis

Executing paired live requests against the active multi-slot serving runtime (`llama-server` on RTX 3090) with CUDA Graph replay will preserve exact semantic response parity (`response_mismatch_rate eq 0`), achieve a median wall-time speedup p50 of at least 1.15x (`paired_wall_speedup_p50 ge 1.15`), prevent p95 latency tail regression (`latency_p95_regression le 0`), and complete without modifying the service PID, restart counter, or embedding port 8081 health (`pid_restart_and_health_restored eq True`).

## Frozen inputs

Source artifacts and historical baselines:
- `runs/research/SLX-05D-CUDA-GRAPH-REPLAY-2026-08-25/RESULT.md` (756 bytes, SHA-256: `badb8dcc1fc341994a6cb681588041217d858db4aef392bd718fc66d031e426b`)
- `runs/research/SLX-01C-SERVING-TORTURE-2026-08-25/RESULT.md` (635 bytes, SHA-256: `c7e448cb78bf954ce04ca6dfa934073a55b6c9236e9b366dfaf456c1f3f7bef9`)

Serving runtime configuration and model assets:
- Service Unit: `/etc/systemd/system/llm-inference.service`
- Service Drop-in: `/etc/systemd/system/llm-inference.service.d/serve-fable-tc.conf`
- Binary Path: `/home/augus/opt/slop.cpp/b10159-068764d92-fable-tc/bin/llama-server`
- Model Path: `/home/augus/models/merges/fable-tc-l1.0-Q4_K_M.gguf`
- Embedding Service: `http://127.0.0.1:8081`

## Command

```powershell
python tools/research/run_serving_cudagraph_benchmark.py --outdir runs/research/BACKLOG-CUDAGRAPH-SERVING-01
```

## Factors

- Serving Endpoint: `http://127.0.0.1:8080` (Fable-TC 4-slot serving runtime).
- Embedding Endpoint: `http://127.0.0.1:8081` (Nomic embed service, monitored for non-interference).
- Paired Request Count: 30 paired deterministic requests (greedy temperature=0.0, max_tokens=48).
- Metrics: Per-request latency, token throughput, exact response string match, p50 and p95 latency percentiles, systemd MainPID and restart count before and after.
- Hardware / Runtime: NVIDIA GeForce RTX 3090 (24.5GB VRAM), WSL2 Ubuntu-24.04.

## Acceptance gates

- `semantic_parity`: `response_mismatch_rate eq 0`
- `paired_speedup`: `paired_wall_speedup_p50 ge 1.15`
- `tail_regression`: `latency_p95_regression le 0`
- `service_recovery`: `pid_restart_and_health_restored eq True`

## Abort conditions

- Systemd MainPID mutation or non-zero restart increments during execution.
- Embedding port 8081 health check failure.
- Semantic mismatch or decoding divergence across paired deterministic requests.
- HTTP timeout or unhandled 5xx error on serving endpoints.

## Allowed claims

- `SERVING_CUDAGRAPH_QUALIFIED`
- `SERVING_CUDAGRAPH_REJECTED`

Claims outside these codes are forbidden even if a metric looks favorable.
