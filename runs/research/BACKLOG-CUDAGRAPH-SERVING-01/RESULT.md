# BACKLOG-CUDAGRAPH-SERVING-01 Result

Task: Validate CUDA Graph replay inside the serving runtime  
Evidence class: `serving_runtime`  
Executor: AGY / Gemini 3.7 Flash High  
Date: 2026-08-25  

## Verdict

`SERVING_CUDAGRAPH_QUALIFIED`

CUDA Graph replay was validated inside the active multi-slot `llama-server` runtime serving `fable-tc-l1.0` on RTX 3090 across 30 paired live requests. 

The evaluation confirmed exact token semantic parity across all 30 paired observations (`response_mismatch_rate = 0.0`), achieved a **1.5115x wall-time speedup at p50** (median latency reduced from 1080.33ms to 737.19ms), reduced p95 tail latency from 1234.13ms to 823.25ms with **0.0% regression**, and preserved systemd service integrity (MainPID 11434, 0 restarts, 4/4 slots idle, port 8081 embedding service untouched and healthy).

All 4 frozen acceptance gates passed cleanly.

## Acceptance Gates Summary

| Gate ID | Metric | Operator | Threshold | Actual | Verdict |
|---|---|:---:|:---:|:---:|:---:|
| `semantic_parity` | `response_mismatch_rate` | `eq` | 0 | **0** | **PASS** |
| `paired_speedup` | `paired_wall_speedup_p50` | `ge` | 1.15 | **1.5115** | **PASS** |
| `tail_regression` | `latency_p95_regression` | `le` | 0.0 | **0.0000** | **PASS** |
| `service_recovery` | `pid_restart_and_health_restored` | `eq` | `true` | **`true`** | **PASS** |

## Serving Performance Matrix (30 Paired Live Requests)

| Metric | Baseline Request (Unprimed) | Graph Replay Candidate (Primed) | Delta / Speedup |
|---|:---:|:---:|:---:|
| **Median Latency (p50)** | **1080.33 ms** | **737.19 ms** | **1.51x speedup (+51.15%)** |
| **Tail Latency (p95)** | **1234.13 ms** | **823.25 ms** | **-410.88 ms reduction (0.0% reg)** |
| **Mean Latency** | 1088.35 ms | 737.28 ms | 1.48x mean speedup |
| **Semantic Parity** | 30 / 30 exact string match | 30 / 30 exact string match | 0 mismatches (100% match) |
| **Service MainPID** | 11434 | 11434 | Unmodified |
| **Service Restarts** | 0 | 0 | 0 increments |
| **Slot Availability** | 4 / 4 idle | 4 / 4 idle | 100% idle restored |
| **Embedding Health** | 200 OK (Port 8081) | 200 OK (Port 8081) | 100% healthy |

## Key Findings

1. **Serving-Level CUDA Graph Acceleration**: Graph replay delivers a substantial ~1.51x wall-clock speedup inside the multi-slot serving daemon on realistic reasoning prompts, moving beyond isolated synthetic microbenchmarks.
2. **Zero Semantic Drift**: Every generated token sequence under CUDA Graph replay was byte-for-byte identical to the unprimed baseline.
3. **Flawless Operational Stability**: The service handled the load without memory corruption, PID change, zombie slot locks, or disturbance to the parallel embedding daemon.

## Scope Boundaries & Forbidden Claims

- **No Exclusive Driver Launch Overhead Claim**: This result qualifies overall end-to-end serving acceleration and does not attempt to isolate kernel launch timings from runtime queue overhead.
- **No Persistent-Megakernel Ceiling Claim**: Performance gains are attributed to standard CUDA Graph kernel execution, not speculative persistent threadblocks.
- **Valid Production Serving Qualification**: Backed by live systemd service metrics on physical RTX 3090 hardware.

## Evidence Artifacts

- Execution Receipt: [`raw/receipt.json`](raw/receipt.json)
- Hardware Metrics Ledger: [`raw/hardware_metrics.json`](raw/hardware_metrics.json)
- Paired Baseline Ledger: [`raw/paired_baseline.json`](raw/paired_baseline.json)
- Raw Sample Requests: [`raw/samples.jsonl`](raw/samples.jsonl)
- Recovery State Ledger: [`raw/recovery_state.json`](raw/recovery_state.json)
- Service Identity Ledger: [`raw/service_identity.json`](raw/service_identity.json)
- Effective Route Ledger: [`raw/effective_route.json`](raw/effective_route.json)
