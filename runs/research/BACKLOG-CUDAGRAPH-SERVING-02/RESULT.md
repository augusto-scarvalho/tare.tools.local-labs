# BACKLOG-CUDAGRAPH-SERVING-02 result

Status: `EXECUTED` — independent review pending  
Execution window: 2026-08-25 22:20:50Z to 22:23:27Z  
Hardware: NVIDIA RTX 3090  
Receipt: `raw/receipt.json`, SHA-256 `eaa409f1620b45cb9bc939223f4407aed2b3be5615e5f4bd0027a00539f225f6`

## Outcome

The explicit CUDA Graph OFF/ON crossover completed all four preregistered blocks and 30 prompt pairs. Six of seven gates passed. The performance gate failed: CUDA Graph ON produced a median paired wall speedup of only `1.036998x`, below the preregistered `1.10x` threshold.

This packet remains at `EXECUTED`. The executor does not issue the rejection claim or author `REVIEW.json`; an independent actor must make that state transition.

## Causal design completed

| Block | Treatment | Process control | Recorded | Discarded warmups |
|---|---|---|---:|---:|
| B1 | OFF | `GGML_CUDA_DISABLE_GRAPHS=1` | 15 | 4 |
| B2 | ON | variable absent | 15 | 4 |
| B3 | ON | variable absent | 15 | 4 |
| B4 | OFF | `GGML_CUDA_DISABLE_GRAPHS=1` | 15 | 4 |

All block PIDs were separately inspected. They used the same executable, identical `argv`, identical model, identical request parameters and one binary SHA-256. The frozen backend library contains the `GGML_CUDA_DISABLE_GRAPHS` control. Prompt IDs 1–15 were observed in B1/OFF then B2/ON; IDs 16–30 were observed in B3/ON then B4/OFF, balancing treatment order over time.

## Metrics

| Metric | OFF | ON | Result |
|---|---:|---:|---:|
| Median wall latency | 1412.608 ms | 1348.767 ms | paired median speedup `1.036998x` |
| p95 wall latency | 1460.224 ms | 1439.080 ms | regression `-1.448%` |
| Exact semantic matches | — | — | 30/30 |
| Pairwise latency wins | 3/30 | 27/30 | ON faster in 90% of pairs |

Every recorded response generated exactly 64 completion tokens. The paired speedup range was `0.980044x` to `1.114601x`. CUDA Graph therefore showed a small, directionally consistent latency benefit on this tuple, but did not satisfy the preregistered minimum effect size.

## Acceptance gates

| Gate | Actual | Pass |
|---|---:|:---:|
| `treatment_identity` | explicit OFF/ON environments verified | yes |
| `binary_identity` | 1 distinct binary hash | yes |
| `balanced_crossover` | 4 valid ABBA blocks | yes |
| `semantic_parity` | mismatch rate 0.0 | yes |
| `paired_speedup` | 1.036998x, threshold 1.10x | **no** |
| `tail_non_regression` | -0.01447997 | yes |
| `service_recovery` | service and embedding restored | yes |

The old `BACKLOG-CUDAGRAPH-SERVING-01` result compared the first and second request to the same always-ON daemon and reported `1.5115x`. This causal successor does not reproduce that magnitude. Its evidence supports interpreting most of the earlier number as request-order/warmup bias rather than an OFF/ON CUDA Graph effect.

## Operational recovery

- The persistent service was stopped and restarted only through systemd.
- `llm-inference.service` returned `active/running` with the original executable and exact argument vector.
- PID changed from 26576 to 27589 after the controlled restart.
- `NRestarts` remained 0.
- Inference port 8080 and embedding port 8081 both returned `{"status":"ok"}`.
- All four transient units were stopped before the next block or final restoration.

## Verification

- Raw structural recount: 60 observations, 30 unique prompt IDs, one OFF and one ON observation per ID.
- Exact semantic matches: 30/30.
- Repository suite: 92/92 tests passed.
- `python tools/analysis/backlog_pipeline.py gate`: `PASS`.

## Claim boundary

The likely independent disposition is `SERVING_CUDAGRAPH_CAUSAL_REJECTED_R2` because the mandatory speed gate failed. That claim has not been issued by the executor. No driver-launch attribution, cross-model generalization or production promotion is supported.
