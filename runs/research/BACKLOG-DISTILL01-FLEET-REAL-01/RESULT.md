# BACKLOG-DISTILL01-FLEET-REAL-01 result

Status: `EXECUTED`  
Executor: Codex executor  
Independent review: pending AGY

The historical DISTILL-01 promotion is not reproduced on clean, process-isolated real generations.

| Arm / composition | Math | QA | Total |
|---|---:|---:|---:|
| `target_all_linear` monolith | 10/32 | 3/16 | 13/48 |
| Routed fleet (`target_mlp_only` math + `target_attn_only` QA) | 10/32 | 5/16 | 15/48 |

The routed fleet's relative gain is `0.153846` (15.38%), below the frozen 20% threshold. Its math specialist scores 10/32, below the mandatory 15/32 threshold. The QA specialist reaches exactly 5/16 and passes that gate. All 144 source generations were independently rescored; stored flags matched, required arm/panel coverage was complete, and the source real-execution receipt was hash-verified.

This supports `DISTILL01_FALSE_POSITIVE_CONFIRMED_R1` if AGY independently verifies the packet. It evaluates deterministic routing composition of saved adapters, not a causal distillation-training effect or dynamic-router latency.
