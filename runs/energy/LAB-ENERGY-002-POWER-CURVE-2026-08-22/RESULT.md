# LAB-ENERGY-002 — RTX 3090 power-limit Pareto curve

**Status:** `COMPLETE / QUALIFIED / RETAIN 420 W`

The 24-cell counterbalanced campaign completed without endpoint, boundary, telemetry,
power-limit-readback, or restoration failures. The original 420 W board limit was
restored in the harness `finally` block and independently read back afterward. The
voltage/frequency curve remained stock; **no undervolt was applied**.

## Decision

Keep the deployed power limit at **420 W**. None of the reduced limits met the frozen
rule: retain at least 95% of both long-workload prompt and decode throughput while
not increasing either gross energy metric.

- 378 W retained 99.31% of long prompt throughput and 95.66% of decode throughput,
  but gross prefill and decode energy increased by 1.69% and 0.86%. It is dominated
  by 420 W on the long workload.
- 336 W reduced long prefill/decode energy by 3.16%/4.94%, but throughput fell
  7.39%/9.10%, beyond the 5% guardrail.
- 294 W reduced long prefill/decode energy by 4.05%/6.91%, but throughput fell
  18.20%/17.91%.

This is a measurement decision only. The experiment did not change service defaults.

## Median results (three repetitions per cell)

| Limit | Workload | Prompt tok/s | Prefill J/prompt-token | Decode tok/s | Decode J/token | Peak W | Peak C |
|---:|---|---:|---:|---:|---:|---:|---:|
| 420 W | short | 1067.17 | 0.2056 | 78.15 | 4.787 | 406.99 | 71 |
| 378 W | short | 1114.30 | 0.2029 | 75.28 | 4.903 | 375.72 | 67 |
| 336 W | short | 1002.99 | 0.2282 | 70.71 | 4.709 | 336.17 | 66 |
| 294 W | short | 923.61 | 0.2348 | 63.00 | 4.649 | 294.05 | 65 |
| 420 W | long | 1233.26 | 0.2770 | 73.88 | 5.221 | 405.60 | 72 |
| 378 W | long | 1224.75 | 0.2817 | 70.67 | 5.265 | 377.70 | 71 |
| 336 W | long | 1142.15 | 0.2682 | 67.16 | 4.963 | 336.59 | 69 |
| 294 W | long | 1008.84 | 0.2658 | 60.65 | 4.860 | 296.02 | 66 |

The short workload illustrates why the decision is based on gross energy and both
workload lengths: lower limits reduce decode energy but can increase gross prefill
J/token when the brief prefill phase stretches. No universal “lower power is more
efficient” claim is supported.

## Protocol and validity

- Incumbent endpoint: historical Qwen3.8 Q4_K_XL, one slot, 131,072 context,
  q4_0/q4_0 KV and MTP n3.
- Limits: 420/378/336/294 W (100/90/80/70% of the 420 W default).
- Workloads inherited from qualified LAB-ENERGY-001: about 2.7k and 13.2k prompt
  tokens, 128-token greedy forced decode, `cache_prompt=false`.
- Three repetitions per limit/workload, counterbalanced limit order and alternating
  cell order; 80 ms telemetry and trapezoidal boundary interpolation.
- 24/24 runs had monotonic boundaries and zero telemetry errors. Prompt-token range
  was 2,678–2,685 for short and 13,239–13,244 for long.
- Peak observed board draw was 406.99 W and peak temperature was 72 C.
- Post-run state: power limit 420.00 W; ports 8080 and 8081 healthy;
  `llm-inference.service` active/running.

## Evidence

- Frozen protocol: `PRE_REGISTRATION.md`
- Raw receipts, aggregate and machine decision: `results.json`
- Harness: `tools/benchmarks/energy_power_curve.py`
- `results.json` SHA-256:
  `c212c077eb48a5133de3dd41fcc0d05e3023352ee7a1ca068f4569116b00433f`

