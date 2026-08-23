# LAB-ENERGY-002 — RTX 3090 power-limit Pareto curve

Frozen before the first non-baseline power-limit run on 2026-08-22.

## Question

Can the incumbent Qwen3.8 IQ4_XS service reduce gross GPU energy per token without
materially reducing throughput when the RTX 3090 board power limit is reduced?

## Fixed conditions

- Live incumbent endpoint: `http://127.0.0.1:8080`.
- Initial/default board limit: 420 W; test limits: 420/378/336/294 W
  (100/90/80/70% of the 420 W default).
- Voltage condition: stock voltage/frequency curve, **no undervolt**. Power-limit is
  the only intended GPU control variable.
- Existing LAB-ENERGY-001 measurement method: 80 ms `nvidia-smi power.draw`
  sampling; trapezoidal integration; request-start to first streamed token is
  prefill/TTFT; first token to final event is decode; gross GPU energy is primary.
- `cache_prompt=false`, greedy generation, `ignore_eos=true`, 128 predicted tokens.
- Workloads: short (`repeats=240`, about 2.7k prompt tokens) and long
  (`repeats=1200`, about 13.2k prompt tokens), three repetitions per limit/cell.
- Limit order is counterbalanced across repetitions; cell order alternates within
  each limit. A three-second settling interval follows every limit change.

## Validity gates

- Every requested power limit must be read back within 1 W.
- Every run must have monotonic phase boundaries and no telemetry errors.
- Prompt token counts must be stable within each workload (unique nonce aside).
- The initial power limit must be restored and read back in a `finally` block.
- Any service failure, failed limit transition, or failed restoration makes the
  campaign operationally unqualified; partial receipts are retained.

## Decision rule

For each limit and workload, use medians across three repetitions. Report prompt
throughput, decode throughput, gross prefill J/prompt-token, and gross decode
J/token. A point is Pareto-dominated when another tested limit is no worse on all
four metrics for that workload and strictly better on at least one.

The deployment recommendation is the lowest limit that, on the **long** workload,
retains at least 95% of both 420 W prompt throughput and 420 W decode throughput,
while not increasing either gross energy metric. If no reduced limit meets that
rule, retain 420 W. This experiment does not itself mutate deployment defaults.

