# LAB-OPS-002 — controlled endpoint interference matrix

**Status:** `COMPLETE / QUALIFIED / GPU COLOCATION MATERIAL`

The 15-cell counterbalanced matrix completed with no endpoint, telemetry, cleanup or
health failures. Bounded CPU, retained RAM and direct disk-read contenders caused
small short-workload prefill shifts below the preregistered 10% materiality threshold.
The CUDA contender was material because it sharply increased gross GPU energy even
though throughput loss alone remained below 10%.

## Median results (three repetitions)

| Condition | Prompt tok/s | Prompt degradation | Prefill J/token | Energy change | Decode tok/s | Decode degradation | Decode J/token | Energy change | Material |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 1107.35 | — | 0.1988 | — | 78.15 | — | 4.614 | — | no |
| CPU, 12 workers | 1059.27 | 4.34% | 0.2009 | +1.06% | 78.93 | −0.99% | 4.421 | −4.19% | no |
| RAM, 8 GiB retained | 1058.45 | 4.42% | 0.1839 | −7.49% | 78.15 | 0.00% | 4.605 | −0.19% | no |
| disk, direct read | 1051.43 | 5.05% | 0.2100 | +5.68% | 78.88 | −0.93% | 4.527 | −1.89% | no |
| GPU, FP16 matmul | 1073.20 | 3.08% | 0.3061 | **+54.00%** | 72.57 | **7.14%** | 4.997 | **+8.30%** | **yes** |

Negative degradation means a small apparent improvement and must not be interpreted
as a contender benefit at n=3. The purpose is to identify damaging colocation, not to
promote background loads.

## Interpretation

- CPU saturation of half the logical CPUs, 8 GiB anonymous memory retention and
  repeated direct disk reads did not materially interfere with this short, mostly
  GPU-resident single-request workload. Their prefill shifts were 4.3–5.1%.
- GPU matmul shared the same board with the server while preserving the 1 GiB reserve
  gate (minimum post-ready free VRAM 1,568 MiB). It raised prefill energy per token by
  54.0%, the primary material failure, and also reduced decode throughput by 7.14%.
- Throughput-only monitoring would have missed the largest interference cost. GPU
  energy must remain part of the colocation guard.
- These results are bounded to one short prompt and one request at a time. They do not
  qualify CPU/RAM/disk colocation under long context, concurrency or host-memory
  pressure near the reserve boundary.

## Protocol and validity

- Incumbent: canonical Qwen3.8 Q4_K_XL, q4 KV, MTP n3, one slot, 420 W.
- Workload: approximately 2.7k prompt tokens and forced 128-token greedy decode,
  `cache_prompt=false`, three repetitions per contender.
- Gross phase energy: 80 ms `nvidia-smi power.draw` and trapezoidal integration.
- CPU/RAM/disk/GPU contenders were bounded to 90 seconds, emitted readiness, accepted
  explicit stop and exited zero after each cell.
- RAM preflight observed more than 16 GiB available. The GPU contender used only about
  34 MB explicit tensor allocation plus the CUDA context and left 1,568 MiB free.
- 15/15 endpoint runs had monotonic boundaries and zero telemetry errors.
- No contender remains. `lmctl mode check serve` is coherent; 8080 and 8081 are
  healthy; board power is 420 W.

## Evidence

- `PRE_REGISTRATION.md`: frozen loads, safety gates and materiality rule.
- `results.json`: all per-cell telemetry and aggregates, SHA-256
  `b27ccc56eae33a1cb258de9961a33a5d79a4d3020350f13a89ca06c6f485672c`.
- Orchestrator: `tools/benchmarks/interference_matrix.py`.
- Bounded load generator: `tools/contenders/interference_load.py`.

