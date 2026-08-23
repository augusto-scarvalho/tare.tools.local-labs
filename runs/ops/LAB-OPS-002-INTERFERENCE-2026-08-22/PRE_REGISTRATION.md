# LAB-OPS-002 — controlled endpoint interference matrix

Frozen on 2026-08-22 before starting any contender.

## Question

How much do bounded CPU, RAM, disk-read and GPU-compute contenders degrade the live
canonical endpoint relative to an uncontended baseline?

## Fixed target and workload

- Canonical Qwen3.8 Q4_K_XL endpoint on 8080, one slot, 131,072 context, q4 KV,
  MTP n3, board power 420 W, `lmctl` mode `SERVE`.
- LAB-ENERGY-001 short workload: about 2.7k prompt tokens, forced 128-token greedy
  decode, unique prompt, `cache_prompt=false`.
- Three repetitions per condition, counterbalanced order; medians are primary.
- Same 80 ms phase-aligned telemetry captures prompt/decode throughput and gross
  GPU J/token. Contender setup time is outside the endpoint timing window.

## Contenders

- `baseline`: no load generator.
- `cpu`: 12 busy-loop worker processes on the 24-logical-CPU host.
- `ram`: allocate and touch 8 GiB anonymous WSL memory, then retain it.
- `disk`: repeated read-only direct-I/O scan of the inactive Fable GGUF.
- `gpu`: repeated 2048x2048 FP16 PyTorch matmul using the existing CUDA environment.

Each contender must emit `READY`, remains bounded by 90 seconds, accepts an explicit
stdin stop, and is verified dead after its cell. GPU cells abort before inference if
post-ready free VRAM is below 1,024 MiB. RAM preflight requires at least 16 GiB host
available. No clocks, voltages, model files or deployment settings are mutated.

## Decision

For each contender report median prompt/decode throughput and gross energy relative
to baseline. Mark an axis `MATERIAL` when absolute throughput degradation exceeds
10% or either gross energy/token metric increases over 10%. This is a characterization,
not an authorization to colocate workloads. Any crash, telemetry error, cleanup
failure or endpoint health failure invalidates the campaign. Embedding 8081 must
remain healthy.

