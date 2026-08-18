# Campaign GDN — Gated Delta Net CUDA Kernel Optimization 🚀

## Overview
Profiled and benchmarked custom CUDA kernels for Gated Delta Net (GDN) hybrid linear-attention blocks, exploring TensorFloat-32 (TF32) precision, chunked scans, and kernel fusion to maximize decoding throughput.

## Key Files & Artifacts
- [`GDN_KERNEL.md`](GDN_KERNEL.md): Kernel performance analysis and architectural breakdown.
- [`GDN_M4_RESUME.md`](GDN_M4_RESUME.md): M4 milestone resume and execution path.
- [`GDN_NEXT_LEVERS.md`](GDN_NEXT_LEVERS.md): Next optimization levers for recurrent linear models.
- [`GDN_TF32_PLAN.md`](GDN_TF32_PLAN.md): TF32 precision plan and error tolerances.
- `tools/benchmarks/gdn_conc_bench.py`: GDN concurrency and chunked scan benchmark.

## Core Conclusion
**ADAPTED**: Demonstrated that linear attention blocks eliminate 75% of KV-cache layers in hybrid models (only 10 of 40 layers require KV cache in Qwen3.5 MoE), yielding massive memory savings for long-context workloads.
