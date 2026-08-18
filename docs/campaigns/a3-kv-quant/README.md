# Campaign A3 — KV-Cache Quantization & Memory Limits 💾

## Overview
Evaluated symmetric vs asymmetric KV cache quantization (`q4_0`, `q8_0`, `iq4_nl`, `k8v4`, `k4v8`) to double context headroom and eliminate VRAM exhaustion on 24GB GPUs.

## Key Files & Artifacts
- [`A3_KV_QUANT.md`](A3_KV_QUANT.md): Complete measurement record and kernel analysis.
- `tools/analysis/gguf_kv.py`: KV cache geometry and VRAM calculator.

## Core Conclusion
**CLOSED / ALREADY OPTIMAL**: Symmetric `q4_0 / q4_0` KV cache is the definitive Pareto optimum for GPU inference (88.55 t/s, fused FlashAttention on-GPU). Asymmetric KV (`q8_0 / q4_0`) causes a **57% throughput penalty** due to kernel fallback to CPU.
