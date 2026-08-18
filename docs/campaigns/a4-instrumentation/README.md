# Campaign A4 — High-Resolution Telemetry & Latency Instrumentation ⏱️

## Overview
Engineered deterministic profiling harnesses to capture accurate Time-to-First-Token (TTFT), inter-token latency variance, GPU clock stability, PCIe bandwidth constraints, and power envelope modulation.

## Key Files & Artifacts
- [`A4_INSTRUMENTATION.md`](A4_INSTRUMENTATION.md): Comprehensive instrumentation methodology and calibration guidelines.
- `src/model_lifecycle/collectors/host.py`: High-frequency host GPU/CPU metric sampler.
- `src/model_lifecycle/collectors/request.py`: Low-overhead streaming latency tracer.

## Core Conclusion
**OPERATIONALIZED**: Established host noise floor (~2.3% paired scatter on host `aaaaa`), validated exact sign test bounds ($p = 0.0312$ at $n=6$), and implemented GPU cooldown requirements for unbiased A/B inference trials.
