# Campaign RNN — Neural Memory & Recurrent State Models (Mamba-2 / TPTT) 🧠

## Overview
Comprehensive empirical research into stateful linear-time architectures (Mamba-2, TTT, LaCT, Growing Memory, TPTT), investigating deterministic state capture, in-process state replay, and long-context needle retrieval.

## Key Files & Artifacts
- [`RNN_RESEARCH_LEDGER.md`](RNN_RESEARCH_LEDGER.md): Master research ledger with primary source audit and classifications.
- [`RNN_STATE_MODEL.md`](RNN_STATE_MODEL.md): Mathematical specification of linear recurrent state transitions.
- [`RNN_MEMORY_CACHING_SPEC.md`](RNN_MEMORY_CACHING_SPEC.md): Growing memory caching specification.
- [`RNN_RECONCILIATION.md`](RNN_RECONCILIATION.md): Architectural reconciliation.
- [`RNN_ARCHITECTURE_MATRIX.json`](RNN_ARCHITECTURE_MATRIX.json) & [`.csv`](RNN_ARCHITECTURE_MATRIX.csv): Taxonomy of 18 neural memory architectures.

## Core Conclusion
**CONFIRMED & BOUNDED**:
1. Deterministic in-process state reload achieves **bit-exact reproducibility (40/40)** on Mamba-2 fast-path kernels.
2. In-run intermediate recurrent state capture showed **no net signal** for recovering lost information on NoLiMa benchmarks compared to final output generation.
