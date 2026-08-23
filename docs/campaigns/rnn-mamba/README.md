# Campaign RNN — Neural Memory & Recurrent State Models (Mamba-2 / TPTT) 🧠

## Overview
Comprehensive empirical research into stateful linear-time architectures (Mamba-2, TTT, LaCT, Growing Memory, TPTT), investigating deterministic state capture, in-process state replay, and long-context needle retrieval.

## Key Files & Canonical Documentation
- [`REPLICATION_CATALOG_AND_PRELIMINARY_RESULTS.md`](REPLICATION_CATALOG_AND_PRELIMINARY_RESULTS.md): **Master Replication Ledger** — upstream code, Hugging Face checkpoints, published claims vs local lab measurements.
- [`COMPREHENSIVE_AUDIT_HYBRID_MEMORY_AND_ROADMAP_2026.md`](COMPREHENSIVE_AUDIT_HYBRID_MEMORY_AND_ROADMAP_2026.md): **Comprehensive Audit** — root cause analysis of abandoned lines, mathematical formulations, and engineering roadmap.
- [`HYBRID_RECURRENT_ECOSYSTEM_2026.md`](HYBRID_RECURRENT_ECOSYSTEM_2026.md): **Global 2026 Ecosystem Overview** — Llama.cpp PRs (Mamba2 #9126, Stateful API #23817), bug analyses, FLA, and Hugging Face hubs.
- [`RNN_RESEARCH_LEDGER.md`](RNN_RESEARCH_LEDGER.md): Master research ledger with primary source audit and classifications (21 architectures cataloged).
- [`RNN_STATE_MODEL.md`](RNN_STATE_MODEL.md): Mathematical specification of linear recurrent state transitions.
- [`RNN_MEMORY_CACHING_SPEC.md`](RNN_MEMORY_CACHING_SPEC.md): Growing memory caching specification (GRM / SSC / Memory Soup).
- [`RNN_RECONCILIATION.md`](RNN_RECONCILIATION.md): Architectural reconciliation.
- [`RNN_ARCHITECTURE_MATRIX.json`](RNN_ARCHITECTURE_MATRIX.json) & [`.csv`](RNN_ARCHITECTURE_MATRIX.csv): Taxonomy of neural memory architectures.

## Core Conclusion
**CONFIRMED & BOUNDED**:
1. **Deterministic State Reload**: Achieves **bit-exact reproducibility (40/40)** on Mamba-2 fast-path kernels.
2. **In-Run State Recovery (NoLiMa)**: Showed **no net signal** ($\Delta \approx 0$) for recovering lost historical information in-run vs final output generation.
3. **Multi-Query Associative Recall (MQAR) Capacity Cliff**:
   - **Mamba-2 1.3B (SSM)**: Exhibits a steep monotonic capacity cliff under KV interference ($P=4: 96.9\% \to P=128: 23.4\%$).
   - **DeltaNet 1.3B (Linear Attention)**: Shows flat-medium retention ($P=4: 71.9\% \to P=128: 54.7\%$).
   - **Qwen 2.5-0.5B / 1.5B (Dense Transformer)**: Maintains resilience ($45.3\% - 48.4\%$ at $P=128$).
   - **Qwen 3.8-27B (Dense Attention)**: Demonstrates **100.0% accuracy** through $P=1024$ (~11k tokens) and **95.0% NIAH deep context recall** through 30k tokens.
