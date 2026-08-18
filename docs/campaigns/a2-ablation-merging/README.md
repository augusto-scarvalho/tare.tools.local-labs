# Campaign A2 — Model Merging, Layer Ablation & Concision Gating 🎭

## Overview
Explored model concision, refusal mitigation (abliteration of directional refusal vectors), and parameter merging (SLERP, DARE, Task Arithmetic) to produce fast, non-verbose, highly accurate models without degraded writing craft.

## Key Files & Artifacts
- [`A2_STAGE1_CONCISE_FABLE.md`](A2_STAGE1_CONCISE_FABLE.md): Stage 1 concision and token-budget reduction proofs.
- [`A2_STAGE2_PLAN.md`](A2_STAGE2_PLAN.md): Stage 2 ablation & merging plan.
- [`A2_STAGE2_EVIDENCE_ablit.md`](A2_STAGE2_EVIDENCE_ablit.md): Directional ablation evidence.
- [`A2_STAGE2_EVIDENCE_merging.md`](A2_STAGE2_EVIDENCE_merging.md): SLERP / DARE merge comparisons.
- [`A2_GATE3_RESULT.md`](A2_GATE3_RESULT.md): Blind multi-judge pairwise writing-quality evaluation quorum.
- [`A2_THINKINGCAP.md`](A2_THINKINGCAP.md): ThinkingCap fine-tune performance evaluation.
- [`A2_PERF_LEVERS.md`](A2_PERF_LEVERS.md): Performance levers summary.

## Core Conclusion
**PROMOTED**: `fable-tc-l1.0` passed the writing quality gate with **-55% reasoning tokens** and **-23% creative output length** while maintaining strict quality parity with full-length verbose models across a 4-judge blind panel.
