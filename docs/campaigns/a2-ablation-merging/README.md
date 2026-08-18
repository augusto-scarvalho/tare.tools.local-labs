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

## 🧬 Model Lineage in this Campaign
- **`Qwen3.6-Fable-Heretic` (Hugging Face fine-tune)**: Upstream uncensored fine-tune.
- **`Qwen3.6-ThinkingCap` (BottleCapAI fine-tune)**: Upstream token-efficiency fine-tune.
- **`Qwen3.6-Fable-TC` / `fable-tc-l1.0` (Author-Created Original Merge by Augusto Carvalho)**: Proprietary full-rank task-arithmetic merge combining ThinkingCap concision with Fable base:
  $$W = W_{\text{Fable}} + \lambda (W_{\text{TC}} - W_{\text{base}})$$
  Empirical parameter sweeps ($\lambda \in \{0.4, 0.7, 1.0\}$) proved $\lambda=1.0$ is the optimal sweet spot (98.3% GSM8K accuracy, -54.8% reasoning tokens, 0 starved runs).
- **`Qwen3.6-Fable-Fusion-711` (DavidAU Community Merge)**: External community merge evaluated strictly as a baseline.

## Core Conclusion
**PROMOTED**: `fable-tc-l1.0` passed the writing quality gate with **-55% reasoning tokens** and **-23% creative output length** while maintaining strict quality parity with full-length verbose models across a 4-judge blind panel.
