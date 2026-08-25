# Strategic Research & Lab Knowledge Base 📖

## Overview
Houses long-range architectural analyses, experimental registries, mechanism deep-dives, and prioritized research backlogs for the `tare.tools` inference infrastructure.

## Epistemic posture

Artifact hashes freeze attribution, not authority. Every model, quant, experiment design, promotion rule,
and working assumption remains open to supersession by stronger evidence or a better operational Pareto
trade-off. Preserve prior receipts and mark displaced conclusions `SUPERSEDED`; do not silently delete them,
and do not give an incumbent preference merely because it was qualified first.

## Key Files & Artifacts
- [`../EXECUTION_CLOSEOUT_2026-08-24_25.md`](../EXECUTION_CLOSEOUT_2026-08-24_25.md): consolidated execution ledger, persistent host/WSL effects, recovered failures, explicit non-actions, Git/CI receipts, live baseline, and trigger-gated pending work for the 2026-08-24/25 continuation.
- [`BEELLAMA_SLOP_PEFT_ANALYSIS_2026-08-24.md`](BEELLAMA_SLOP_PEFT_ANALYSIS_2026-08-24.md): transcript reconciliation, pinned source archaeology, runtime challenger gates, adapter mechanics/matrix results, and the codec-independent KV qualification queue.
- [`STATUS.md`](STATUS.md): Master empirical register of all settled questions, noise floors, and paired trials.
- [`EXPERIMENTS.md`](EXPERIMENTS.md): Chronological ledger of historical, active, and completed experiments.
- [`IDEAS_BACKLOG.md`](IDEAS_BACKLOG.md): Ranked research backlog with priority matrices.
- [`BACKLOG_V2_STATUS.md`](BACKLOG_V2_STATUS.md): Backlog V2 execution tracking.
- [`AUTONOMOUS_CAMPAIGN_REPORT_2026-08-23.md`](AUTONOMOUS_CAMPAIGN_REPORT_2026-08-23.md): consolidated outcomes, operational impact, findings, provenance policy, and residual queue from the 2026-08-21/23 autonomous campaign.
- [`REMAINING_EXPERIMENTS_2026-08-22.md`](REMAINING_EXPERIMENTS_2026-08-22.md): reconciled post-wave queue, with executable design work separated from blocked, parked and cancelled experiments.
- [`LANDSCAPE.md`](LANDSCAPE.md): Competitive and scientific panorama of open-weight LLMs.
- [`MECHANISMS.md`](MECHANISMS.md): Low-level hardware, memory bandwidth, and kernel mechanics.
- [`CONTEXT_PLAN.md`](CONTEXT_PLAN.md): Context extension planning and YaRN frequency scaling.
- [`FORK.md`](FORK.md): Fork maintenance and upstream synchronization policy.
- [`QWEN38_IMATRIX_COLD_FUSION_ASSESSMENT_2026-08-21.md`](QWEN38_IMATRIX_COLD_FUSION_ASSESSMENT_2026-08-21.md): Revision-pinned assessment and proposed qualification plan for the current Unsloth imatrix/quants and DavidAU Cold Fusion.
- [`MUSE_GLIMMER_3090_EXPERIMENT_PACKET_2026-08-21.md`](MUSE_GLIMMER_3090_EXPERIMENT_PACKET_2026-08-21.md): executed fail-closed qualification packet for the official Muse Glimmer 30B 17 GB quant, perception encoder, and target-matched DFlash drafter on the RTX 3090; final decision `HOLD`, drafter rejected.
- [`../../runs/requalification/QWEN38-UNSLOTH-REVISION-2026-08-21/RESULT.md`](../../runs/requalification/QWEN38-UNSLOTH-REVISION-2026-08-21/RESULT.md): executed revision-drift screen; current IQ4_XS and Q2_K_XL both rejected for supersession after compact and broad gates.
- [`../../runs/requalification/COLD-FUSION-2026-08-22/RESULT.md`](../../runs/requalification/COLD-FUSION-2026-08-22/RESULT.md) and [`COLD-FUSION-MTP`](../../runs/requalification/COLD-FUSION-MTP-2026-08-22/RESULT.md): the compact base role and later explicitly authorized nine-cell descriptive MTP arm both closed as rejected.
- [`../../runs/cache/LAB-CACHE-001-MTP-2026-08-22/RESULT.md`](../../runs/cache/LAB-CACHE-001-MTP-2026-08-22/RESULT.md): explicit no-spec slot persistence passed; intermittent MTP cache/restore oracle failure blocks persistent speculative state.
- [`../../runs/agent/LAB-AGENT-002-2026-08-22/RESULT.md`](../../runs/agent/LAB-AGENT-002-2026-08-22/RESULT.md): 39/40 perturbation matrix; systematic positional tool-order sensitivity blocks robust irreversible recovery.
- [`../../runs/agent/LAB-AGENT-003-2026-08-22/RESULT.md`](../../runs/agent/LAB-AGENT-003-2026-08-22/RESULT.md): corrected bounded stress/scale matrix passed 16/16 through 32 tools, fan-out 12, depth 8, and 16 history turns.
- [`../../runs/code/LAB-CODE-002-BCB-HARD-2026-08-22/RESULT.md`](../../runs/code/LAB-CODE-002-BCB-HARD-2026-08-22/RESULT.md): official BigCodeBench-Hard Instruct baseline, 48/148 raw pass@1 and 48/147 after one ground-truth web failure.
- [`../../runs/context/LAB-CTX-002-RULER-V1-2026-08-22/RESULT.md`](../../runs/context/LAB-CTX-002-RULER-V1-2026-08-22/RESULT.md): all-task official RULERv1 bounded pilot at 64k/128k; 64k failed the preregistered pilot gate and exposed fragile VT/CWE/FWE, while 128k passed all 19 observed cells.
- [`../../runs/context/LAB-CTX-003-REPOBENCH-P-2026-08-22/RESULT.md`](../../runs/context/LAB-CTX-003-REPOBENCH-P-2026-08-22/RESULT.md): full 500-example cross-file completion baseline; quality gate failed at 39.56 and the 8k+ source-length slice fell to 14.10 despite every input fitting the live context.
- [`../../runs/energy/LAB-ENERGY-002-POWER-CURVE-2026-08-22/RESULT.md`](../../runs/energy/LAB-ENERGY-002-POWER-CURVE-2026-08-22/RESULT.md): qualified 420/378/336/294 W Pareto curve; lower limits saved some energy only by exceeding the frozen throughput-loss guardrail, so 420 W remains the recommendation.
- [`../../runs/close-outs/LAB-CLOSE-001-MMAP-2026-08-22/RESULT.md`](../../runs/close-outs/LAB-CLOSE-001-MMAP-2026-08-22/RESULT.md): the historical no-mmap decode penalty was confounded; decode was noise-equivalent, while fresh-process time and cold-page behavior favored no-mmap for the exact Qwen3.6 MoE/ncmoe=6 profile.
- [`../../runs/close-outs/LAB-CLOSE-002-FABLE-TERMINATION-2026-08-22/RESULT.md`](../../runs/close-outs/LAB-CLOSE-002-FABLE-TERMINATION-2026-08-22/RESULT.md): 32-cell termination qualification; instruct stopped naturally 8/8, but thinking stopped only 6/16 and neither larger budgets nor explicit stops made it agent-safe.
- [`../../runs/ops/LAB-OPS-001-MODE-LOCK-2026-08-22/RESULT.md`](../../runs/ops/LAB-OPS-001-MODE-LOCK-2026-08-22/RESULT.md): qualified fail-closed SERVE/LAB state in `lmctl`; atomic audited transitions prevent canonical and experimental text servers from overlapping while leaving embedding 8081 independent.
- [`../../runs/ops/LAB-OPS-002-INTERFERENCE-2026-08-22/RESULT.md`](../../runs/ops/LAB-OPS-002-INTERFERENCE-2026-08-22/RESULT.md): 15-cell controlled colocation matrix; CPU/RAM/disk stayed below the 10% threshold, while GPU compute raised gross prefill energy per token 54% and was material.
- [`../../runs/provenance/LAB-PROV-002-REQUANT-2026-08-22/RESULT.md`](../../runs/provenance/LAB-PROV-002-REQUANT-2026-08-22/RESULT.md): official Qwen3.8 source-to-IQ4_XS lineage closed with 18/18 hashes and a pinned local build; the authorial quant failed parity on size, VRAM, termination and deterministic-output drift, so the Unsloth UD artifact remains preferred.
- [`../../runs/requalification/RWKV7-1.5B-20260805-2026-08-22/RESULT.md`](../../runs/requalification/RWKV7-1.5B-20260805-2026-08-22/RESULT.md): official RWKV7 1.5B recurrent mechanism qualified on the RTX 3090 with exact cached-continuation parity and constant 12.8 MB state; first-use runtime is immature and the unasserted weight license blocks deployment.
- [`../../runs/vlm/LAB-VLM-001-2026-08-22/RESULT.md`](../../runs/vlm/LAB-VLM-001-2026-08-22/RESULT.md): visual-coding expansion for the resident Gemma-4-12B Vision profile; 4/4 stack-trace, UI-overflow, visual-diff and terminal-failure cases passed all 20 frozen clauses.
- [`../../runs/requalification/FALCON-H1R-7B-2026-08-22/RESULT.md`](../../runs/requalification/FALCON-H1R-7B-2026-08-22/RESULT.md): official hybrid Transformer+Mamba2 Q8 compact screen; fit and passed smoke/tools/GSM, but repeated empty 2k/4k coding completions forced `HOLD_ROLE` before context expansion.
- [`../../runs/requalification/QWEN38-HAUHAUCS-AGGRESSIVE-2026-08-23/RESULT.md`](../../runs/requalification/QWEN38-HAUHAUCS-AGGRESSIVE-2026-08-23/RESULT.md): revision-bound HauhauCS Qwen3.8 candidate; 56/60 HumanEval+, 44/44 benign comply, 131k fit, 72k needle pass, and 91.37 tok/s native-MTP median; qualified but not the broad default.
- [`../../runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/RESULT.md`](../../runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/RESULT.md): deterministic ordinary-question comparison against vanilla Qwen3.8 and Fable-TC; raw candidate deficit localized to Portuguese-language adherence.
- [`../../runs/requalification/QWEN38-HAUHAUCS-LOCALE-CONTROL-2026-08-23/RESULT.md`](../../runs/requalification/QWEN38-HAUHAUCS-LOCALE-CONTROL-2026-08-23/RESULT.md): frozen locale contract closed the PT-BR gap at 48/48 for both HauhauCS and Fable on blind test; no weight edit was justified.
- [`../../runs/image/LAB-IMG-001-QWEN-IMAGE-2026-08-22/RESULT.md`](../../runs/image/LAB-IMG-001-QWEN-IMAGE-2026-08-22/RESULT.md): official Qwen-Image NF4/offload 3090 qualification; fit and deterministic replay passed, while the frozen semantic panel held quality at 10/13.
- [`../../runs/image/LAB-IMG-002-SDXL-2026-08-22/RESULT.md`](../../runs/image/LAB-IMG-002-SDXL-2026-08-22/RESULT.md): matched SDXL FP16 baseline; much faster but rejected at 3/13 semantic clauses.
- [`../../runs/agent-product/LAB-HARNESS-001-2026-08-22/RESULT.md`](../../runs/agent-product/LAB-HARNESS-001-2026-08-22/RESULT.md): digest-bound task/evidence/baseline primitives with 83.85% model-token context reduction.
- [`../../runs/agent-product/LAB-HARNESS-003-CRITIC-2026-08-22/RESULT.md`](../../runs/agent-product/LAB-HARNESS-003-CRITIC-2026-08-22/RESULT.md): deterministic maintainability gate and independent critic, 8/8 frozen classifications with zero unsafe accepts.
- [`../../runs/a2/stage2-2026-08-22/RESULT.md`](../../runs/a2/stage2-2026-08-22/RESULT.md): optional A2 Stage-2 direction search; 0/44 candidates passed induction/KL, so G0 killed all weight-edit and merge descendants.
