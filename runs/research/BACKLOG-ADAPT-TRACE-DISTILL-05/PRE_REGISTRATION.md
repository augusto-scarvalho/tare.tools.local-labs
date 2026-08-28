# BACKLOG-ADAPT-TRACE-DISTILL-05 preregistration

Task: Confirm trace distillation with seven paired seeds and a teacher-disjoint broad held-out panel  
Evidence class: `distillation`  
Executor: Codex executor  
Date: 2026-08-26

## Hypothesis

For fresh LoRA MLP adapters on `Qwen/Qwen3.5-0.8B-Base`, completion-only SFT on complete correct teacher traces produces a positive and repeatable held-out GSM8K accuracy gain over a matched answer-only SFT control when both arms receive the same examples, order, optimizer budget, seed, and decoding configuration.

R3 was only a three-seed, 128-step, 32-question screen. This confirmatory run uses seven independent paired seeds, all 168 eligible teacher examples for three epochs (504 steps per arm), 256 teacher-disjoint GSM8K questions, and all 48 protected-QA questions. R4 was stopped before execution because its proposed held-out interval overlapped the noncontiguous teacher sample.

## Frozen inputs

- Admission: `config/research_backlog_admissions/BACKLOG-ADAPT-TRACE-DISTILL-05.json`, 2,558 bytes, SHA-256 `83b748014a470a4e6f88409a38f6e87538ae7e7ee5a40b29e89f94525ead96b7`.
- Teacher generations: `runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json`, SHA-256 `dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e`.
- GSM8K corpus: `workloads/gsm8k.jsonl`, SHA-256 `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`.
- Protected QA corpus: `runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl`, SHA-256 `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`.
- Fail-closed R4 preregistration: `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-04/PRE_REGISTRATION.md`, 5,689 bytes, SHA-256 `725e178bf4349298f70f6183971f268dc8f4dae6b8199a96a77e703a535eb9f4`.
- Base model: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe`; weights SHA-256 `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`, config SHA-256 `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`, tokenizer SHA-256 `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`.

The training pool is the same 168 successful nonempty teacher completions used by R3 after excluding its historical 32-item evaluation panel. The R5 held-out panel is computed before training as the 256 lowest numeric GSM8K task IDs absent from the complete 200-task teacher artifact. Its canonical compact-JSON ID-list SHA-256 is `78a3b7ef26cdf932b79eb6f64dfb576d66770d30a5ce0fd3251536cb6e76901f`. This makes evaluation disjoint from every answer-only and full-trace training source, not merely from the selected 168.

## Command

```powershell
python tools/research/run_trace_distillation_confirmation_r5.py --outdir runs/research/BACKLOG-ADAPT-TRACE-DISTILL-05
```

## Factors

- Arms: `answer_only` (`#### <gold>`) and `full_trace` (complete correct teacher completion).
- Seeds: `20260830` through `20260836`; arm execution order alternates by seed.
- Pairing: both arms within each seed receive identical epoch orders and stochastic seeds.
- Training: fresh base per worker; MLP LoRA rank 8, alpha 16, dropout 0; AdamW `1e-4`, weight decay `0.01`; bfloat16; completion-only loss; maximum sequence length 512; three deterministic reshuffles of 168 examples, exactly 504 optimizer steps.
- Evaluation: greedy decoding; 256 teacher-disjoint GSM8K prompts at 192 maximum new tokens and all 48 protected-QA prompts at 128 maximum new tokens.
- Primary estimand: mean paired trace-minus-answer math accuracy across seven seeds.
- Uncertainty: deterministic hierarchical paired bootstrap with 20,000 replicates, resampling seeds and prompts, reported as a percentile 95% interval.
- Diagnostics: per-seed effects, pooled discordant pairs, exact one-sided seven-seed sign-flip test, response lengths, EOS behavior, and protected-QA regression.

## Acceptance gates

- `treatment_materiality`: `matched_distinct_training_targets_verified eq True`
- `training_budget`: `training_steps_per_arm_per_seed eq 504`
- `seed_coverage`: `completed_paired_seeds eq 7`
- `heldout_coverage`: `heldout_math_samples_per_arm_per_seed eq 256`
- `confirmed_gain`: `hierarchical_bootstrap_95ci_lower_trace_math_gain gt 0.0`
- `directional_repeatability`: `seeds_with_positive_trace_math_gain ge 5`
- `protected_retention`: `mean_protected_qa_regression_vs_answer_only le 0.05`
- `service_recovery`: `service_and_embedding_restored eq True`

Failed scientific gates remain valid results. Interpretation must report effects and uncertainty, not merely gate labels.

## Abort conditions

- Any frozen input, model, or held-out ID-list hash differs.
- Training and held-out IDs overlap, are duplicated, or have unexpected cardinality.
- Paired arms differ in task order, optimizer budget, or decoding configuration.
- A worker starts with PEFT modules, diverges, exits nonzero, or returns fewer than 304 evaluation samples.
- CUDA OOM persists after managed serving shutdown, port 8081 becomes unhealthy, or the initial serving route cannot be restored.

## Allowed claims

- `TRACE_DISTILLATION_CONFIRMED_R5`
- `TRACE_DISTILLATION_NOT_CONFIRMED_R5`

No teacher-level noninferiority, production promotion, unisolated mechanism attribution, or generalization beyond this model, data, and budget is permitted. The executor stops at `EXECUTED`; independent review is required for a terminal transition.
