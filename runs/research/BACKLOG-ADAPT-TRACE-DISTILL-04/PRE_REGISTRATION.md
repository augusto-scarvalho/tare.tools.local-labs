# BACKLOG-ADAPT-TRACE-DISTILL-04 preregistration

Task: Confirm trace distillation with seven paired seeds, longer training, and broad held-out panels  
Evidence class: `distillation`  
Executor: Codex executor  
Date: 2026-08-26

## Hypothesis

For fresh LoRA MLP adapters on `Qwen/Qwen3.5-0.8B-Base`, completion-only SFT on complete correct teacher traces produces a positive and repeatable held-out GSM8K accuracy gain over a matched answer-only SFT control when both arms receive the same examples, order, optimizer budget, seed, and decoding configuration.

This is a confirmatory successor to the small R3 screen. R3 used three seeds, 128 optimizer steps, 32 held-out math prompts, and 16 protected-QA prompts. R4 increases the independent training replications to seven, uses all 168 eligible teacher examples for three epochs (504 optimizer steps per arm), and evaluates 256 untouched GSM8K prompts plus all 48 protected-QA prompts per arm and seed.

## Frozen inputs

- Admission: `config/research_backlog_admissions/BACKLOG-ADAPT-TRACE-DISTILL-04.json`, 2,779 bytes, SHA-256 `2d78cd4eee66dc6ea3f2b0f20467794098ba39b503b4d888247da6c61c1bc676`.
- Teacher generations: `runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json`, 634,971 bytes, SHA-256 `dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e`.
- GSM8K corpus: `workloads/gsm8k.jsonl`, 389,701 bytes, SHA-256 `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`.
- Protected QA corpus: `runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl`, 11,016 bytes, SHA-256 `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`.
- R3 receipt: `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/receipt.json`, 15,477 bytes, SHA-256 `2d157b63a1b342f6b5c9c9f7f075bd550404a76b03787f4893ef00d585a5f23d`.
- Base model: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe`; weights SHA-256 `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`, config SHA-256 `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`, tokenizer SHA-256 `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`.

The 168 eligible training examples are the successful nonempty teacher completions among `gsm8k/0` through `gsm8k/199`, excluding the 32 historical frozen IDs. The held-out math panel is the deterministic contiguous set `gsm8k/200` through `gsm8k/455`, which is disjoint from all teacher/training examples. The protected panel is all 48 rows of the frozen QA corpus.

## Command

```powershell
python tools/research/run_trace_distillation_confirmation_r4.py --outdir runs/research/BACKLOG-ADAPT-TRACE-DISTILL-04
```

## Factors

- Arms: `answer_only` (`#### <gold>`) and `full_trace` (complete correct teacher completion).
- Seeds: `20260830` through `20260836`, seven independent paired replications.
- Pairing: within a seed, both arms receive the identical shuffled task sequence and all stochastic seeds are identical; arm order alternates by seed to limit temporal bias.
- Training: fresh base load per worker; LoRA on `gate_proj`, `up_proj`, `down_proj`, rank 8, alpha 16, dropout 0; AdamW, learning rate `1e-4`, weight decay `0.01`; bfloat16; completion-only loss; maximum sequence length 512; three epochs over 168 examples, exactly 504 optimizer steps.
- Evaluation: greedy decoding on 256 untouched GSM8K prompts with at most 192 new tokens and all 48 protected-QA prompts with at most 128 new tokens.
- Primary estimand: mean paired trace-minus-answer math accuracy across seeds.
- Uncertainty: deterministic hierarchical paired bootstrap with 20,000 replicates, resampling seeds and then prompts within resampled seeds. The 2.5th and 97.5th percentiles form the reported 95% interval.
- Secondary diagnostics: per-seed deltas, pooled discordant-pair counts, exact one-sided sign-flip p-value over the seven seed deltas, token counts, natural EOS, loss curves, and protected-QA regression.

## Acceptance gates

- `treatment_materiality`: `matched_distinct_training_targets_verified eq True`
- `training_budget`: `training_steps_per_arm_per_seed eq 504`
- `seed_coverage`: `completed_paired_seeds eq 7`
- `heldout_coverage`: `heldout_math_samples_per_arm_per_seed eq 256`
- `confirmed_gain`: `hierarchical_bootstrap_95ci_lower_trace_math_gain gt 0.0`
- `directional_repeatability`: `seeds_with_positive_trace_math_gain ge 5`
- `protected_retention`: `mean_protected_qa_regression_vs_answer_only le 0.05`
- `service_recovery`: `service_and_embedding_restored eq True`

Every gate is evaluated mechanically, but a failed scientific gate is a valid result, not an execution failure. The main interpretation will include effect sizes and uncertainty rather than only a pass/fail label.

## Abort conditions

- Any frozen input or base-model hash differs before launch.
- The 168-example training pool or 256-example held-out panel is not unique and disjoint.
- Paired arms within a seed receive different task order, optimizer budget, or decoding settings.
- A worker starts with a pre-existing PEFT module, diverges, exits nonzero, or produces fewer than 304 evaluation samples.
- CUDA out-of-memory persists after stopping the managed inference service.
- The embedding endpoint on port 8081 becomes unhealthy.
- The initial serving route cannot be restored exactly after training.

## Allowed claims

- `TRACE_DISTILLATION_CONFIRMED_R4`
- `TRACE_DISTILLATION_NOT_CONFIRMED_R4`

No teacher-level noninferiority, production promotion, untested mechanism attribution, or generalization beyond this model, data, and budget is allowed. The executor stops at `EXECUTED`; an independent reviewer must recompute the receipt before any terminal transition.
