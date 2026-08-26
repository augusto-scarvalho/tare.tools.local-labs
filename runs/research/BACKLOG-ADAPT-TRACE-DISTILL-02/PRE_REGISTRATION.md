# BACKLOG-ADAPT-TRACE-DISTILL-02 preregistration

Task: Test the trace-distillation rejection with matched answer-only and full-trace SFT arms  
Evidence class: `distillation`  
Executor: Codex executor  
Date: 2026-08-25

## Hypothesis

With architecture, training examples, example order, optimizer, step count and seed held constant, SFT on complete correct ThinkingCap teacher reasoning traces will produce strictly higher mean held-out GSM8K accuracy than SFT on answer-only targets. The direction must be nonnegative in at least two of three seeds and mean protected-QA regression must not exceed five percentage points.

This packet is a direct false-negative test of `BACKLOG-ADAPT-TRACE-DISTILL-01`. The predecessor did not create a trace-distilled model and compared two labels over the same PEFT-mutated object.

## Frozen inputs

- Admission specification: `config/research_backlog_admissions/BACKLOG-ADAPT-TRACE-DISTILL-02.json`, 2,869 bytes, SHA-256 `4b88fa1a699c886e6ba3f2a3654e4cb6f79aafa12e006ae7455cf20e423e2690`.
- Independent invalidation audit: `docs/AUDIT_2026-08-25_CODEX_INDEPENDENT_AGY_EXECUTION.md`, 16,057 bytes, SHA-256 `e4364456156a3c2f015306d986192792fb1aa9ae9333b63a2237ec46e3ffc11f`.
- Teacher corpus: `runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json`, 634,971 bytes, SHA-256 `dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e`.
- Math corpus: `workloads/gsm8k.jsonl`, 389,701 bytes, SHA-256 `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`.
- Protected QA: `runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl`, 11,016 bytes, SHA-256 `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`.
- Invalid predecessor teacher subset: `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-01/raw/teacher_samples.json`, 28,345 bytes, SHA-256 `9b9f86bdcfae10ccdd28a0f8a48ccf95da57b8b04cbf06c2f94cb9d6c14e8d08`.
- Prior mislabeled behavioral training trace: `runs/research/BACKLOG-ADAPT-TRAIN-01/raw/training_trace.json`, 17,450 bytes, SHA-256 `a1c21848acf5d6cf90610806db8f67a3d61acc979be52f36aff998ad37826a31`.
- Base weights: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/model.safetensors-00001-of-00001.safetensors`, SHA-256 `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`.
- Base config: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/config.json`, SHA-256 `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`.
- Base tokenizer: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/tokenizer.json`, SHA-256 `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`.
- Frozen evaluation IDs are exactly the 32 math and 16 QA IDs used by `BACKLOG-ADAPT-REQUAL-02`.

The teacher corpus contains 168 eligible correct examples after excluding all 32 held-out math IDs. No evaluation prompt may appear in training.

## Command

```powershell
python tools/research/run_trace_distillation_training_r2.py --outdir runs/research/BACKLOG-ADAPT-TRACE-DISTILL-02
```

## Factors

- Seeds: `20260824`, `20260825`, `20260826`.
- Each seed deterministically shuffles the same 168-example eligible pool and selects the first 128 examples.
- Within each seed, both arms receive the exact same 128 prompts in the exact same order.
- `answer_only` target: `#### <gold answer>`.
- `full_trace` target: the complete correct teacher completion, including reasoning and final answer.
- Worker order balances drift: seed 20260824 answer→trace; seed 20260825 trace→answer; seed 20260826 answer→trace.
- Every arm/seed is a distinct WSL Python process that loads a fresh base model and must report zero pre-existing PEFT/tuner modules.
- Adapter: LoRA MLP only, `r=8`, `alpha=16`, dropout 0, targets `gate_proj`, `up_proj`, `down_proj`.
- Optimizer: AdamW, learning rate `1e-4`, weight decay `0.01`, gradient clip `1.0`.
- Training: bfloat16, 128 optimizer steps, one selected example per step, maximum sequence length 512, completion-only loss.
- Evaluation: greedy, 192 maximum new tokens for math and 128 for QA, same frozen panels and scorers as R2.
- Hardware: RTX 3090. The inference service may be stopped through systemd for VRAM; port 8081 must remain healthy and the original serving tuple must be restored.

The primary estimand is the arithmetic mean across seeds of `(full_trace math accuracy - answer_only math accuracy)`. No adaptive hyperparameter search or seed substitution is permitted.

## Acceptance gates

- `treatment_materiality`: `matched_distinct_training_targets_verified eq True`
- `clean_base`: `fresh_base_workers eq 6`
- `paired_training`: `matched_pairs_per_arm_per_seed eq 128`
- `seed_coverage`: `completed_paired_seeds eq 3`
- `heldout_gain`: `mean_trace_math_gain_over_answer_only gt 0.0`
- `directional_repeatability`: `seeds_with_nonnegative_trace_math_gain ge 2`
- `protected_regression`: `mean_protected_qa_regression_vs_answer_only le 0.05`
- `service_recovery`: `service_and_embedding_restored eq True`

## Abort conditions

- Any frozen hash, base identity or evaluation ID differs.
- Fewer than 168 eligible teacher examples remain or any held-out ID enters a training subset.
- Within-seed prompts or order differ between treatments.
- The two treatments have identical target-sequence ledgers.
- Any worker reports a pre-existing PEFT/tuner module, exits nonzero, OOMs, diverges, saves no checkpoint or produces fewer than 48 held-out samples.
- Port 8081 becomes unhealthy or the persistent inference service cannot be restored.

## Allowed claims

- `TRACE_DISTILLATION_FALSE_NEGATIVE_CONFIRMED_R2`
- `TRACE_DISTILLATION_REJECTED_R2`

Claims outside these codes are forbidden even if a metric looks favorable.

Even if trace supervision wins, this packet cannot select a production finalist or establish teacher-level noninferiority. The executor stops at `EXECUTED` for AGY review.
