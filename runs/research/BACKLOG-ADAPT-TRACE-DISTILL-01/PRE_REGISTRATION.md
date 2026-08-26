# BACKLOG-ADAPT-TRACE-DISTILL-01 preregistration

Task: Reopen ThinkingCap trace distillation only after a behavioral finalist
Evidence class: `distillation`

## Hypothesis

Following the independent verification and promotion of the trained behavioral finalist (`target_mlp_only` LoRA MLP from `BACKLOG-ADAPT-TRAIN-01`), evaluating a student model adapted with teacher reasoning traces from `runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json` on `Qwen/Qwen3.5-0.8B-Base` across 32 paired held-out GSM8K tasks will achieve positive mathematical reasoning accuracy gain over the promoted behavioral finalist baseline (heldout_gain_over_finalist > 0) while maintaining protected ordinary-QA retention with regression no greater than 5% (protected_regression <= 0.05).

## Frozen inputs

Source artifacts and historical ledgers:
- `runs/research/ADAPT-00C-BEHAVIORAL-2026-08-24/RESULT.md` (2776 bytes, SHA-256: `c704ec68d6319e61d247d9da1d46fd86e064b362c024fd8479fbe08473ce81f0`)
- `docs/research/REMAINING_EXPERIMENTS_2026-08-24.md` (3342 bytes, SHA-256: `6e9be60052d2ad54f7f538017cf9511eebb620fab1c089b7037e05623286c068`)
- `runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json` (634971 bytes, SHA-256: `dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e`)
- `workloads/gsm8k.jsonl` (389701 bytes, SHA-256: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`)
- `runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl` (11016 bytes, SHA-256: `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`)

Model and adapter weights:
- Base weights: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/model.safetensors-00001-of-00001.safetensors` (1746942600 bytes, SHA-256: `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`)
- Base config: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/config.json` (2907 bytes, SHA-256: `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`)
- Base tokenizer: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/tokenizer.json` (12807196 bytes, SHA-256: `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`)
- Promoted finalist adapter: `runs/research/BACKLOG-ADAPT-TRAIN-01/raw/checkpoint_seed_20260824` / `runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw/target_mlp_only/adapter`

## Command

```powershell
python tools/research/run_trace_distillation_evaluation.py --outdir runs/research/BACKLOG-ADAPT-TRACE-DISTILL-01
```

## Factors

- Promoted behavioral finalist baseline: `target_mlp_only` (15/32 on GSM8K in requalification, 14/32 in fresh training).
- Trace-distilled student arm: `Qwen/Qwen3.5-0.8B-Base` loaded with trace-distilled adapter trained on 128 teacher traces from `ThinkingCap-27B-Q4`.
- Math evaluation panel: 32 frozen disjoint GSM8K task IDs (`gsm8k/392`, ..., `gsm8k/386`).
- Protected QA panel: 16 tasks (`f01`, ..., `s02`).
- Decoding contract: Greedy decoding (`do_sample=False`, `temperature=0.0`), seed=20260824, max_new_tokens=192 for math, 128 for QA.
- Hardware / Runtime: NVIDIA GeForce RTX 3090 (24.5GB VRAM), WSL2 Ubuntu-24.04, Python virtual environment `/home/augus/.venvs/adapt00-20260824` (PyTorch 2.5.1+cu124, Transformers 5.15.1, PEFT 0.20.0).

## Acceptance gates

- `behavioral_finalist`: `promoted_behavioral_finalist_present eq True`
- `paired_traces`: `paired_teacher_student_traces ge 32`
- `heldout_gain`: `heldout_gain_over_finalist gt 0`
- `protected_regression`: `protected_regression le 0.05`

## Abort conditions

- Absence of a promoted behavioral finalist.
- Dataset, base model, or adapter SHA-256 mismatch.
- Scorer mismatch between generation extractions and independent evaluation.
- Out of memory (OOM) or CUDA execution failure.

## Allowed claims

- `TRACE_DISTILLATION_QUALIFIED`
- `TRACE_DISTILLATION_REJECTED`

Claims outside these codes are forbidden even if a metric looks favorable.
