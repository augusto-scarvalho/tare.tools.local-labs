# BACKLOG-DISTILL-REAL-01 preregistration

Task: Rebuild DISTILL-00 from actual teacher and student generations
Evidence class: `distillation`

## Hypothesis

Evaluating paired reasoning generations from the teacher (`ThinkingCap 27B / Fable-TC` teacher traces) and concise distilled student (`Qwen/Qwen3.5-0.8B-Base` with verified LoRA MLP adapter) on the frozen 32-sample GSM8K math panel will demonstrate non-inferior accuracy (student accuracy delta >= -0.03 versus teacher) while achieving at least a 20% reduction in median reasoning tokens (median_reasoning_token_reduction >= 0.20), replacing superseded synthetic random metrics with verifiable, deterministic raw sample evidence.

## Frozen inputs

Source artifacts and datasets:
- `runs/research/DISTILL-00-MOE-CONCISE-2026-08-25/RESULT.md` (2182 bytes, SHA-256: `98cdf946d8916fc566c816055607e1a7f1e75574f571b43aab4f3bb4d4a3eb0b`)
- `runs/research/GEMINI-BACKLOG-REMEDIATION-2026-08-25/RESULT.md` (3417 bytes, SHA-256: `6ab01e6de1ec5cdd93974c35e9db4f45ac87cb9bbdb300db9759082c18e15196`)
- `runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json` (634971 bytes, SHA-256: `dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e`)
- `workloads/gsm8k.jsonl` (389701 bytes, SHA-256: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`)

Model weights and configurations:
- Base weights: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/model.safetensors-00001-of-00001.safetensors` (1746942600 bytes, SHA-256: `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`)
- Base config: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/config.json` (2907 bytes, SHA-256: `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`)
- Base tokenizer: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/tokenizer.json` (12807196 bytes, SHA-256: `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`)
- Student adapter: `runs/research/BACKLOG-ADAPT-TRAIN-01/raw/checkpoint_seed_20260824` / `runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw/target_mlp_only/adapter`

## Command

```powershell
python tools/research/run_distillation_evaluation.py --outdir runs/research/BACKLOG-DISTILL-REAL-01
```

## Factors

- Teacher arm: Fable-TC / ThinkingCap 27B teacher completions evaluated on the frozen 32 math tasks.
- Student arm: `Qwen/Qwen3.5-0.8B-Base` loaded with verified concise LoRA MLP adapter.
- Task panel: 32 frozen disjoint GSM8K task IDs (`gsm8k/392`, `gsm8k/1226`, ..., `gsm8k/386`).
- Decoding contract: Greedy decoding (`do_sample=False`, `temperature=0.0`), seed=20260824, max_new_tokens=192.
- Hardware / Runtime: NVIDIA GeForce RTX 3090 (24.5GB VRAM), WSL2 Ubuntu-24.04, Python virtual environment `/home/augus/.venvs/adapt00-20260824` (PyTorch 2.5.1+cu124, Transformers 5.15.1, PEFT 0.20.0).

## Acceptance gates

- `no_fabricated_metrics`: `scores_derived_from_raw_samples eq True`
- `paired_panel`: `paired_scored_samples ge 32`
- `accuracy_noninferiority`: `student_accuracy_delta ge -0.03`
- `token_reduction`: `median_reasoning_token_reduction ge 0.2`

## Abort conditions

- Dataset, model weight, or configuration SHA-256 identity mismatch.
- Missing paired samples in the evaluation denominator.
- Scorer discrepancy between raw sample extraction and independent evaluation.
- Out of memory (OOM) or CUDA hardware execution faults.

## Allowed claims

- `DISTILLATION_QUALIFIED`
- `DISTILLATION_REJECTED`

Claims outside these codes are forbidden even if a metric looks favorable.
