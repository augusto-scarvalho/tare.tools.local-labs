# BACKLOG-ADAPT-TRAIN-01 preregistration

Task: Reproduce training only for adapter finalists
Evidence class: `model_training`

## Hypothesis

Re-running PEFT LoRA training on the promoted `target_mlp_only` finalist geometry (`gate_proj`, `up_proj`, `down_proj`, r=8, alpha=16) using teacher traces (`runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json`) across two distinct random seeds (20260824 and 20260825) on `Qwen/Qwen3.5-0.8B-Base` in clean, isolated output directories will reproduce training loss convergence, achieve positive held-out accuracy gain over the base model on the 32-sample math panel (heldout_gain_over_base > 0), and preserve protected ordinary QA performance without regression exceeding 5% (protected_regression <= 0.05).

## Frozen inputs

Source artifacts and datasets:
- `runs/research/TRAIN-00B-GALORE-3090-2026-08-25/RESULT.md` (632 bytes, SHA-256: `1a24d8a8f34e91330fdbd06f3aef3425105c49a109601310c53e673815151a75`)
- `runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json` (634971 bytes, SHA-256: `dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e`)
- `workloads/gsm8k.jsonl` (389701 bytes, SHA-256: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`)
- `runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl` (11016 bytes, SHA-256: `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`)

Base model weights and config:
- Weights: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/model.safetensors-00001-of-00001.safetensors` (1746942600 bytes, SHA-256: `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`)
- Config: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/config.json` (2907 bytes, SHA-256: `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`)
- Tokenizer: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/tokenizer.json` (12807196 bytes, SHA-256: `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`)

## Command

```powershell
python tools/research/run_adapter_training.py --outdir runs/research/BACKLOG-ADAPT-TRAIN-01
```

## Factors

- Finalist adapter geometry: LoRA MLP targeting `["gate_proj", "up_proj", "down_proj"]`, r=8, alpha=16, dropout=0.0.
- Repeated seeds: Seed 20260824 (seed 1) and Seed 20260825 (seed 2).
- Training split: 128 teacher-distilled reasoning pairs from `runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json`.
- Training parameters: AdamW optimizer, lr=2e-4, 60 training steps, max sequence length 384, bf16/fp16 mixed precision.
- Held-out evaluation panels: 32 GSM8K math problems and 16 protected ordinary-QA questions.
- Hardware / Runtime: NVIDIA GeForce RTX 3090 (24.5GB VRAM), WSL2 Ubuntu-24.04, Python virtual environment `/home/augus/.venvs/adapt00-20260824` (PyTorch 2.5.1+cu124, Transformers 5.15.1, PEFT 0.20.0).

## Acceptance gates

- `fresh_output`: `preexisting_output_files eq 0`
- `repeatability`: `successful_repeated_seeds ge 2`
- `behavioral_gain`: `heldout_gain_over_base gt 0`
- `retention`: `protected_regression le 0.05`

## Abort conditions

- Pre-existing files found in output directory prior to execution start.
- Model, teacher, or dataset SHA-256 identity mismatch.
- Loss NaN, infinity, or divergence (> 100.0).
- Out of memory (OOM) or CUDA runtime failure.
- Protected QA regression greater than 5% (0.05).

## Allowed claims

- `TRAINING_REPRODUCED`
- `TRAINING_REJECTED`

Claims outside these codes are forbidden even if a metric looks favorable.
