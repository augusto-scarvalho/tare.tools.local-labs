# TRAIN-00B short GaLore/LoKr/AdamW requalification — preregistration

Executor: Codex  
Predecessor: `TRAIN-00` (`UNVERIFIED_PRELIMINARY`)  
Evidence class: real 60-step training micro-bakeoff on RTX 3090; this is not a
quality or convergence study.

## Frozen identity and command

- Repository HEAD: `8bb0197d4a280aafb20e118db8ff5a7fc21d0631`.
- Probe SHA-256: `cb36745f4d7e74871b799f28274486ae461045d0a08d84bb54a008c934479f7f`.
- Provenance helper SHA-256: `230ae7266707f4469f9add33ca8a4e0ed9148760e8a85d81598915cc931d9fb9`.
- Teacher input SHA-256: `dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e`.
- Prompt input SHA-256: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`.
- Model revision: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- Seed: `20260824`; steps: `60` per arm.
- Command:

```powershell
wsl -d Ubuntu-24.04 -- /home/augus/.venvs/adapt00-20260824/bin/python `
  /mnt/c/projects/tare.tools.local-labs/tools/probes/train00_galore_bakeoff.py `
  --model-path /home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe `
  --model-revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 `
  --teacher runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json `
  --prompts workloads/gsm8k.jsonl `
  --steps 60 --seed 20260824 `
  --output runs/research/TRAIN-00B-GALORE-3090-2026-08-25/raw/receipt.json
```

## Frozen arms and gates

Arms: LoKr PEFT with AdamW, the repository's custom rank-16 GaLore optimizer,
and full AdamW. Each reloads the same base model and resets CUDA peak-memory
statistics after optimizer construction, so the current allocation and later
transients are both included in `max_memory_allocated`.

GaLore must save at least 30% peak allocated VRAM relative to full AdamW,
sustain at least 2 steps/s, and finish below its initial loss. A pass applies
only to this custom optimizer and 60-step micro-bakeoff. A failure blocks the
longer adapter wave from being justified by GaLore resource savings.
