# SLX-09B 2:4 structural-mask requalification — preregistration

Executor: Codex  
Predecessor: `SLX-09` (`UNVERIFIED_PRELIMINARY`)  
Evidence class: real Qwen3.5 weights and forward passes, dense tensors with a
2-of-4 zero mask; no sparse packing and no throughput claim.

## Frozen identity and command

- Repository HEAD: `8bb0197d4a280aafb20e118db8ff5a7fc21d0631`.
- Probe SHA-256: `b63b3ad8e2349a8bb343f63c5b5410cc3502c4dbe1305b317786a208eacf8242`.
- Provenance helper SHA-256: `230ae7266707f4469f9add33ca8a4e0ed9148760e8a85d81598915cc931d9fb9`.
- Model revision: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- Command:

```powershell
wsl -d Ubuntu-24.04 -- /home/augus/.venvs/adapt00-20260824/bin/python `
  /mnt/c/projects/tare.tools.local-labs/tools/probes/slx09_sparsity_oracle.py `
  --model-path /home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe `
  --model-revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 `
  --output runs/research/SLX-09B-SPARSITY-24-2026-08-25/raw/receipt.json
```

## Frozen scope and gates

This is an exact rerun of the narrow single-sentence calibration probe, now
with complete artifact and environment provenance. Gates remain: Wanda logit
cosine similarity at least 0.90, exact 2:4 conformity, and at least 20% MSE
improvement over magnitude pruning.

A pass would only admit a broader multi-corpus quality evaluation and a packed
sparse-kernel benchmark. It would not establish Ampere acceleration. A failure
rejects this zero-shot mask under the frozen calibration input only.
