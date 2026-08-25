# SLX-05D inference-safe restored-state CUDA Graph replay — preregistration

Executor: Codex  
Predecessors: `SLX-05` (`SUPERSEDED_INVALID_METRIC`), `SLX-05B`
(`INVALID_IMPLEMENTATION`), `SLX-05C` (`REJECTED_OR_UNVERIFIED`)  
Model: `Qwen/Qwen3.5-0.8B-Base`, revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`

## Frozen identity and command

- Repository HEAD before run: `8bb0197d4a280aafb20e118db8ff5a7fc21d0631`.
- Probe SHA-256: `3a3792b758eb5f31d1950cd28721270cb9bc9851ff36765328c6b5c07a0b189d`.
- Provenance helper SHA-256: `230ae7266707f4469f9add33ca8a4e0ed9148760e8a85d81598915cc931d9fb9`.
- Model config SHA-256: `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`.
- Tokenizer SHA-256: `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`.
- Model weights SHA-256: `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`.
- Command:

```powershell
wsl -d Ubuntu-24.04 -- /home/augus/.venvs/adapt00-20260824/bin/python `
  /mnt/c/projects/tare.tools.local-labs/tools/probes/slx05_launch_oracle.py `
  --model-path /home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe `
  --model-revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 `
  --iterations 100 `
  --output runs/research/SLX-05D-CUDA-GRAPH-REPLAY-2026-08-25/raw/receipt.json
```

## Delta from SLX-05C

The sole implementation change is that the post-prefill cache restoration
immediately before graph capture and each replay is performed under
`torch.inference_mode()`, matching the mode in which those tensors were
created. Cache restoration remains outside every timed region.

Cells and gates remain frozen: `(1,128)`, `(1,512)`, `(1,2048)`, `(2,512)`,
`(4,512)`; 100 paired observations; graph support in all cells; max absolute
logit difference at most `1e-2`; median batch-1 wall speedup at least `1.15x`;
restored fixed cache recorded; complete provenance.

The result may qualify CUDA Graph replay only for this model/software/hardware
tuple. It must not be described as exclusive driver-launch overhead or as a
persistent-megakernel ceiling.
