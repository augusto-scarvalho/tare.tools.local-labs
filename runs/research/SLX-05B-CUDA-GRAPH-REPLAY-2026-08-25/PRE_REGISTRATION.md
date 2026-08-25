# SLX-05B fixed-cache CUDA Graph replay — preregistration

Executor: Codex  
Predecessor: `SLX-05` (`SUPERSEDED_INVALID_METRIC`)  
Model: `Qwen/Qwen3.5-0.8B-Base`, revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`

## Frozen identity and command

- Repository HEAD before run: `8bb0197d4a280aafb20e118db8ff5a7fc21d0631`.
- Probe SHA-256: `a61ce54647b06db27682c71ecc48899f7e93df63ae241783fe24d1b5300aa410`.
- Provenance helper SHA-256: `230ae7266707f4469f9add33ca8a4e0ed9148760e8a85d81598915cc931d9fb9`.
- Model config SHA-256: `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`.
- Tokenizer SHA-256: `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`.
- Full weight SHA-256 will be captured in the receipt before it can pass provenance.
- Command:

```powershell
wsl -d Ubuntu-24.04 -- /home/augus/.venvs/adapt00-20260824/bin/python `
  /mnt/c/projects/tare.tools.local-labs/tools/probes/slx05_launch_oracle.py `
  --model-path /home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe `
  --model-revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 `
  --iterations 100 `
  --output runs/research/SLX-05B-CUDA-GRAPH-REPLAY-2026-08-25/raw/receipt.json
```

## Factors and gates

Cells: batch/context `(1,128)`, `(1,512)`, `(1,2048)`, `(2,512)`, `(4,512)`. Each cell uses a `StaticCache` and overwrites one frozen decode position; no cache may grow across iterations. Eager and graph measurements use 100 paired fixed-shape observations after warmup.

All must pass: CUDA Graph capture in every cell; maximum absolute logit difference no greater than `1e-2`; median batch-1 wall speedup at least `1.15x`; fixed static cache position recorded; full provenance complete.

The result may qualify CUDA Graph replay for this tuple. It must not be described as exclusive driver-launch overhead or as a persistent-megakernel ceiling.
