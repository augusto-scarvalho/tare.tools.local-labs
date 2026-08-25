# SLX-05C restored-state CUDA Graph replay — preregistration

Executor: Codex  
Predecessors: `SLX-05` (`SUPERSEDED_INVALID_METRIC`), `SLX-05B`
(`INVALID_IMPLEMENTATION`)  
Model: `Qwen/Qwen3.5-0.8B-Base`, revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`

## Frozen identity and command

- Repository HEAD before run: `8bb0197d4a280aafb20e118db8ff5a7fc21d0631`.
- Probe SHA-256: `f9f6f002da79b673b52e7627e586aae925fde1821bf6f4b7c7d075cecdc131c`.
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
  --output runs/research/SLX-05C-CUDA-GRAPH-REPLAY-2026-08-25/raw/receipt.json
```

## Factors and cache invariant

Cells: batch/context `(1,128)`, `(1,512)`, `(1,2048)`, `(2,512)`,
`(4,512)`. After one prefill, the probe snapshots full-attention key/value
tensors and counters plus all linear-attention convolutional and recurrent
states. It restores this snapshot, without changing tensor addresses, before
every eager observation, capture, and replay. Restoration occurs outside the
timed region. Thus all measurements represent the same fixed one-token decode
from the same post-prefill state.

Eager and graph measurements use 100 paired fixed-shape observations after
warmup. All must pass: CUDA Graph capture in every cell; maximum absolute logit
difference no greater than `1e-2`; median batch-1 wall speedup at least `1.15x`;
restored fixed cache recorded; full provenance complete.

The result may qualify CUDA Graph replay only for this model/software/hardware
tuple. It must not be described as exclusive driver-launch overhead or as a
persistent-megakernel ceiling.
