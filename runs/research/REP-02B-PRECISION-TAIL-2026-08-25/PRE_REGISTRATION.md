# REP-02B precision-tail corrected-decode simulation — preregistration

Executor: Codex  
Predecessor: `REP-02` (`SUPERSEDED_INVALID_COMPARATOR`)  
Evidence class: real Qwen3.5 forward passes with simulated INT4
quantize/dequantize; no packed kernel and no measured VRAM saving.

## Frozen identity and command

- Repository HEAD: `8bb0197d4a280aafb20e118db8ff5a7fc21d0631`.
- Probe SHA-256: `dc7f65511d10e0f9f00ed5bae0e184005d7b550a147f223ffb0752dad31fc314`.
- Provenance helper SHA-256: `230ae7266707f4469f9add33ca8a4e0ed9148760e8a85d81598915cc931d9fb9`.
- Model revision: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- Config, tokenizer and model-weight hashes must match the SLX-05D frozen
  identities and be present in the receipt.
- Command:

```powershell
wsl -d Ubuntu-24.04 -- /home/augus/.venvs/adapt00-20260824/bin/python `
  /mnt/c/projects/tare.tools.local-labs/tools/probes/rep02_precision_tail.py `
  --model-path /home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe `
  --model-revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 `
  --output runs/research/REP-02B-PRECISION-TAIL-2026-08-25/raw/receipt.json
```

## Corrected comparator and gates

At each context length, all policies receive the same fixed next-token ID and
an independent clone of the same post-prefill cache. The BF16 control is the
logit vector from that same next-token decode, not the last prefill position.

Frozen contexts are 256, 1024 and 4096. Promotion requires all original gates:
at least 50% MSE reduction for precision-tail-64 versus uniform simulated INT4
at 4096; needle retrieval; and at least 65% *analytical packed-storage*
estimate. Even on pass, the result can only qualify a codec/kernel candidate;
it cannot claim realized compression or VRAM savings.
