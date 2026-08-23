# slop.cpp ownership-boundary qualification recheck

**Date:** 2026-08-23
**Purpose:** Validate the engine-owned replacement for the duplicated
`bless_fork.sh` before making local-labs a compatibility-only caller. This was a
single qualification run, not a soak.

## Exact tuple

- Harness source: `C:/projects/slop.cpp`, branch
  `docs/reconcile-local-labs-boundary` (uncommitted candidate at run time).
- Engine binary: `/home/augus/src/slop.cpp-main/build/bin/llama-server`.
- Binary-reported version: build `10159` (`068764d92`).
- Binary SHA-256:
  `5719c246ec3622ea1df3c3f498075879f12f1f70b969f8b591e87b3a1f3c8808`.
- Model: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
- Model SHA-256:
  `0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b`.
- Placement: `--n-cpu-moe 8`; test port: `8097`.
- Artifact directory during the run: `/tmp/slop-boundary-bless`.

## Command

```bash
cd /mnt/c/projects/slop.cpp
SLOP_BIN=/home/augus/src/slop.cpp-main/build/bin/llama-server \
SLOP_MODEL=/home/augus/models/qwen36-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
SLOP_ARTIFACT_DIR=/tmp/slop-boundary-bless \
SLOP_PORT=8097 \
bash tools/scripts_sh/bless_fork.sh -- --n-cpu-moe 8
```

## Result

| Gate | Observation | Verdict |
|---|---|---|
| G1 B2b | 20 `CUDA_Host(B2b)` allocation records | PASS |
| G2 MTP identity | base 95.108 tok/s; MTP 142.122 tok/s; 194/239 drafts accepted; 952 chars in each mode; `IDENTICAL=True` | PASS |
| G3 GPU KV | 830 chars; longest run 3; unique-character ratio 0.06 | PASS |
| G3 host KV | 839 chars; longest run 3; unique-character ratio 0.06 | PASS |

**Overall: 3/3 PASS.** The process exited `0`. The harness launched and stopped
only its own server children; the pre-existing inference service was inactive.

## Interpretation

This validates the relocated harness and rechecks the historical binary/model/
placement tuple. It is not evidence that a newer `slop.cpp` binary, another
model, or another placement has been qualified. The timing delta is diagnostic
for this one run and is not a new promotion claim.
