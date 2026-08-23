# Model disk cleanup — 2026-08-23

The user authorized removal of local model artifacts that no longer have an active role. The cleanup was
limited to exact paths under `/home/augus/models`; no experiment receipts were removed. Before deletion,
the non-soak backlog was reconciled, the text service was confirmed inactive, and the only running
`llama-server` was the protected embedding endpoint on port 8081.

## Result

| Metric | Before | After |
|---|---:|---:|
| WSL filesystem used | 875,294,363,648 bytes (86%) | 637,127,208,960 bytes (63%) |
| WSL filesystem available | 150,814,457,856 bytes | 388,981,612,544 bytes |
| `/home/augus/models` | — | 503 GiB |

The deleted paths occupied 238,166,467,040 bytes (238.17 GB, 221.81 GiB). The filesystem reported
238,167,154,688 additional free bytes after deletion. These files are not locally recoverable; upstream
models can be downloaded again, and authorial artifacts can be rebuilt from the retained source, hash,
revision, quantizer, imatrix, and decision receipts in the repository.

## Removed

| Exact target | Bytes | Reason |
|---|---:|---|
| `qwen38-27b/unsloth-4ca72078/` | 25,451,428,649 | Both revision candidates failed supersession; the separate MTP draft in the same candidate directory had no retained role. |
| `qwen38-27b/cold-fusion-27a5cb2c/` | 17,033,680,701 | Base role and descriptive embedded-MTP arm were both rejected. |
| `qwen38-27b/official-1d4bf0f2/` | 55,586,126,178 | Downloaded BF16 source for the closed authorial requant probe; exact source revision and verification receipts remain. |
| `qwen38-27b/requant-87a416bd/` | 68,904,732,864 | Intermediate BF16 conversion and authorial IQ4_XS output; provenance closed, behavioral parity rejected. |
| `qwen38-27b/bartowski/` | 17,772,547,858 | Redundant community Q4_K_M; the qualified Unsloth Q4/IQ4 artifacts remain. |
| `engine-parity-qwen3-4b/` | 16,112,215,750 | One-shot cross-engine comparison is complete and reproducible from its pinned official revision. |
| `qwen38-27b/unsloth/Qwen3.8-27B-UD-Q3_K_XL.gguf` | 13,441,059,904 | Quant-frontier intermediate with no residual decision role. |
| `qwen38-27b/unsloth/Qwen3.8-27B-UD-IQ3_XXS.gguf` | 11,913,559,104 | Quant-frontier intermediate with no residual decision role. |
| `qwen38-27b/unsloth/Qwen3.8-27B-UD-IQ2_M.gguf` | 10,319,907,904 | Measured long-context cliff; not retained as the low-footprint profile. |
| `muse-glimmer-30b/meta-70bf1b61/dflash-Muse-Glimmer-30B-Q4_K_M.gguf` | 1,631,208,128 | DFlash changed greedy output and failed the full-stack reserve/equivalence decision. |

## Deliberately retained

- Canonical Qwen3.8 `UD-Q4_K_XL`, qualified `IQ4_XS`, and the measured low-footprint `UD-Q2_K_XL`.
- Embedding weights and the healthy port-8081 process.
- Muse text model plus `mmproj`, preserving its qualified VQA-specialist role.
- Qwen-Image and SDXL, preserving the research candidate and high-speed comparison baseline.
- The FP16 parents and promoted `fable-tc-l1.0` authorial merge, plus the resident HOLD/research fleet.
- All immutable experiment, provenance, and decision receipts in Git.

## Verification

- Every deletion target was resolved to its literal absolute path below `/home/augus/models` before removal.
- All ten targets were absent afterward; every named protected artifact was present.
- `llm-inference.service` remained inactive.
- `http://127.0.0.1:8081/health` returned `{"status":"ok"}` after deletion.
- Current host `C:` free space after deletion was 254,950,301,696 bytes (237.44 GiB).

