# LIGER-PAPER-SNAPSHOT-2026-08-20 — pre-registration

**Status:** PRE-REGISTERED before cloning the historical snapshot, creating its
environment, constructing models, or executing tensors.

## Why this is a distinct campaign

Current upstream `main` combines a Qwen-era source revision and unpinned
Transformers. It failed both the Qwen3 state-transfer gate and the original Llama
construction gate under 4.52.4. Repository history identifies a coherent paper
snapshot from 2025-03-20 whose README links the paper and released weights and
whose requirements still pin Transformers 4.47.1.

This campaign tests that historical substrate rather than patching current code.

## Frozen source and environment

- Linearization snapshot:
  `0e3cfae33a700fa5f644cf5752d8434c6afc2412`
- FLA gitlink at that snapshot:
  `95a895680065346884f08fb31dfb9f297fa2b8d8`
- lm-evaluation-harness gitlink:
  `1ba35e623b9bd9ca48df926f1a028043e159a6f2`
- clone: `/home/augus/src/Linearization-paper-0e3cfae`
- venv: `/home/augus/.venvs/liger-paper-20260820`
- Python 3.10.20; torch 2.5.1; Triton 3.1.0; Transformers 4.47.1
- FlashAttention is pinned to 2.7.4.post1 unless the historical source declares a
  stricter compatible release before installation. All resolved packages are
  frozen after install.
- hardware and seed: RTX 3090, BF16, `20260820`

No pretrained checkpoint or benchmark dataset is downloaded.

## Ordered gates

1. **Provenance:** all three exact SHAs reachable, detached, and clean. Missing
   `.gitmodules` is repaired only via the README-named repositories and recorded.
2. **Environment:** exact top-level requirements import from the new venv; no
   prior venv is mutated.
3. **State transfer:** reduced Llama base and Liger candidate have zero missing,
   unexpected, or shape-mismatched tensors; strict load succeeds.
4. **Construction:** deterministic `[1,8]` BF16 CUDA forward/backward yields
   logits `[1,8,256]` and finite loss/logits/gradients.
5. **Repeatability:** two fresh processes must reproduce the same gate vector;
   successful scalar results agree within `1e-3`.
6. **Recurrence:** full causal and eight cached single-token forwards have
   `max_abs <= 5e-2`, `max_rel <= 5e-2`, and cache lengths exactly `1..8`.

The decision is lexicographic and fail-closed. A construction or recurrence
failure blocks checkpoint download, quality evaluation, and training. A local
patch would require another campaign and cannot be labeled paper-snapshot
replication.

## Declared risks

Source review already exposed two falsification targets: the Liger attention
updates cache offset using `q.shape[1]`, and its local sliding-attention branch
does not visibly persist K/V in `FlaCache`. The historical FLA API may also differ
in tensor-layout defaults. These observations are recorded before execution and
will not be repaired in this campaign.

The live Qwen38 llama.cpp service stays running; the reduced model must fit in
remaining VRAM or stop with an allocation failure. Nothing is pushed remotely.
