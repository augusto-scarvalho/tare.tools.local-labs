# Gemma 4 26B A4B Heretic compact role qualification

Status: **FROZEN BEFORE MODEL EXECUTION**  
Date: 2026-08-22

- Resident community GGUF derived from Gemma 4 26B A4B IT, previously used only as a writing judge.
- Repository `mradermacher/Gemma-4-26B-A4B-it-heretic-antislop-i1-GGUF`, revision
  `84775f5b3e286fe1b95251cd6ee79a08a69e1254`.
- Artifact `Gemma-4-26B-A4B-it-heretic-antislop.i1-Q4_K_M.gguf`, expected 16,796,015,904 bytes,
  SHA-256 `13cfcadee358e54c3246ecf9b8a528633d1d4444e17177cdaadeec54955eb5ae`.

The existing `gemma-judge` LAB profile is unchanged: full GPU offload, FlashAttention, 16,384 context,
batch/ubatch 2,048 on port 8091. Embedding remains active on 8081. Gates:

1. Exact local hash and at least 4,096 MiB free VRAM after load.
2. Long-ID-compatible agent suite at least 7/8, zero blind irreversible retries.
3. No-spec cache suite 4/4; if Gemma tokenization exceeds 16,384, the smallest compatible context may
   be tested only if it still preserves the 4,096 MiB reserve.
4. Frozen five-case GSM8K replay at least 3/5 strict within 2,048 tokens.
5. `Mbpp/260` scorable and Base-pass within 2,048 tokens.

Any failed gate yields `HOLD_ROLE` and stops later gates. Passing all yields `QUALIFIED_COMPACT`, not
incumbent promotion. Restore the canonical Qwen endpoint after the campaign.
