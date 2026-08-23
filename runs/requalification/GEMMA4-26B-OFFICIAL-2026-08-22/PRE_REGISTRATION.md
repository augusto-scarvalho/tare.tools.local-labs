# Gemma 4 26B A4B official QAT compact role qualification

Status: **FROZEN BEFORE MODEL EXECUTION**  
Date: 2026-08-22

- Official Google GGUF repository `google/gemma-4-26B-A4B-it-qat-q4_0-gguf`, revision
  `d1c082be9cf3c8a514acf63b8761f4b41935842e`.
- Artifact `gemma-4-26B_q4_0-it.gguf`, expected 14,439,363,584 bytes and SHA-256
  `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d`.
- Independent question from the community Heretic arm: can the official QAT instruction model supply a
  reliable compact worker role on the RTX 3090?

Runtime: pinned llama.cpp, LAB port 8092, full GPU offload, FlashAttention, Jinja, 32,768 context,
no speculative drafter; embedding port 8081 unchanged. Gates in order:

1. Exact hash and at least 4,096 MiB free VRAM.
2. Long-ID agent suite at least 7/8 with zero blind irreversible retries.
3. Cache correctness 4/4.
4. Frozen GSM8K replay at least 3/5 strict within 2,048 tokens.
5. `Mbpp/260` scorable and Base-pass within 2,048 tokens.

Failure yields `HOLD_ROLE` and stops expansion; all pass yields `QUALIFIED_COMPACT`, not incumbent
supersession. Restore canonical Qwen at campaign end.
