# GPT-OSS 20B Q4 compact role qualification

Status: **FROZEN BEFORE MODEL EXECUTION**  
Date: 2026-08-22

- Resident open-weight MoE candidate: `unsloth/gpt-oss-20b-GGUF` revision
  `ce6ba6163271f5d73dbe2a20b85e66d79126e942`.
- Artifact `gpt-oss-20b-Q4_K_M.gguf`, expected 11,624,759,488 bytes and SHA-256
  `c27536640e410032865dc68781d80a08b98f8db5e93575919af8ccc0568aeb4f`.
- Purpose: determine whether the already-resident 20B model provides a compact coding/agent role on one
  RTX 3090; prior kernel experiments do not constitute role qualification.

Runtime: pinned local llama.cpp, LAB port 8092, full GPU offload, FlashAttention, Jinja, 32,768 context,
no speculative drafter, embedding port 8081 unchanged. Gates execute in order:

1. Exact local hash and at least 4,096 MiB free VRAM.
2. Long-ID-compatible agent suite at least 7/8, no blind irreversible retry.
3. Cache correctness 4/4.
4. Frozen GSM8K replay at least 3/5 strict within 2,048 tokens.
5. `Mbpp/260` produces a scorable Base-passing answer within 2,048 tokens.

Failure stops later gates and yields `HOLD_ROLE`; all gates passing yields `QUALIFIED_COMPACT`, not
incumbent supersession. Restore canonical Qwen after the resident-candidate wave.
