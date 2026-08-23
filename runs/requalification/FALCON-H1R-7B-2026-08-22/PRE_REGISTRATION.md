# Falcon-H1R-7B official Q8 — compact qualification pre-registration

Frozen before model execution: 2026-08-22.

## Candidate

- Official repository: `tiiuae/Falcon-H1R-7B-GGUF`
- Revision: `2dc053e015a9e3c5b954aa81e00aaed24bef830f`
- Artifact: `Falcon-H1R-7B-Q8_0.gguf`
- Expected bytes: `8,069,003,296`
- Expected SHA-256: `4c96b2ff690766257852dfd51044996521ba0bacaf9939953e5aaabdfd50898a`
- Architecture: hybrid Transformer + Mamba2; official llama.cpp GGUF
- License: Falcon-LLM License, to be reviewed separately from technical qualification

## Runtime

Single RTX 3090, llama.cpp endpoint on LAB port 8092, 32,768-token context, full GPU offload,
FlashAttention and Jinja. Embedding port 8081 remains active. Recommended sampling is temperature
0.6/top-p 0.95 with fixed seed 42; deterministic harnesses retain their own frozen sampling.

## Compact gates

1. Artifact hash/bytes match and the server loads with at least 8 GiB free VRAM.
2. Four bounded smoke prompts are non-empty and at least 3/4 terminate naturally within 2,048 tokens.
3. Existing agent/tool suite scores at least 6/8 without a blind irreversible retry.
4. Five historical GSM8K failures score at least 3/5 strict at a 2,048-token budget.
5. `Mbpp/260` produces a scorable final answer within 2,048 tokens and passes the base tests.
6. If gates 1–5 pass, the local 8k/16k retrieval/multikey/multihop/aggregation slice scores at least
   7/8 with no empty or truncated output.

Failure of gates 2–5 stops expansion and yields `HOLD_ROLE`; a runtime/load failure yields
`HOLD_RUNTIME`. Passing all gates yields `QUALIFIED_COMPACT`, not a replacement for the 27B incumbent.

