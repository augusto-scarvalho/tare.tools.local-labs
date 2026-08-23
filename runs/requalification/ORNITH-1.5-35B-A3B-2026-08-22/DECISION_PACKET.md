# Ornith 1.5 35B-A3B compact 3090 qualification

Status: **FROZEN BEFORE DOWNLOAD**  
Date: 2026-08-22

## Question

Can the newly published MIT-licensed Ornith 1.5 35B-A3B provide a materially different local
open-weight coding/agent option on the RTX 3090 while preserving the lab's 4 GiB free-VRAM reserve?

## Frozen artifact

- Repository: `bartowski/Ornith-1.5-35B-A3B-GGUF`
- Revision: `64b0493d34a5ca4c1b4ad67bb99b41d74b4f07d6`
- File: `Ornith-1.5-35B-A3B-IQ4_XS.gguf`
- Expected bytes: `19278554784`
- Expected SHA-256: `d6aef57fa948e9bba3ca4959b3c237ed898c605471f48c73a32cedbd24aabe70`
- Source model: `ornith-ai/Ornith-1.5-35B-A3B` (MIT)

IQ4_XS is selected instead of the official Q4_K_M because the official file is 21,713,462,848 bytes
and is unlikely to preserve the frozen reserve once the embedding service and 32k KV are resident.
This is a fit-first screen, not a claim that IQ4_XS is quality-equivalent to Q4_K_M.

## Dependency-gated sequence

1. Download to a new directory; do not replace any existing artifact. Require exact bytes and SHA.
2. Serve through the pinned llama.cpp fork at 32,768 context with the embedding endpoint left resident.
   Require at least 4,096 MiB free after health and one warm request.
3. Run the long-tool-ID eight-case agent suite. Require at least 7/8 and no blind irreversible retry.
4. If agent passes, run the four-case cache-correctness probe. Require 4/4.
5. If cache passes, run the five frozen GSM8K replay cases. Require at least 3/5.
6. If GSM passes, run `Mbpp/260` at 2,048 tokens. Require a terminating, scored answer.

Any failed gate stops later spending. A compact pass qualifies a candidate role; it does not change the
canonical service or claim parity with the publisher's reported large-scale benchmarks.

