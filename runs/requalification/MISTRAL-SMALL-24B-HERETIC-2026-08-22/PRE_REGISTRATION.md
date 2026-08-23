# Mistral Small 3.2 24B Heretic compact role qualification

Status: **FROZEN BEFORE MODEL EXECUTION**  
Date: 2026-08-22

## Candidate and purpose

- Resident community GGUF derived from Mistral Small 3.2 24B Instruct 2506.
- Repository `mradermacher/Mistral-Small-3.2-24B-Instruct-2506-Heretic-v1.2-2-i1-GGUF`, revision
  `87199b98e64c6cd63c0814600ad348f495f5e9f4`.
- Artifact `Mistral-Small-3.2-24B-Instruct-2506-Heretic-v1.2-2.i1-Q4_K_M.gguf`, expected
  14,333,923,776 bytes and SHA-256 `5079999cca0823bbbddadbf905564311930901b155919386933e5143623da7cf`.
- It already served as a writing judge; this packet asks a different question: whether it offers a
  bounded agent/coding role as an additional open-weight option on the RTX 3090.

## Frozen compact gates

The existing `mistral-judge` profile is used unchanged on LAB port 8090: full GPU offload,
FlashAttention, 8,192 context, batch/ubatch 2,048. Embedding port 8081 remains active.

1. Full local hash matches the upstream receipt; load leaves at least 4,096 MiB free VRAM.
2. Agent suite passes at least 7/8 with no blind irreversible retry.
3. No-spec cache correctness passes 4/4.
4. Frozen GSM8K failures `153,241,584,1019,1312` score at least 3/5 strict within 2,048 tokens.
5. `Mbpp/260` submits a scorable answer within 2,048 tokens and passes Base tests.

Failure of gates 1–5 yields `HOLD_ROLE` and stops expansion. Passing all yields
`QUALIFIED_COMPACT`; it does not supersede the Qwen3.8 incumbent. The canonical service must be
restored exactly after the run.
