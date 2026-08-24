# 81,920-token reserve-profile runtime A/B

Frozen before execution on 2026-08-23.

- Binary: canonical candidate `b10165-71676e46c`.
- Model and shared flags: Qwen3.8-27B UD-Q4_K_XL, q4_0/q4_0 KV, one slot, flash attention,
  all GPU layers, 32 context checkpoints, batch 2,048.
- Control: MTP n3, explicit ubatch 512, context 81,920.
- Challenger: MTP n4, explicit ubatch 1,024, context 81,920.
- Three deterministic equivalence probes and three counterbalanced short/long repetitions per arm.
- Hard gates: at least 4,096 MiB free after load, byte-identical outputs, valid streaming telemetry,
  no kernel/CUDA alert, and complete candidate cleanup.
- Recommendation gate: at least 5% gain on one primary throughput axis and no more than 3%
  regression on the other.

This packet measures the named reserve profile. It does not replace the canonical 131,072-token
profile or mutate the production service automatically.
