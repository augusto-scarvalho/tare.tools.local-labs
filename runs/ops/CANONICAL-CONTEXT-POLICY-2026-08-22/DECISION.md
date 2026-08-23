# Canonical Qwen3.8 context policy

Decision: **RETAIN 131,072 CANONICAL / 81,920 RESERVE PROFILE**  
Date: 2026-08-22

No service mutation was required: the deployed profile already matches the selected canonical policy.

## Evidence reconciled

- LAB-OPS-003 measured 4,151 MiB free at 81,920 tokens and 2,807 MiB at 131,072. The former is the
  largest tested ladder point that preserves the historical 4 GiB reserve.
- LAB-CTX-002 found the official-data bounded RULER panel fragile at 64k (82.82% pilot, four output
  truncations; 91.97% after selective replication) and clean at 128k (100% in the 13-task pilot and
  100% in the bounded 19-receipt panel). Length-conditioned instances prevent a causal comparison, but
  only 128k has direct adequate-quality evidence near the advertised ceiling.
- LAB-CTX-003 showed repo-completion quality degrading well below either allocation, so reducing the
  server ceiling would not repair that model-quality failure.
- LAB-OPS-001 prevents LAB launches from silently sharing the canonical SERVE GPU, and LAB-OPS-002
  showed same-GPU interference is materially costly. Avoiding co-location is therefore the primary
  protection; unused allocation reserve is not a substitute for the mode boundary.

## Policy

1. Keep `CTX_SIZE=131072` for the canonical, single-slot SERVE endpoint.
2. Keep 81,920 as the named reserve-preserving profile for temporary cases that require at least
   4,096 MiB free VRAM. It is not the default and does not inherit a broad effective-context claim.
3. The 4 GiB floor remains mandatory for new candidate admission and any explicitly co-resident arm.
   The exclusive canonical endpoint is an intentional exception backed by its measured 2,807 MiB margin.
4. Do not co-locate GPU experiments with the 131k endpoint. Use the SERVE/LAB transition and stop the
   service cleanly when a GPU experiment requires the board.
5. Reopen this decision if the model/runtime/KV type changes, free VRAM falls below 2,560 MiB on clean
   startup, or an 81,920-specific effective-context packet demonstrates equivalent task coverage.

This closes the allocation-policy backlog item while preserving the measured resource curve and the
128k effective-context evidence without claiming that nominal context equals repository-scale coding quality.
