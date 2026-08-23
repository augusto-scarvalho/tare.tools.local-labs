# LAB-OPT-001 — bounded multi-objective runtime tuning

Frozen before execution on 2026-08-22.

## Question

Can the current Qwen3.8 27B Q4_K_XL llama.cpp runtime improve prefill or decode performance on the
RTX 3090 by changing only MTP draft depth and micro-batch size, without changing deterministic
outputs, violating the 4 GiB VRAM reserve, or regressing the other throughput axis materially?

## Fixed factors

- Model: `/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf`.
- Binary: `/home/augus/src/slop.cpp/build/bin/llama-server`.
- Context: 32,768 tokens for the tuning screen; one slot; 32 context checkpoints.
- KV: q4_0/q4_0; flash attention on; all model layers on GPU; mmap default.
- Batch size: 2,048. Power limit: existing 420 W default.
- Greedy decoding: temperature 0, top-k 1, seed 0, prompt cache disabled.
- The embedding endpoint on port 8081 remains live. The canonical text service is stopped through
  `llm-inference.service`, the mode lock moves SERVE -> LAB, and every candidate uses port 8092.

These factors are not searched because KV, mmap, power and slot topology already have dedicated
evidence. Reopening them here would multiply the space and confound attribution.

## Search space and scheduler

The complete grid is six cells:

- `spec_draft_n_max ∈ {2, 3, 4}`
- `ubatch_size ∈ {1024, 2048}`

Optuna records a four-objective study. A deterministic, single-GPU successive-halving schedule is
used instead of pretending the host can run asynchronous trials:

1. Round 1 evaluates all six cells once on equivalence probes plus a short performance request.
2. Infeasible cells are removed. The non-dominated frontier is retained, then filled to at most
   three survivors by the frozen normalized rank score.
3. Round 2 measures each survivor with three counterbalanced short/long repetitions.

The incumbent (`draft_n=3`, `ubatch=2048`) is forced into Round 2 even if noisy Round 1 ranking would
otherwise remove it.

## Hard gates

A cell is infeasible if any condition holds:

- server load or health fails;
- any of three fixed greedy probe outputs differs byte-for-byte from the incumbent;
- free VRAM after load falls below 4,096 MiB;
- telemetry or request boundaries fail;
- any request returns no predicted tokens.

Hard-gate failures cannot be traded against speed.

## Objectives and promotion rule

Round 2 objectives are: minimize median TTFT, minimize median decode seconds per generated token,
minimize peak VRAM used, and minimize wall-clock seconds per accepted equivalence probe. The full
Pareto frontier is retained.

No deploy default is changed automatically. A candidate can only be recommended over the incumbent
when it is feasible, improves at least one throughput median by at least 5%, and regresses neither
throughput median by more than 3%. With only three repetitions this is a bounded screen, not a claim
of universal optimality.

## Abort and restoration

Abort the campaign on embedding health loss, less than 16 GiB available host RAM, unexpected extra
text servers, or cleanup failure. Partial receipts remain evidence. Finally free port 8092, verify
only the embedding server remains, restore SERVE mode, start `llm-inference.service`, and verify both
8080 and 8081 health. The cancelled reliability soak is not resumed.
