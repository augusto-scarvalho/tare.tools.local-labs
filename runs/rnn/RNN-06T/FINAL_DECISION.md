# RNN-06T — FINAL DECISION (all states reported separately)

Official Mamba transportability + single-pass historical recovery train. Frozen subject
`state-spaces/mamba2-1.3b` @ `c5b59d00ec85d313adea86a08cad2a43c962dd3b`, official `mamba_ssm` 2.2.4 +
`causal_conv1d` 1.5.0.post8 Triton/CUDA fast path, torch 2.6.0+cu124, RTX 3090. Nothing pushed.

## Decision states (do NOT compress to "Mamba PASS")

| state | verdict |
|---|---|
| `OFFICIAL_MAMBA_FASTPATH` | **RUNNABLE** (kernel firing proven; 0 fallback) |
| `OFFICIAL_MAMBA_LIFECYCLE` | **QUALIFIED** (lifecycle A–J; neighbor isolation bit-exact) |
| `SINGLE_PASS_HISTORICAL_CAPTURE` | **QUALIFIED** (real in-run states, one runId, hash==replay) |
| `HISTORICAL_RECOVERY_TRANSPORT` (3A) | **QUALIFIED** (FIXED_76 & MAX_CONF ≫ FINAL) |
| `ADAPTIVE_SELECTOR_ADVANTAGE` (3A, narrow band) | **DIRECTIONAL** (MAX_CONF−FIXED_76 +0.052, CI incl. 0) |
| `OFFICIAL_MAMBA_SYNTHETIC_TRANSPORT` (3A) | **PARTIAL** (recovery QUALIFIED, adaptive DIRECTIONAL) |
| `WIDE_TARGET_RECOVERY` (3B) | **QUALIFIED** (MAX_CONF−FINAL +0.490, robust 4/4) |
| `ADAPTIVE_SELECTION` (3B, wide band) | **QUALIFIED** (MAX_CONF−BEST_FIXED +0.339 [0.271,0.401], robust 3/4) |
| `END_TO_END_RECOVERY_UTILITY` | **QUALIFIED** (marginal: +991.5 ≤ 1000 ms/query envelope) |
| `NON_SYNTHETIC_RECOVERY_SCOUT` | **NO_SIGNAL** (FINAL already 1.0 on NL needle; nothing to recover) |

Appended RNN-06D audit reconciliation verdicts (see `AUDIT_RECONCILIATION.md`) are preserved.

## What transported, what didn't, and where it matters

1. **The fast path is real and lifecycle-sound.** The official kernel fires (proven by counters), and
   the recurrent state supports deterministic replay, save/reload continuation, branch/fork,
   **bit-exact request isolation** (neighbor identity/content), reset, serialization, and temporal
   identity. A single batch-SIZE numerical sensitivity (0.5) is a benign Triton-tiling artifact,
   neutralized by using a fixed batch size throughout.
2. **Genuine single-pass capture works** — historical snapshots are ACTUAL in-run states from one
   trajectory (runId identical, hashes match uninterrupted replay), not the RNN-06D re-prefill
   approximation. This closes the RNN-06D `SINGLE_PASS_HISTORICAL_CAPTURE_PARITY = NOT_TESTED` gap.
3. **Historical recovery transports exactly.** On the official substrate, pool-per-slot accuracies
   {38:0.50, 76:0.77, 115:0.47, 153:0.26} match RNN-06D (transformers) almost identically, and
   MAX_CONFIDENCE 0.82 ≈ 06D's 0.83. Even a fixed early snapshot beats FINAL by +0.55.
4. **The adaptive selector's value is regime-dependent — the key scientific result.** In the narrow
   band [8,64] where one fixed slot (76) is near-optimal, adaptive selection is only DIRECTIONAL
   (+0.052, CI includes 0) — prospectively **confirming the RNN-06D audit**
   (`ADAPTIVE_SELECTOR_INCREMENTAL_ADVANTAGE = NOT_QUALIFIED`). In the wide band [8,144], where no
   fixed snapshot can see every target, adaptive confidence selection is **decisively QUALIFIED**
   (+0.339 over the best fixed slot). Adaptive selection earns its keep only when the target location
   is genuinely unknown/variable.
5. **End-to-end cost is dominated by the capture path, not the selector.** The recovery mechanism
   itself is cheap (capture+restore+readout+selection ≈ 64 ms/query; intrinsic 13 ms). The premium
   (+992 ms/query) is almost entirely the **step-path penalty**: single-pass capture forces the
   recurrent step decode (965 ms/query) instead of the fused prefill (37 ms/query), because the fused
   kernel cannot expose intermediate states. QUALIFIED but marginal; a custom kernel exposing
   chunk-boundary states would collapse this cost.
6. **Practical scope is bounded (honest negative).** The forgetting that recovery exploits is specific
   to the dense unique-load MQAR construction. On a natural-language needle-in-haystack, Mamba-2 does
   NOT forget (FINAL = 1.000) — so there is no recovery signal, and the mechanism does no harm. The
   phenomenon is real and transportable, but its utility depends on the workload actually inducing
   recurrent-state saturation.

## Exactly one next recommendation (NOT executed)

The synthetic transport is PARTIAL and the non-synthetic scout showed NO_SIGNAL because ordinary NL
context does not saturate the state. **OPEN, in a NEW session, a bounded investigation of whether any
realistic long-context workload induces the recurrent-state forgetting regime** (e.g. many-key
retrieval / high-interference RULER variants at longer context), under independent audit and fresh
pre-registration — i.e. establish whether the qualified recovery mechanism has a real operating point
before investing in a deployable selector or a capture-exposing kernel. Do NOT run Qwen, a trained
reader, DART, StateX, SDM, GDN-2, INT8 archive, or ReplaySSM. Nothing pushed.
