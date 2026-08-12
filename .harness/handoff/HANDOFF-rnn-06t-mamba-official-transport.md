# HANDOFF — RNN-06T Official Mamba Transportability + Single-Pass Historical Recovery

**Author:** Claude (Opus 4.8) via Claude Code. **Date:** 2026-08-12. **Branch:** `master` (no
upstream). **Pushed:** NO. Two backlog items (T0 lifecycle+capture; B transport+selector) + economics
+ scout, all gated.

## HEAD boundary

- **START HEAD:** `8f4b18a2b2add8a2d17235d43ed9b4882f3bb7f8` (end of RNN-06D train).
- **FINAL HEAD:** `dc4e471` (see git_evidence.txt for exact rev at packaging).
- Tree clean; nothing pushed; no amend/rebase of outcome history.

## Official subject + fast path (proven)

`state-spaces/mamba2-1.3b` @ **`c5b59d00ec85d313adea86a08cad2a43c962dd3b`**, loaded via official
`mamba_ssm.MambaLMHeadModel` (48 layers, d_model 2048, bf16). Fast-path stack: **mamba_ssm 2.2.4**,
**causal_conv1d 1.5.0.post8** (cu12torch2.6cxx11abiFALSE-cp312), triton 3.2.0, torch 2.6.0+cu124,
CUDA 12.4, RTX 3090, driver 591.86. `OFFICIAL_MAMBA_FASTPATH = RUNNABLE`: kernel firing PROVEN by
instrumented counters (prefill `mamba_chunk_scan_combined`×48 + `causal_conv1d_fn`×48; step
`selective_state_update`+`causal_conv1d_update` = n_layer×n_step; **0 fallback**). ABI note: the
1.6.x/2.2.5+ "abiFALSE" wheels are mislabeled (new-ABI) against this old-ABI torch; 1.5.x/2.2.4
abiFALSE match. State per seq = 52,002,816 bytes (conv (4352,4) + ssm (64,64,128) bf16 ×48).

## Decision states (report ALL separately — never "Mamba PASS")

```
OFFICIAL_MAMBA_FASTPATH            = RUNNABLE
OFFICIAL_MAMBA_LIFECYCLE           = QUALIFIED
SINGLE_PASS_HISTORICAL_CAPTURE     = QUALIFIED
HISTORICAL_RECOVERY_TRANSPORT (3A) = QUALIFIED
ADAPTIVE_SELECTOR_ADVANTAGE  (3A)  = DIRECTIONAL      (narrow band [8,64])
OFFICIAL_MAMBA_SYNTHETIC_TRANSPORT = PARTIAL
WIDE_TARGET_RECOVERY         (3B)  = QUALIFIED
ADAPTIVE_SELECTION           (3B)  = QUALIFIED        (wide band [8,144])
END_TO_END_RECOVERY_UTILITY        = QUALIFIED        (marginal: +991.5 <= 1000 ms/query)
NON_SYNTHETIC_RECOVERY_SCOUT       = NO_SIGNAL        (FINAL already 1.0 on NL needle)
```

## Item A — T0 (both QUALIFIED ⇒ Item B ran)

Canonical single-pass path: prefill first slot (chunk_scan) then step decode (selective_state_update),
capture in-run states at token boundaries [156,308,464,616,768]. Lifecycle A–J all pass BIT_EXACT
(same-path replay, save/reload-continue, branch/fork parent-unchanged, **neighbor isolation bit-exact**
under order+content permutation, reset, serialize roundtrip, batch-slice ownership, temporal identity,
weights immutable, backend frozen). A **D/G test-semantics correction** was applied (first impl
compared batch-1 vs batch-6 — a batch-SIZE change; the preregistered property is neighbor identity at
fixed batch — bit-exact) before any result was committed; the batch-SIZE numerical sensitivity (0.5,
benign Triton tiling) is reported descriptively and neutralized by fixing batch size in Item B.
`SINGLE_PASS_HISTORICAL_CAPTURE`: one runId, monotonic, every snapshot hash == uninterrupted-replay
hash — real in-run states, not re-prefills (closes 06D's NOT_TESTED parity gap).

## Item B — 3A exact-contract transport (fresh disjoint set 5e47408e, MAX_CONFIDENCE frozen)

pool-per-slot {38:0.50, 76:0.77, 115:0.47, 153:0.26} ≈ RNN-06D; MAX_CONF 0.823 ≈ 06D 0.833.
**CLAIM 1** FIXED_76−FINAL +0.552 [0.479,0.620], MAX_CONF−FINAL +0.604 [0.531,0.677] ⇒ recovery
QUALIFIED. **CLAIM 2** MAX_CONF−FIXED_76 +0.052 CI **[−0.016, 0.115]** (includes 0) ⇒ adaptive
DIRECTIONAL — **prospectively confirms the 06D audit**. ⇒ TRANSPORT PARTIAL.

## Item B — 3B wide-target (band [8,144], calib-frozen BEST_FIXED=slot115, fresh disjoint d8012e61)

Per-region best snapshot marches with target (R0→38:0.96, R1→76:0.85, R2→115:0.79, R3→153:0.88;
pre-target slots 0.0); no fixed slot works across regions (best 0.484). MAX_CONF−FINAL +0.490
[0.417,0.557] robust 4/4 ⇒ WIDE_TARGET_RECOVERY QUALIFIED. **MAX_CONF−BEST_FIXED +0.339 [0.271,0.401]
robust 3/4 ⇒ ADAPTIVE_SELECTION QUALIFIED** (only region 2, where slot-115 is optimal, is negative).
Adaptive selection earns its keep exactly when the target location is unknown/variable.

## Section 4 — end-to-end economics (capture INCLUDED)

FINAL fused 37.5 ms/q; FINAL step 964.7 ms/q; recovery 1029 ms/q. Added vs fused **991.5 ms/q**
(≤ 1000 envelope ⇒ QUALIFIED, marginal); added vs step only 64.2 ms/q; intrinsic restore+readout 13
ms/q. Snapshot bytes 3.33 GB/batch (K×52MB×16). The premium is the **step-path penalty** (capture
forbids the fused prefill), not the selector — a capture-exposing kernel would collapse it.

## Section 5 — non-synthetic scout (exploratory)

NL needle-in-haystack (real-English filler, single-token needle, variable depth, no download): FINAL
= **1.000** — Mamba-2 does not forget a single salient needle in ordinary context, so `MAX_CONF−FINAL
= 0.0` ⇒ NO_SIGNAL. Not a mechanism failure (MAX_CONF 1.0, no harm) but a **boundary condition**: the
forgetting recovery exploits is specific to dense unique-load interference, not generic long context.

## Confirmations

`OFFICIAL_FAST_PATH_PROVEN=TRUE (0 fallback)` · `SINGLE_PASS_REAL_IN_RUN_STATES=TRUE (hash==replay)` ·
`NEIGHBOR_ISOLATION_BIT_EXACT=TRUE` · `MAX_CONFIDENCE_FROZEN_BEFORE_QUAL=TRUE` · `ALL_QUAL_SETS_DISJOINT=TRUE` ·
`THRESHOLDS_NOT_TUNED_AFTER_OUTCOMES=TRUE` · `NO_SEED_SCREENING=TRUE` · `WEIGHTS_IMMUTABLE=TRUE` ·
`06A_06D_ARTIFACTS_UNMODIFIED=TRUE (06D reconciliation append-only)` · `NO_READER=TRUE` · `NO_DART=TRUE` ·
`NO_MEMORY_CACHING=TRUE` · `NO_STATEX_SDM_GDN2_INT8_REPLAYSSM=TRUE` · `NO_QWEN=TRUE` · `NO_SERVING_CHANGE=TRUE` ·
`NO_HOST_POLICY_CHANGE=TRUE` · `NOTHING_PUSHED=TRUE`.

## Exactly one next recommendation (NOT executed)

**OPEN, in a NEW session, a bounded investigation of whether any realistic long-context workload
induces the recurrent-state forgetting regime** (high-interference / many-key retrieval at longer
context) that the qualified recovery mechanism needs — i.e. find a real operating point before
building a deployable selector or a capture-exposing kernel. Under independent audit and fresh
pre-registration. Do NOT run Qwen / trained reader / DART / StateX / SDM / GDN-2 / INT8 / ReplaySSM.

**STOP. Do not start the next train here.**
