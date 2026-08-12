# RNN-06T — T0 PRE-REGISTRATION (lifecycle + single-pass capture)

Frozen BEFORE any lifecycle/capture outcome. Held-out deterministic lifecycle sequences (own seed,
not copied from RNN-06A). Old RNN-06A verdicts are NOT changed.

## Canonical path (as in TRAIN_PROTOCOL)

Prefill first slot (`mamba_chunk_scan_combined`), then autoregressive step decode
(`selective_state_update` + `causal_conv1d_update`) under one `InferenceParams`; capture in-run at
token boundaries {156,308,464,616} and FINAL 768. State per seq = 52,002,816 bytes (conv (4352,4) +
ssm (64,64,128) bf16 × 48 layers). State hash = SHA-256 over the bf16 bytes of the concatenated
per-layer (conv,ssm) tensors.

## Lifecycle tests + preregistered tolerances

Combined-state **BIT_EXACT** = identical SHA-256 of the full 52,002,816-byte state.

- **A deterministic same-path replay** — two runs of the same sequence ⇒ every boundary state
  BIT_EXACT. (PASS iff all boundary hashes equal.)
- **B uninterrupted vs save/destroy/reload/restore/continue** — save state at slot 76, destroy the
  cache object, reload from serialized bytes, continue to FINAL ⇒ FINAL state BIT_EXACT vs the
  uninterrupted run.
- **C branch/fork** — from a parent snapshot, branch P (query readout A) and branch Q (query readout
  B); the parent snapshot hash is BIT_EXACT unchanged after both branches; P and Q are independent.
- **D neighbor/request isolation** — a sequence's captured boundary states and its readout are
  invariant to which other sequences share its batch. Primary: readout argmax-invariant AND
  state max-abs-diff ≤ `TOL_BATCH = 3e-2` (bf16 scale) vs run-alone. (Recorded separately whether it
  is additionally BIT_EXACT; batch-reduction numerics in the prefill may perturb below TOL_BATCH.)
- **E reset/reuse** — after `reset`, states equal a freshly allocated cache (all-zero); a reused-cache
  run is BIT_EXACT vs a fresh-cache run.
- **F serialize/deserialize roundtrip** — state → CPU bytes → state, then continue ⇒ BIT_EXACT vs
  no-roundtrip continuation.
- **G batch slice ownership** — row i's captured state BIT_EXACT vs example i run alone at batch 1
  (with the same TOL_BATCH fallback recorded as in D).
- **H snapshot temporal identity** — each captured snapshot's hash == the uninterrupted-replay state
  hash at that exact boundary; `cache/seq position == boundary token`; boundaries monotonic. BIT_EXACT.
- **I model weights immutable** — `model_weights_identity` (Σ params) identical before/after all tests.
- **J source/backend frozen** — repo/revision/mamba_ssm/causal_conv1d/triton/torch/kernel identity
  constant; `FAST_PATH_ACTIVE` remains true (kernel counters fire, no fallback).

## T0 gate

`OFFICIAL_MAMBA_LIFECYCLE = QUALIFIED` iff A,B,C,E,F,G,H are BIT_EXACT (D within TOL_BATCH and
argmax-invariant), and I,J hold. Any hard BIT_EXACT failure on A/B/C/E/F/H, or D exceeding TOL_BATCH
with argmax change, ⇒ `NOT_QUALIFIED`. If the model/kernels cannot run ⇒ `NOT_RUNNABLE`.

`SINGLE_PASS_HISTORICAL_CAPTURE = QUALIFIED` iff, on the synthetic single-pass run:
(a) all snapshots + FINAL share one `runId`; (b) boundaries strictly monotonic;
(c) each snapshot hash == the uninterrupted-replay state hash at that boundary (proving it is the
ACTUAL in-run state, not a later recompute); (d) restoring each snapshot into an independent branch
and reading out succeeds with correct temporal position; (e) fast-path counters confirm the step
kernels fired for the whole trajectory with no fallback. Else `NOT_QUALIFIED`.

**Both QUALIFIED are required to run Item B.** No threshold changed after outcomes.
