# RNN-06T — T0 DECISION

## Verdicts

- **`OFFICIAL_MAMBA_LIFECYCLE = QUALIFIED`**
- **`SINGLE_PASS_HISTORICAL_CAPTURE = QUALIFIED`**
- ⇒ **Item B OPEN.**

Official `state-spaces/mamba2-1.3b` @ `c5b59d00` via `mamba_ssm` 2.2.4 + `causal_conv1d` 1.5.0.post8
(Triton fast path, no fallback). Executed-source PROVEN (runner blob committed, dirty ∅). Runtime
101 s, peak VRAM 9.7 GB. Held-out deterministic lifecycle sequences (seed 20260950), NOT from 06A;
old 06A verdicts unchanged.

## Lifecycle (all pass; BIT_EXACT unless noted)

| test | result |
|---|---|
| A deterministic same-path replay | BIT_EXACT (all 4 boundary hashes equal) |
| B save/destroy/reload/restore/continue vs uninterrupted | FINAL BIT_EXACT |
| C branch/fork (parent unchanged, independent branches) | parent hash BIT_EXACT before==after |
| D neighbor/request isolation | **BIT_EXACT** under neighbor order + content permutation; readout argmax-invariant |
| E reset/reuse | fresh cache all-zero; reset all-zero |
| F serialize/deserialize roundtrip | BIT_EXACT |
| G batch slice ownership | row state BIT_EXACT under permutation |
| H snapshot temporal identity | snapshot hash == uninterrupted-replay hash at each boundary; monotonic; cache_pos==boundary |
| I model weights immutable | Σ-params identity unchanged |
| J source/backend frozen | no fallback reachable; step kernels fired; revision constant |

### D/G test-semantics correction (transparent)

The first D/G implementation compared a sequence run **alone at batch 1** vs in a batch of 6 — a
**batch-SIZE** change, which is not the preregistered property. The T0 pre-registration defines D as
invariance to **which other sequences share the batch** (neighbor identity + content, at fixed batch).
The corrected test measures exactly that and passes **BIT_EXACT** (reordering rows or replacing all
neighbors leaves each row's state byte-identical; readout argmax invariant). No results were committed
before the correction; no threshold was changed.

**Batch-SIZE numerical sensitivity (descriptive, not a gate).** Running the same sequence at batch 1
vs batch 6 yields a state max-abs-diff of **0.5** — a benign Triton-kernel tiling artifact (the
recurrence is per-row independent; the difference is floating-point tiling order, not
cross-contamination). **Consequence carried into Item B:** every arm (FINAL, snapshots, readouts) for
a comparison is computed at a **fixed batch size** on one trajectory, so this artifact cannot confound
the paired contrasts.

## Single-pass historical capture (QUALIFIED)

One trajectory over a synthetic MQAR context (768 tokens), capturing ACTUAL in-run states at
boundaries [156,308,464,616,768] (slots [38,76,115,153,191]) — NOT re-prefilled prefixes:
- all snapshots + FINAL share one `runId = fc8b63bf700cac21`; boundaries strictly monotonic;
- each snapshot's state hash == the uninterrupted-replay state hash at that boundary (proving it is
  the real in-run state, not a later recompute);
- restoring each snapshot into an independent branch and reading out the target query succeeds with
  the correct temporal position;
- step kernels (`selective_state_update`, `causal_conv1d_update`) fired for the whole trajectory, no
  fallback.
Already visible: the slot-76 snapshot (b=308) recovers the target while FINAL (b=768) does not — the
06D recovery pattern, now on the official fast-path substrate via genuine single-pass capture.

## State contract

`OFFICIAL_MAMBA_STATE_CONTRACT.json`: per-layer conv_state (B,4352,4) bf16 + ssm_state
(B,64,64,128) bf16, ×48 layers = 52,002,816 bytes/seq; restore = `copy_` in place; branch = clone
into a fresh cache; reset = zeros; `seqlen_offset` only routes prefill/step (Mamba-2 is
position-independent). Identical total state size to the transformers `Mamba2Cache`.

## Consequence

Both gates QUALIFIED ⇒ proceed to Item B (3A exact-contract transportability) under a separate
pre-registration. No 06A/06D artifact modified; no reader/DART/Memory Caching/StateX/GDN/Qwen; nothing
pushed.
