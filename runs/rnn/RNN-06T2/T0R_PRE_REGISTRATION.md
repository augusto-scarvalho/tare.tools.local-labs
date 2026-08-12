# RNN-06T2-T0R — PRE-REGISTRATION (frozen before substantive lifecycle outcomes)

This document is committed **before** the T0R substantive run. It freezes the new fixed-batch
lifecycle contract, the property split, all numerical tolerances, the model/backend/source identity,
and the decision gate. It does **NOT** rewrite or relax historical RNN-06T T0.

## 1. Exact subject (verified LIVE before the run)

| Item | Value |
|---|---|
| model repository | `state-spaces/mamba2-1.3b` |
| immutable revision | `c5b59d00ec85d313adea86a08cad2a43c962dd3b` |
| loader | official `mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel` |
| mamba_ssm | 2.2.4 (+cu12torch2.6cxx11abiFALSE-cp312) |
| causal_conv1d | 1.5.0.post8 (+cu12torch2.6cxx11abiFALSE-cp312) |
| triton | 3.2.0 |
| torch | 2.6.0+cu124 (cxx11abi=False) |
| CUDA runtime | 12.4 |
| python | 3.12.3 |
| device / dtype | RTX 3090 (cc 8.6), bf16 |
| chunk_size | model config `chunk_size` (recorded as part of state identity) |

`ENVIRONMENT_PROVENANCE.json` records the live-resolved revision, config identity, weight
fingerprint, kernel path, and source hashes. **If the exact subject is unavailable or the official
fast path cannot fire within a bounded engineering repair:**
`OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE = NOT_RUNNABLE` — persist evidence, package, STOP. No
substitution of another checkpoint.

"Package installed" is not evidence: the runner instruments kernel entry points in
`mamba_ssm.modules.mamba2` and asserts the official prefill + recurrent-step kernels fire with exact
expected counts and that no fallback branch is reachable (Test J).

## 2. New lifecycle contract — two SEPARATE properties

RNN-06T conflated these; RNN-06T2 separates them **before outcomes**.

### PROPERTY A — `FIXED_BATCH_REQUEST_ISOLATION` (the operational contract)

At a **fixed batch shape** (`BATCH_T0R = 8`, held constant), a focal sequence's recurrent state
**and** its continuation/readout must be invariant when, for the other rows:
- neighbor **order** changes (row permutation),
- neighbor **identity/content** changes (sibling substitution),
- focal **row position** changes within the batch.

Equivalence for state: **BIT_EXACT** (SHA-256 over bf16 raw bytes of the focal row's per-layer
`(conv_state, ssm_state)`). Equivalence for continuation/readout: **argmax-identical** over the
constrained scored-value set **and** continuation state BIT_EXACT for the focal row. This is the
property the downstream single-pass recovery contract actually needs (capture, restore, readout,
selection all occur at one fixed batch shape).

### PROPERTY B — `BATCH_SHAPE_NUMERICAL_PORTABILITY` (batch1 vs batchB)

Whether the focal row's state is numerically portable across a **change of batch shape**
(batch1 vs batchB). This is a **separate** property.

**Preregistered scoping decision (before outcomes):** batch-shape numerical portability is
**OUT_OF_SCOPE_FOR_FIXED_BATCH_RECOVERY**. Rationale: the operational recovery contract holds the
batch shape fixed end-to-end (a serving deployment pins its batch/slot geometry; capture→restore→
readout→selection never crosses a batch-shape boundary). Therefore batch-shape portability is
**not required for downstream**. It will be measured as a **diagnostic** (`batch1` vs `batchB`
max-abs state diff) and reported. If the diagnostic shows any divergence beyond the historical
`TOL_BATCH = 0.03`, we mint `BATCH_SHAPE_NUMERICAL_PORTABILITY = OUT_OF_SCOPE_NOT_QUALIFIED`.

This scoping is permitted **only** because RNN-06T2 is a NEW fixed-batch operational contract. It
does **not** alter the historical RNN-06T strict verdict, and the historical `0.5` divergence is
**not** re-labeled "benign" — it remains an unexplained (out-of-scope) numerical divergence.

## 3. Fresh qualification set (disjoint; frozen)

- Lifecycle sequences master seed: **20261050** (random token-id sequences, `LSEQ=320`,
  boundaries `[80,160,240,320]`, `BATCH_T0R=8`).
- Neighbor-substitution alt seed: **20261051**.
- Single-pass historical-capture MQAR seed: **20261060** (06D v2 anti-oracle construction, `M=192`,
  `K=4` schedule `[38,76,115,153]`, held-out subset boundaries).
- Pools: 06B2 disjoint single-token pools over the GPT-NeoX tokenizer (`EleutherAI/gpt-neox-20b`),
  construction seed **20260817** (pool *vocabulary* is shared by design; example *specs* are what
  must be disjoint).

All seeds are in the `20261xxx` range, disjoint from every prior RNN-06/06A/06A2/06T seed
(`{20260811,13,14,15,16,17,18, 20260901,02,50,60,70,80,81}`). The runner emits
`lifecycleQualificationSetSha256_T0R` computed over the actually-materialized token ids + example
specs, and asserts it disjoint from RNN-06A / 06A2 / RNN-06T T0. No seed screening; frozen before
substantive outcomes.

**Self-record before the substantive run:** runner SHA-256, git blob, HEAD, dirty state, protocol
SHA-256, qualification-set SHA-256, model revision, backend source identity, kernel path.

## 4. Required T0R tests (all at FIXED batch shape unless a test is the batch-shape diagnostic)

- **A. Deterministic same-path replay.** Run the canonical path twice; compare state at all declared
  boundaries (BIT_EXACT) and final continuation + readout (argmax-identical + state BIT_EXACT).
- **B. Destroy / reload / restore / continue.** Run prefix; capture full state; serialize; destroy
  the original runtime cache; reconstruct; restore; continue. Compare against the **uninterrupted
  same-path continuation** (NOT a different full-prefill algorithm). BIT_EXACT final state.
- **C. Real branch / fork** (no tautologies). From one frozen parent snapshot create branch P and
  branch Q. For P: run P, then independently reconstruct a fresh P-reference from the same parent and
  require equivalence (BIT_EXACT state + argmax-identical readout). Same for Q. Prove: parent
  unchanged after P and after Q; P execution does not alter the later Q result/state; Q execution
  does not alter the later P result/state. Persist actual hashes/readouts.
- **D. Fixed-batch neighbor isolation.** At one frozen batch shape: neighbor order permutation,
  neighbor identity/content replacement, row permutation. For the focal row compare captured
  recurrent state (BIT_EXACT), continuation state (BIT_EXACT), readout logits/argmax
  (argmax-identical).
- **E. Reset / reuse** (the property RNN-06T did NOT test). Dirty a cache; invoke the declared reset;
  reuse that same reset cache to run a deterministic continuation; separately run the identical
  continuation with a genuinely fresh cache; compare state (BIT_EXACT) + continuation + readout.
  "Tensors became zero" alone is insufficient.
- **F. Serialization roundtrip + continuation.** Snapshot; serialize to CPU/durable; destroy the
  original; restore; **continue execution**; compare against a no-roundtrip continuation from the
  same boundary (BIT_EXACT). Immediate hash equality alone is insufficient.
- **G. Fixed-batch slice ownership.** Fixed batch shape; row permutation + sibling substitution; each
  focal row's state/continuation/readout invariant to unrelated sibling changes. Do NOT claim batch1
  equivalence from this test.
- **H. Temporal snapshot identity.** For actual in-run captures record: runId, exampleId, snapshot
  ordinal, token position, recurrence boundary, cache/inference position, conv-state hash,
  SSM-state hash, combined-state hash. Replay the same path independently and prove the state
  captured at a declared boundary is the state that boundary actually represents. Boundary mismatch
  = FAIL.
- **I. Weight immutability.** Separate `OFFICIAL_CHECKPOINT_IDENTITY` (revision) from
  `LOADED_WEIGHT_MUTATION_SENTINEL` (sum-based tensor fingerprint — a cheap **mutation sentinel**,
  explicitly NOT a cryptographic full-weight hash). No training/weight mutation.
- **J. Backend / fast-path identity.** Instrument and prove official prefill kernels fired, official
  recurrent step kernels fired, and the fallback path count. Unexpected fallback invalidates the
  corresponding fast-path claim.

### Batch-shape diagnostic (Property B; not a gate criterion for lifecycle)

Measure `batch1` vs `batchB` focal-row state max-abs diff. Report descriptively. Governs only the
independent `BATCH_SHAPE_NUMERICAL_PORTABILITY` mint (preregistered OUT_OF_SCOPE).

## 5. Single-pass historical capture T0R

A full example executes as **ONE** trajectory (prefill first slot, then autoregressive step decode
under one `InferenceParams`); historical states captured **in-run** at declared boundaries; the
**same** run continues to FINAL. Snapshots must NOT be reconstructed by independent prefix prefill.
For a deterministic held-out subset: captured state hash **==** independent same-path replay state
hash at **every** declared boundary. Persist real assertion records. `snapshotBoundaryChecks` counts
**actual** boundary comparisons performed (never a configured-but-unexecuted count).

## 6. Numerical tolerances (frozen)

| Comparison | Equivalence | Value |
|---|---|---|
| Recurrent state (fixed batch, same inputs) | SHA-256 of bf16 bytes equal | BIT_EXACT |
| Continuation state (fixed batch) | SHA-256 equal | BIT_EXACT |
| Readout (constrained scored set) | argmax identical | exact |
| Readout logits (report only) | max-abs diff alarm | `TOL_READOUT_LOGIT = 1e-2` (descriptive) |
| Batch-shape diagnostic (Property B) | max-abs state diff vs historical `TOL_BATCH` | `0.03` (out-of-scope) |

Fixed-batch, identical-input, deterministic-kernel execution is expected BIT_EXACT (historically
demonstrated for A/B/F/H at fixed batch). BIT_EXACT is therefore the frozen equivalence; a nonzero
readout-logit diff at fixed batch would itself be an alarm.

## 7. Decision gate

Mint exactly:
```
OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE = QUALIFIED | NOT_QUALIFIED | NOT_RUNNABLE
BATCH_SHAPE_NUMERICAL_PORTABILITY    = QUALIFIED | NOT_QUALIFIED | OUT_OF_SCOPE_NOT_QUALIFIED
SINGLE_PASS_HISTORICAL_CAPTURE_T0R   = QUALIFIED | NOT_QUALIFIED
```

`OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE = QUALIFIED` requires ALL of tests A–J to pass under the
frozen tolerances (Property A / fixed-batch), with J showing no reachable fallback and the official
kernels firing at expected counts. Property B does NOT gate the lifecycle mint (it is out-of-scope).

**Item 2 (RNN-06T2-T1R) executes ONLY if**
`OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE = QUALIFIED` **AND**
`SINGLE_PASS_HISTORICAL_CAPTURE_T0R = QUALIFIED`.
Otherwise `RNN-06T2-T1R = BLOCKED_BY_T0R` — persist evidence, package, STOP. No post-outcome
amendment to make the gate green.
