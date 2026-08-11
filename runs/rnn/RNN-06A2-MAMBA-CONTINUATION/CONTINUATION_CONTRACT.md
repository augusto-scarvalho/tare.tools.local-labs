# RNN-06A2 — Operational Continuation / Checkpoint Contract (derivation)

**Written BEFORE 06A2 outcomes.** This document derives the continuation/checkpoint contract
from the *actual* requirements of downstream RNN-06B (in this train) and the anticipated
RNN-06C (NOT in this train), then names the equivalence property 06A2 must qualify. It does
NOT reclassify or weaken historical RNN-06A (permanently `NOT_QUALIFIED`); it asks a
different question.

## 1. Why 06A2 is a new experiment, not a 06A patch

Historical RNN-06A failed its strict gate for two reasons (see 06A `AUDIT_RECONCILIATION.md`):
- **Claim B** compared *two different numerical algorithms* — a single full-sequence prefill
  vs. a prefill+segmented-decode of the same tokens — and the bf16 values differed
  (max_abs 0.625 > 0.5) even though argmax agreed 100%. That is a cross-algorithm numerical
  gap, NOT a checkpoint defect.
- **Claim E**'s preregistered primary test (P-alone vs P-in-batch) was likewise a
  batch-shape GEMM difference (BOUNDED_DIFFERENCE), demoted from a leakage test.

06A2 does **not** retro-fit a looser criterion onto 06A's frozen preregistration. It defines
the contract that downstream work truly needs and qualifies *that*, on new held-out data.

## 2. What the downstream experiments actually require

### RNN-06B (this train) — confirmatory BASE retrieval-loss region

06B scores the frozen model's **last-token logits** on many MQAR prompts, **batched**, and
compares constrained-argmax accuracy across a memory-pressure ladder. From the substrate it
requires exactly:

- **R1 Determinism** — the same prompt yields identical last-token logits on repeat, so a
  measured accuracy is a property of (model, prompt), not of run-to-run noise.
- **R2 Request isolation under batching** — a prompt's last-token logits are identical
  whether it is scored **alone** or **inside a batch with different neighbors**. 06B batches
  examples for throughput; if batch composition changed a row's logits, batched accuracy
  would be contaminated. This is the load-bearing property for 06B.
- **R3 Frozen-weight invariance** — weights (and therefore the function) do not change across
  conditions.

06B does **not** require checkpoint/restore or multi-token generation. It requires R1–R3.

### RNN-06C (future, NOT in this train) — historical-state information

06C will checkpoint the recurrent state mid-sequence, later restore it, and continue
generation to test whether retained state carries recoverable information. From the substrate
it requires, additionally:

- **R4 Recurrent-cache checkpoint/restore fidelity** — serialize {conv_states, ssm_states}
  (+ metadata), destroy the runtime, restore, and continue with the **same** stepping
  algorithm, obtaining an identical continuation to the uninterrupted run.
- **R5 Generation-frontier sufficiency** — the information needed to resume autoregressive
  generation at the checkpoint boundary is explicitly enumerated and shown sufficient (06A's
  gap: C/D seeded the frontier logit from *outside* the serialized snapshot →
  `COMPLETE_AUTOREGRESSIVE_GENERATION_CHECKPOINT = NOT_PROVEN_BY_06A`).
- **R6 Branch / isolation integrity** — one snapshot can seed multiple independent
  continuations without cross-contamination; a neighbor's request cannot alter mine.

## 3. The three contract objects

- **Recurrent Cache State** `= conv_states ⊕ ssm_states ⊕ required cache metadata`. For this
  backend the metadata that matters is the **caller-managed `cache_position`** (Mamba2Cache
  holds no `seqlen_offset`; `cache_position[0]==0 ⇒ prefill/chunked-ssd`, `>0 ⇒ single-token
  decode`). Measured size: **52,002,816 bytes/seq** (bf16), carried from 06A.
- **Generation Frontier** `=` the minimum needed to resume autoregressive generation at the
  boundary. For the **declared deterministic/greedy** downstream contract this is:
  `{last-step logits (or equivalently the already-selected next token), next cache_position}`.
  Sampler config and RNG state are **NOT** part of this contract because downstream scoring
  and continuation are deterministic greedy — see §5. We implement only these fields (do not
  invent unused ones).
- **Execution Identity** `=` `{model, revision, backend, dtype, config (incl. chunk_size),
  implementation source hashes, runtime versions, runner blob/HEAD/dirty}`. Recorded into the
  results BEFORE outcomes (repeating the 06A executed-source improvement).

## 4. Primary semantic property — CONTINUATION EQUIVALENCE

The contract's core property is **continuation equivalence**, not retrospective full-prefill
tensor value-identity:

```
run an uninterrupted reference to boundary t, then continue N tokens   (REFERENCE)
                          ── vs ──
prefill to boundary t → snapshot {cache state + generation frontier}
   → destroy execution context → restore → continue N tokens           (RESTORED)
```

**Both sides use the SAME continuation algorithm after the boundary** (identical
token-by-token single-step decode with identical `cache_position` sequence). We do NOT
compare two knowingly-different numerical algorithms (that was 06A Claim B's error) and call
the difference cache corruption. Because the restored path resumes from a *bitwise-identical*
recurrent state and frontier and runs an *identical* algorithm, the correct bar is
**BIT_EXACT**, and any deviation is a genuine checkpoint/restore defect.

## 5. Deterministic/greedy scope (why no stochastic test)

Downstream 06B uses deterministic constrained **argmax** over the last-token logits; 06C's
continuation for information-recovery is likewise run as deterministic greedy decode. The
declared contract is therefore **deterministic greedy**. Per the train's §9.J, the stochastic
frontier test (RNG-state serialization) is run **only if** stochastic sampling is part of the
declared contract — it is not — so test J is recorded `NOT_APPLICABLE_BY_CONTRACT` and scope
is not broadened to sampling.

## 6. Isolation criterion (per train §6)

Because P-alone-vs-P-in-batch changes the GEMM shape (a numerical-path artifact, not
leakage), the **primary** isolation test is **neighbor invariance**: for a fixed P row and
fixed batch shape, `[P, Q1]` vs `[P, Q2]` must leave P's last-token logits, P's per-row
recurrent state, AND P's restored continuation **BIT_EXACT**. This directly tests whether
changing *another* request changes mine — the property 06B's batched scoring depends on
(R2). P-alone-vs-P-in-batch is retained as a **diagnostic** numerical-path channel and is
**not** allowed to fail the gate.

## 7. Contract summary (qualified by 06A2 tests A–J → gate §PRE_REGISTRATION)

| Requirement | Contract object / property | 06A2 test |
|---|---|---|
| R1 determinism | reproducible logits+state | A |
| R4 cache checkpoint/restore | continuation equivalence | B, C, D |
| R5 frontier sufficiency | frontier ∈ snapshot; resume from snapshot alone | B, C (frontier serialized INSIDE snapshot) |
| R6 branch integrity | branch replay, parent immutability | E, F |
| R2 request isolation | neighbor invariance | G (primary) + diagnostic |
| — reset semantics | reset==fresh | H |
| — serialization | round-trip exact | I |
| deterministic scope | greedy; no RNG | J = NOT_APPLICABLE_BY_CONTRACT |
| R3 frozen weights | weight fingerprint invariant | weight-immutability check |
