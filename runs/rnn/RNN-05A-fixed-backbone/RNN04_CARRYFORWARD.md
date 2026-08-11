# RNN-05A — Stage 0: RNN-04 carry-forward / clarification note (2026-08-10)

RNN-05A continues the Memory Caching line on a **fixed (frozen) backbone**. Before any RNN-05A code, this
note carries forward the accepted RNN-04 audit findings so they are not silently re-inflated. It does **not**
rewrite RNN-04 raw results (immutable under `runs/rnn/RNN-04-memory-caching/`, governed by that dir's
`AUDIT_CORRECTIONS.md`). Where this note and an RNN-04 raw artifact disagree on interpretation, the audit
note + this note govern.

Starting HEAD for RNN-05A: `40ced93` (`40ced9300b0e6c8c7ce07e81093559eae4e77196`), branch `master`, nothing
pushed.

## Findings carried forward from RNN-04 (authoritative)

1. **Substrate is plain additive Linear Attention, not DeltaNet.** The executed path
   `DeltaMemory._seg_linear` computes `S = Σ kᵢ vᵢᵀ`, read `Sᵀq` (paper Eq. 2). The beta/delta-rule is
   reference-only and unused. Valid because Linear Attention is itself a Memory-Caching-paper base memory.
   RNN-05A **inherits this substrate** as its first frozen reference backbone (as the packet directs);
   actual DeltaNet / Gated-DeltaNet semantics are deferred to **RNN-05B**. → substrate.py module docstring
   corrected in this stage (comments only; executed behaviour unchanged, RNN-04 remains bit-reproducible).

2. **Same `[d_k,d_v]` state shape ≠ Gated-DeltaNet semantic equivalence.** Shape match is necessary, not
   sufficient. RNN-05A does not upgrade this.

3. **RNN-04's equal-memory control confounds bytes with model size** (bigger single state also changes
   dimensionality / param count). Carried as *"large single state dominates GRM under ~equal recurrent-state
   bytes"*, NOT "bytes alone caused the gain".

4. **RNN-04's N={1,2,4,8,16} curve retrained a model per N** (TRAINED_PER_N_MEMORY_CAPACITY_CURVE). RNN-05A
   explicitly **repairs this**: its memory-budget curve holds ONE frozen backbone (and one trained reader)
   constant and varies only the retained-state budget N.

5. **RNN-04 executed only the independent-compressor lifecycle** (`warm_start=False`);
   continuous-checkpoint / warm-start was NOT_TESTED. RNN-05A explicitly **tests both** lifecycles and proves
   checkpoint/restore correctness for each (closes the RNN-04 spec↔evidence gap).

6. **RNN-04's post-training moving-average negative (0.361 < base)** is scoped to the tested
   independent-compressor toy-MQAR setup; it does **NOT** falsify the paper's post-training
   length-extrapolation claim. RNN-05A keeps `TRAINING_FREE_MC` and `READER_TRAINED_MC` as **separate**
   questions and never merges their conclusions.

7. **`QWEN_GDN_TRANSPLANT_GATE = CONDITIONAL / DEFER`** (RNN-04 downgraded it from a raw "PASS"). RNN-05A
   does **not** advance this gate: Qwen progression still needs RNN-05A fixed-backbone signal → RNN-05B real
   Gated-DeltaNet reproduction → only then a Qwen candidate packet.

## What RNN-05A adds (scope)

Frozen-backbone transfer semantics only: can cached historical recurrent states + a **small trained
reader** (or a param-free aggregation) beat the *same frozen backbone's* ordinary single-state inference,
with **BACKBONE_WEIGHT_MUTATION = 0** across all arms. No deep (2-layer MLP) memory (that is RNN-05C). No
Qwen. No real DeltaNet yet (that is RNN-05B).
