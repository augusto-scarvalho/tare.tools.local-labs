# RNN-04 — AUDIT CORRECTIONS (supersession note, 2026-08-10)

RNN-04 is **accepted** as a useful mechanism-level **LINEAR-ATTENTION** Memory Caching reproduction, with the
independent-audit qualifications below. This note **supersedes** the corresponding wording in the raw
artifacts; it does **not** edit them. The raw evidence (`rnn04_results.json`, `rnn04_analysis.json`,
`pareto.csv`, `run.log`, `substrate_unittest.json`, `RNN-04-benchmark-selftest.json`, the handoff, and the
already-delivered ZIP) is preserved **immutable**, including its earlier interpretations. Where this note and
a raw artifact disagree, **this note governs interpretation.**

Repo `C:\projects\local-model-lifecycle`, branch `master`, **nothing pushed**. Authored while HEAD was
`40ced93`; now **committed** (RNN-05A final durability closure) so a clean clone retains this governing
RNN-04 supersession — it is referenced by `runs/rnn/RNN-05A-fixed-backbone/RNN04_CARRYFORWARD.md`. Committing
this clarification does **not** edit the immutable RNN-04 raw artifacts.

1. **Substrate is plain additive Linear Attention, not DeltaNet/Gated DeltaNet.** The executed path
   `DeltaMemory._seg_linear()` computes additive `S = Σ kᵢ vᵢᵀ` (linear attention, paper Eq. 2). The
   beta/delta-rule scan exists in code history but is **not used** by the experiment. The reproduction
   remains valid because Linear Attention is itself a Memory-Caching paper substrate.

2. **Same `[d_k,d_v]` state shape does NOT establish Gated DeltaNet semantic equivalence.** Shape match is
   necessary, not sufficient, for a GDN mapping claim.

3. **Equal-memory control confounds bytes with model size.** The larger-single-state control (d=68) changes
   not only recurrent-state bytes but also model dimensionality / trainable-parameter count. Correct
   statement: **"large single state dominates GRM under approximately equal recurrent-state bytes"** — NOT
   "bytes alone caused the gain".

4. **The N={1,2,4,8,16} curve retrains a model for each N.** Label: **TRAINED_PER_N_MEMORY_CAPACITY_CURVE**.
   It is NOT an inference-only cached-state-count sweep on fixed weights.

5. **Segmentation coverage:** the runner uses default `warm_start=False`, so
   **INDEPENDENT_COMPRESSOR = TESTED** and **CONTINUOUS_CHECKPOINT / WARM_START = NOT_TESTED**. (The spec
   describes both; only the independent-compressor variant was executed.)

6. **Post-training moving-average negative is scoped.** The 0.361 < base result applies only to the tested
   independent-compressor toy-MQAR setup. It does **NOT** falsify the paper's post-training
   length-extrapolation claim (a different setup that was not reproduced here).

7. **`aggregation_read_ms` is incremental MC overhead**, computed as (agg-mode − single-mode) wall-clock,
   NOT a directly measured read-only time. Interpret as MC incremental overhead unless/until read-only time
   is measured directly.

8. **SSC learned-vs-random is a router ablation, not a full causal control.** It swaps the selection policy
   at eval on the **learned-router-trained** model; it is not yet an independently-trained random-policy
   model. Strong evidence the router learns relevance, but not a complete causal separation.

9. **QWEN_GDN_TRANSPLANT_GATE = CONDITIONAL / DEFER** — this **supersedes** the raw analyzer/handoff value
   "PASS". The gate remains DEFERRED until BOTH (a) a fixed-backbone recurrent experiment and (b) an actual
   DeltaNet / Gated-DeltaNet semantic experiment pass. Do NOT touch Qwen GDN.

**Net:** GRM reproduces a real, replicated benefit over a matched fixed-size linear-attention state, but a
large single state dominates at ~equal recurrent-state bytes (MC_ONLY_HELPS_WITH_MORE_MEMORY), and the
transplant gate is CONDITIONAL/DEFER. Next: **RNN-05A** (await packet; do not auto-start).
