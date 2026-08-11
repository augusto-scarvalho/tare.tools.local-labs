# RNN-05A — Audit Reconciliation (clarification / supersession, 2026-08-10)

**RNN-05A = ACCEPTED_WITH_AUDIT_CLARIFICATIONS.** The core negative result stands: on a frozen single-state
linear-attention backbone, Memory Caching does not beat ordinary single-state inference. This note reconciles
seven interpretation/instrumentation issues raised by independent audit. It **supersedes interpretation only**
— every RNN-05A raw artifact (`rnn05a_results.json`, `backbone_identity.json`, `lifecycle_proofs.json`,
`immutability_gate.json`, `param_accounting.json`, `pareto_fixed_backbone.csv`, `rnn05a_outcomes.json`,
`run.log`) is **immutable and unchanged**. Machine-readable form: `rnn05a_audit_reconciliation.json`. Start
HEAD `25691f6`.

## 1. Lifecycle scope — matrix state, not full module
The lifecycle proofs operate on the Linear-Attention **matrix** state *S* **after** q/k/v were already
computed by causal depthwise convolutions over the **full** sequence. The conv boundary state (the k−1 window)
is neither reset nor serialized, so the proofs cover the matrix state only.
- **QUALIFIED (unchanged):** `LINEAR_MATRIX_INDEPENDENT_STATE`, `LINEAR_MATRIX_CONTINUOUS_STATE`,
  `LINEAR_MATRIX_CHECKPOINT_RESTORE`.
- **Reclassified:** `FULL_MODULE_INDEPENDENT_COMPRESSOR = NOT_QUALIFIED`,
  `FULL_MODULE_CHECKPOINT_RESTORE = NOT_QUALIFIED`.
- **Not a failure of the experiment** — the accuracy comparison remains valid for the implemented substrate;
  this narrows the *scope* of the lifecycle claim.

## 2. Full-module lifecycle requirement for RNN-05B (recorded, not implemented)
Any real DeltaNet/Gated-DeltaNet packet **must** identify, checkpoint and reset **all** sequence-owned state:
recurrent matrix state, **causal/depthwise conv state**, and any normalization/state buffers. Its test must be
`prefix → serialize COMPLETE cache → destroy runtime cache → restore → feed ONLY continuation tokens → compare
continuation`, **without** precomputing future q/k/v from the original prefix. GDN is **not** implemented now.

## 3. Cost instrumentation fix (+ assertion)
`cost_breakdown` previously read `states[:1]` and gated pools from `segs[:1]` — for N>2 that under-counts the
real GRM read/gate work (which is against **all** prior states/pools). **Fixed:** segment *i* now reads
`states[:i]` and gates `pools[:i]`, matching `segmented_forward` exactly. A count assertion (probe historical
states per segment == GRM forward's `len(cached_states)`) is added and **passes** for N∈{2,4,8,16}
(`cost_probe_selfcheck.json`, `COST_PROBE_SELFCHECK=PASS`) on a fresh untrained model (structural, no training).
- **`COMPONENT_COST_REMEASUREMENT = BLOCKED_BY_NON_DURABLE_READER_ARTIFACT`:** the exact trained reader
  (`mem.w_u` after arm D) was never durably saved; per packet §4 cost is **not** recreated from approximate
  training. The code fix is preserved for future runs.
- The timing columns in the immutable `pareto_fixed_backbone.csv` came from the old probe and **understate**
  read/gate cost for N>2 → superseded **as cost** by this note (accuracy/byte columns unaffected).
- *Note:* component timing depends only on shapes/N (einsum + softmax), not on `w_u` values — a future run that
  durably saves the reader (or just the backbone) can regenerate the corrected table.

## 4. Cache-count curve interpretation
The N∈{1,2,4,8,16} sweep changes cached-state count **and** segment size **and** checkpoint spacing together,
so it is not an isolated cached-state-count Pareto. Renamed **`FIXED_BACKBONE_SEGMENTATION_MEMORY_CURVE`**.
`PURE_CACHE_COUNT_CURVE = NOT_TESTED`. The sweep also used the **independent**-lifecycle reader, so
`CONTINUOUS_READER_MEMORY_CURVE = NOT_TESTED` (the continuous reader was the best frozen-MC arm; not swept, and
not run here as it would require the absent exact reader / retraining).

## 5. Co-adaptation wording (hedged)
Replace *"the RNN-04 benefit was co-adaptation"* with: **"The RNN-04 positive result plus the RNN-05A
frozen-transfer negative strongly SUPPORT the backbone–memory co-adaptation hypothesis, but do not yet causally
isolate it."** Dev/holdout direction agreement is a generalization signal, not an independent training
replication: `EVAL_GENERALIZATION = DIRECTION_CONSISTENT_DEV_HOLDOUT`, `TRAINING_REPLICATION_COUNT = 1`.

## 6. Backbone parameter accounting
Distinguish **optimizer_parameter_population** from **effective_gradient_receiving_parameters**. Base training:
optimizer population **66 333**, effective **63 261** (`mem.w_u`'s 3 072 params are unused by single-state
forward → no gradient). Arm-D reader training: both **3 072**. Historical raw `trained_params=66333` (the
optimizer population) is unchanged.

## 7. Exact checkpoint disposition
`rnn05a_backbone.pt` (SHA-256 `8b5977439f4e…762e`, 270 849 B) copied out of the ephemeral session scratchpad to
a durable non-Git lab location and bundled in the refreshed audit ZIP under `external_artifacts/`; **not**
committed to Git. No exact reader checkpoint exists.
