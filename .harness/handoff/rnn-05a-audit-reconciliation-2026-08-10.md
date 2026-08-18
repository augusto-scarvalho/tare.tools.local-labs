# HANDOFF — RNN-05A Audit Reconciliation (lifecycle scope + cost instrumentation) 2026-08-10

Small evidence-repair packet on the accepted RNN-05A negative result. **No training, no tuning, no Qwen /
llama.cpp / serving / deploy, not pushed.** RNN-05A raw evidence immutable; all fixes are supersession /
clarification artifacts + one instrumentation code fix. **RNN-05A = ACCEPTED_WITH_AUDIT_CLARIFICATIONS.**

## Git
- Branch **master**. **Start HEAD `25691f6`** → reconciliation `ea38a45` → **FINAL HEAD `40c5e7c`**
  (`40c5e7c813749ba10d8c8b543c0ee40e9195faf2`, durability closure). Reconciliation commit `ea38a45`
  (4 files: `ops/rnn_mc_05a.py` +86, `AUDIT_RECONCILIATION.md`, `rnn05a_audit_reconciliation.json`,
  `cost_probe_selfcheck.json`); durability commit `40c5e7c` tracks `runs/rnn/RNN-04-memory-caching/
  AUDIT_CORRECTIONS.md` (referenced by RNN04_CARRYFORWARD.md) so a clean clone keeps the governing RNN-04
  supersession. No raw JSON/CSV/log altered; tracked & staged trees CLEAN. **Not pushed.**
- Tests: `COST_PROBE_SELFCHECK=PASS` (N∈{2,4,8,16}); `py_compile` OK. No training executed.
- Untracked (out of Git): `.harness/`, `git_evidence*.txt`, RNN-08/08b adapter dirs. Full state +
  `diff --stat` in `runs/rnn/RNN-05A-fixed-backbone/git_evidence_reconciliation.txt`.

## The seven reconciliations (detail: `runs/rnn/RNN-05A-fixed-backbone/AUDIT_RECONCILIATION.md` + `…_audit_reconciliation.json`)
1. **Lifecycle scope.** Proofs cover the Linear-Attention **matrix** state only (q/k/v precomputed by causal
   convs over the full sequence; conv boundary state not reset/serialized). Preserve
   `LINEAR_MATRIX_INDEPENDENT_STATE / _CONTINUOUS_STATE / _CHECKPOINT_RESTORE = QUALIFIED`; reclassify
   **`FULL_MODULE_INDEPENDENT_COMPRESSOR = NOT_QUALIFIED`**, **`FULL_MODULE_CHECKPOINT_RESTORE = NOT_QUALIFIED`**.
   Not a failure — accuracy comparison stays valid for the implemented substrate.
2. **RNN-05B requirement (recorded, not built).** Real DeltaNet/GDN must checkpoint/reset **all** sequence-owned
   state (recurrent matrix + **causal/depthwise conv** + norm buffers) and pass
   prefix→serialize→destroy→restore→**continuation-only** without precomputing future q/k/v from the prefix.
3. **Cost probe fix.** `do_read` used `states[:1]`, `do_gate` used `segs[:1]` → under-counted read/gate for N>2.
   Fixed to `states[:i]` / `pools[:i]` (matches `segmented_forward`). Added assertion: probe historical count
   per segment == GRM forward `len(cached_states)` → **PASS**. **`COMPONENT_COST_REMEASUREMENT =
   BLOCKED_BY_NON_DURABLE_READER_ARTIFACT`** (exact trained reader never saved; no retrain). Old CSV timing
   understates read/gate for N>2 → superseded **as cost** (accuracy/byte columns unaffected). Timing is
   `w_u`-value-independent, so a future run can regenerate it.
4. **Curve interpretation.** N sweep changes cached-count **and** segment size **and** checkpoint spacing →
   renamed **`FIXED_BACKBONE_SEGMENTATION_MEMORY_CURVE`**; **`PURE_CACHE_COUNT_CURVE = NOT_TESTED`**;
   **`CONTINUOUS_READER_MEMORY_CURVE = NOT_TESTED`** (sweep used the independent reader; continuous reader not
   swept, not run here).
5. **Co-adaptation wording.** "RNN-04 positive + RNN-05A frozen-transfer negative **strongly SUPPORT** the
   backbone–memory co-adaptation hypothesis, but do not yet causally isolate it."
   `EVAL_GENERALIZATION = DIRECTION_CONSISTENT_DEV_HOLDOUT`, `TRAINING_REPLICATION_COUNT = 1`.
6. **Param accounting.** `optimizer_parameter_population = 66 333` vs `effective_gradient_receiving_parameters
   = 63 261` (base; `w_u` unused by single-state forward). Reader arm: both `3 072`. Raw `trained_params=66333`
   unchanged.
7. **Checkpoint disposition.** `rnn05a_backbone.pt` (SHA-256 `8b5977439f4e…762e`, 270 849 B) copied from the
   ephemeral scratchpad to durable **`.harness/artifacts/`** (non-Git) and bundled under `external_artifacts/`.
   No exact reader checkpoint exists.

## Reproduce the instrumentation check
```
V=/home/augus/tptt-venv/bin/python
PYTHONPATH=/mnt/c/projects/local-model-lifecycle/ops $V \
  /mnt/c/projects/local-model-lifecycle/ops/rnn_mc_05a.py \
  --cost-selfcheck /mnt/c/projects/local-model-lifecycle/runs/rnn/RNN-05A-fixed-backbone/cost_probe_selfcheck.json
# -> COST_PROBE_SELFCHECK: PASS  (structural, untrained, no GPU training)
```

## Bundle contents (refreshed ZIP)
Original RNN-05A evidence · this reconciliation (MD + JSON) · corrected `ops/rnn_mc_05a.py`
(+ substrate + bench) · `cost_probe_selfcheck.json` · both git-evidence files ·
`external_artifacts/rnn05a_backbone.pt` (exact frozen backbone) · RNN-04 `AUDIT_CORRECTIONS.md` (context).

## Guardrails (unchanged)
Serving CLOSED · TPTT PARKED · no real DeltaNet yet (RNN-05B) · no deep memory (RNN-05C) · no Qwen · no
push · Qwen gate still **CONDITIONAL / DEFER**. Do not auto-start RNN-05B.
