# HANDOFF — RNN-05B Audit Reconciliation (semantics / provenance / terminology) 2026-08-11

Small evidence/semantic reconciliation on the accepted RNN-05B results. **No training, no benchmark rerun, no
Qwen, no llama.cpp/serving/deploy, no FLA install, not pushed.** Raw RNN-05B evidence immutable; all fixes are
supersession/clarification artifacts + trivial code self-checks. **RNN-05B = ACCEPTED_WITH_AUDIT_CLARIFICATIONS.**

## Git
- Branch **master**. **Start HEAD `25c28f0`** → reconciliation **`aad66fc`** (`aad66fc66edb5549dd4eec7dc70a3afd05e8c9c2`).
  One commit: `ops/rnn_delta_substrate.py` (+parity scope, +single_blob_fork), `ops/rnn_mc_05b.py`
  (+calibration single-source-of-truth constants, +self-check), and 4 evidence files
  (`AUDIT_RECONCILIATION.md`, `rnn05b_audit_reconciliation.json`, `single_blob_fork.json`,
  `calibration_rule_selfcheck.json`). No raw JSON/CSV/log altered; tracked & staged trees CLEAN. **Not pushed.**
- Untracked-by-policy: `.harness/`, `git_evidence*.txt`, RNN-08 adapter dirs. Checkpoints stay non-Git.
- Self-checks executed (CPU, no training): `CALIBRATION_RULE_IDENTITY=PASS`, `SINGLE_BLOB_FORK_ALL=PASS`;
  `py_compile` OK.

## The nine reconciliations (detail: `runs/rnn/RNN-05B-delta-gdn/AUDIT_RECONCILIATION.md` + `…_audit_reconciliation.json`)
1. **Memory-axis defect.** Recorded rule `(0.30,0.90)` vs executed `(0.30,0.96)`; D=36 (GDN 0.9407) fails 0.90.
   `MEMORY_AXIS_ORIGINAL=QUALIFIED_BY_EXECUTED_0.96_RULE`, `MEMORY_AXIS_INTENDED=NOT_QUALIFIED`,
   `MEMORY_AXIS_FINAL_INTERPRETATION=CEILING_LIMITED / NOT_QUALIFIED_FOR_POSITIVE_GAIN_DETECTION`. Code now
   derives the recorded string from the executed constants; `calibration_rule_selfcheck()` = **PASS**.
2. **Co-adaptation wording.** `TRAIN_INFERENCE_MC_INTERACTION=STRONG`, `MC_COADAPTATION_DEPENDENCE=SUPPORTED`,
   `NET_MC_BENEFIT_AFTER_COADAPTATION=NOT_DETECTED`. Regime dependence, **not** a net MC quality benefit.
   GDN 3-seed interaction preserved (0.3025/0.3432/0.2902).
3. **Parity class.** `LOCAL_DUAL_IMPLEMENTATION_PARITY=PASS`, `UPSTREAM_EXECUTABLE_PARITY=NOT_QUALIFIED`,
   `REFERENCE_PARITY_SCOPE=LOCAL_PORTS_ONLY` (FLA not invoked).
4. **Qwen mapping.** `STRUCTURALLY_ANALOGOUS_TO_QWEN_TWO_PART_CACHE`; `QWEN_CACHE_ROLE_MAPPING=SUPPORTED`,
   `QWEN_CACHE_REPRESENTATION_PARITY=NOT_PROVEN`.
5. **Snapshot terminology.** `FULL_RESTORABLE_SEQUENCE_CHECKPOINT` (matrix+conv) vs
   `HISTORICAL_RECURRENT_STATE_SNAPSHOT` (matrix-only MC cache entry). Matrix-only ≠ full checkpoint.
6. **Isolation scope.** `STATELESS_REQUEST_EXECUTION=PASS`, `CHECKPOINT_CONTINUATION=PASS`, and now
   `SINGLE_BLOB_FORK_BRANCHING=PASS` — tiny CPU test restores TWO suffixes from ONE `{S,conv}` blob; each
   matches its full run (err ≤ ~1e-6), branches diverge (~3.09).
7. **Pure-cache.** `PURE_CACHE_COUNT_CURVE=QUALIFIED`; LA degrades with K, DN small monotonic ↑, GDN ↑ K1→K4
   then slight ↓ K8 → `DN_GDN_HISTORICAL_COMPLEMENTARITY=WEAK_DIRECTIONAL_SIGNAL` (H3 **not** positive).
8. **Provenance.** `RAW_RUN_CLASSIFICATION=HISTORICAL_FIRST_PASS` (run.log: NEGATIVE/DEFER) vs
   `FINAL_DERIVED_CLASSIFICATION=SUPERSEDING_POST_ANALYSIS` (NO_EFFECT_NAIVE_MC_NEGATIVE). Metrics unchanged;
   **decision gate = DEFER**.
9. **GDN collapsibility.** Drop "final≈recent" (config-specific: d_k24 last≈0, d_k64 last=0.561); keep
   `GDN_ADDITIVE_COLLAPSE=NO` + measured distances.

## Reproduce the self-checks (CPU; no training/GPU)
```
V=/home/augus/tptt-venv/bin/python ; PP=/mnt/c/projects/local-model-lifecycle/ops
PYTHONPATH=$PP CUDA_VISIBLE_DEVICES= $V $PP/rnn_delta_substrate.py --forktest .../single_blob_fork.json --dk 64
PYTHONPATH=$PP CUDA_VISIBLE_DEVICES= $V $PP/rnn_mc_05b.py --calib-selfcheck .../calibration_rule_selfcheck.json
```

## Bundle (refreshed ZIP)
Original RNN-05B evidence · this reconciliation (MD + JSON) · `single_blob_fork.json` ·
`calibration_rule_selfcheck.json` · corrected `ops/rnn_delta_substrate.py` + `ops/rnn_mc_05b.py` ·
git evidence · `external_artifacts/rnn05b_{la,dn,gdn}_frozen_reader.pt` (exact, hashes preserved).

## Guardrails (unchanged)
Serving CLOSED · TPTT PARKED · no RNN-05C · no Qwen · no llama.cpp/deploy · no push. `QWEN_GDN_TRANSPLANT_GATE
= DEFER`. Do not auto-start RNN-05B-EXT.
