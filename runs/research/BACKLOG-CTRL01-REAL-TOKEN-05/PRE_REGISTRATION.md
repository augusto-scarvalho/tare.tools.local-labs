# BACKLOG-CTRL01-REAL-TOKEN-05 preregistration

Task: Recover canonical CTRL-01 receipt from immutable real-token samples
Evidence class: `mechanism_research`

## Hypothesis

Independent rescoring of the immutable 36-row physical sample ledger will reproduce the source metrics exactly. CTRL-01 qualifies only if all original substantive gates pass; otherwise the historical promotion is a confirmed false positive on this frozen JSON panel.

## Frozen inputs

- `runs/research/BACKLOG-CTRL01-REAL-TOKEN-04/PRE_REGISTRATION.md`
- `runs/research/BACKLOG-CTRL01-REAL-TOKEN-04/raw/samples.jsonl`
- `runs/research/BACKLOG-CTRL01-REAL-TOKEN-04/raw/receipt.json`
- `runs/research/CTRL-01-AST-SIDECAR-2026-08-25/raw/receipt.json`
- `tools/analysis/ast_grammar_sidecar.py`

- Source preregistration SHA-256: `d34b12631dbde35bec0ebb62eadb05e912924d6b17c26e609cdb9ace12d25f87`.
- Source samples SHA-256: `94ef7a81b5b7bd83d8c300bbc68f15852b622ce2180d37d1fbdfe8a802248ec5`.
- Source noncanonical receipt SHA-256: `3a097f0d20e4f8c7c2217899fe35bc72406d44b51143d8b04d163ee964a8d7fc`.
- Historical receipt SHA-256: `0f37ae1d3ff33286a193353731f864d699ce738734fd8cc5b5a55384c2cf2c7c`.
- Historical sidecar SHA-256: `3cb90b1b5aa5aacdff93b7a8b0cdc38e689099e0d1365989f00b7b34acbb1463`.

## Command

```powershell
python tools/research/run_ctrl01_receipt_recovery.py --outdir runs/research/BACKLOG-CTRL01-REAL-TOKEN-05
```

## Factors

- No new inference and no mutation of source evidence.
- Require exactly 24 `real_model` plus 12 `valid_control` rows and the frozen source hashes above.
- Independently parse `raw_content` and `filtered` with the standard JSON parser; do not trust stored validity/preservation booleans.
- Independently recompute accepted-token counts from per-piece decisions, latency percentiles and every acceptance gate.
- Verify every filtered string equals the concatenation of accepted pieces and every token piece concatenation equals the raw model/control string.
- Runtime-binding value is independently recomputed by repository search, not copied from the source receipt.
- Exact agreement with all source summary metrics is a mandatory invariant control, but cannot rescue a failed substantive gate.

## Acceptance gates

- `real_coverage`: `real_model_outputs ge 24`
- `real_validity`: `sanitized_complete_valid_rate eq 1.0`
- `valid_control_recall`: `valid_token_acceptance_rate eq 1.0`
- `valid_control_semantics`: `valid_control_exact_preservation_rate eq 1.0`
- `overhead`: `p95_overhead_us_per_token le 500.0`
- `runtime_binding`: `logit_mask_runtime_integrated eq True`

## Abort conditions

- Any source hash, row count, row kind, per-piece reconstruction or source-summary metric disagrees.
- Any expected evidence file cannot be generated or provenance is incomplete.
- Thresholds or denominators change after rescoring.

## Allowed claims

- `CTRL01_RUNTIME_QUALIFIED_R5`
- `CTRL01_FALSE_POSITIVE_CONFIRMED_R5`

Claims outside these codes are forbidden even if a metric looks favorable.

This successor recovers provenance and scoring only; it makes no new-inference claim.
