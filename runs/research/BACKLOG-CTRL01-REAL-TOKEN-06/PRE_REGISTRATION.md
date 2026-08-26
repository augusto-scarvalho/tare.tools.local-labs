# BACKLOG-CTRL01-REAL-TOKEN-06 preregistration

Task: Recover canonical CTRL-01 receipt with production-only binding search
Evidence class: `mechanism_research`

## Hypothesis

Independent rescoring of the immutable 36-row physical sample ledger will reproduce the source metrics exactly when runtime binding is searched only in production trees. CTRL-01 qualifies only if every substantive gate passes; otherwise the historical promotion is a confirmed false positive on this frozen JSON panel.

## Frozen inputs

- `runs/research/BACKLOG-CTRL01-REAL-TOKEN-05/ABORTED.md`
- `runs/research/BACKLOG-CTRL01-REAL-TOKEN-04/PRE_REGISTRATION.md`
- `runs/research/BACKLOG-CTRL01-REAL-TOKEN-04/raw/samples.jsonl`
- `runs/research/BACKLOG-CTRL01-REAL-TOKEN-04/raw/receipt.json`
- `runs/research/CTRL-01-AST-SIDECAR-2026-08-25/raw/receipt.json`
- `tools/analysis/ast_grammar_sidecar.py`

- Recovery-search abort SHA-256: `6ec8da528b6891684468b212d07cbf6fe577f88f50fbc500ef666d9b2e0d6165`.
- Source preregistration SHA-256: `d34b12631dbde35bec0ebb62eadb05e912924d6b17c26e609cdb9ace12d25f87`.
- Source samples SHA-256: `94ef7a81b5b7bd83d8c300bbc68f15852b622ce2180d37d1fbdfe8a802248ec5`.
- Source noncanonical receipt SHA-256: `3a097f0d20e4f8c7c2217899fe35bc72406d44b51143d8b04d163ee964a8d7fc`.
- Historical receipt SHA-256: `0f37ae1d3ff33286a193353731f864d699ce738734fd8cc5b5a55384c2cf2c7c`.
- Historical sidecar SHA-256: `3cb90b1b5aa5aacdff93b7a8b0cdc38e689099e0d1365989f00b7b34acbb1463`.

## Command

```powershell
python tools/research/run_ctrl01_receipt_recovery_r2.py --outdir runs/research/BACKLOG-CTRL01-REAL-TOKEN-06
```

## Factors

- No new inference and no mutation of source evidence.
- Require exactly 24 `real_model` plus 12 `valid_control` rows and all frozen source hashes.
- Independently parse raw and filtered text; reconstruct both from exact token pieces/decisions; recompute token acceptance and latency percentiles.
- Search `ASTGrammarSidecar` runtime integration only under `src/` and `ops/`; research, tests and probes are excluded by construction.
- Exact agreement with every source summary metric is mandatory but cannot rescue a failed substantive gate.

## Acceptance gates

- `real_coverage`: `real_model_outputs ge 24`
- `real_validity`: `sanitized_complete_valid_rate eq 1.0`
- `valid_control_recall`: `valid_token_acceptance_rate eq 1.0`
- `valid_control_semantics`: `valid_control_exact_preservation_rate eq 1.0`
- `overhead`: `p95_overhead_us_per_token le 500.0`
- `runtime_binding`: `logit_mask_runtime_integrated eq True`

## Abort conditions

- Any source hash, row count, reconstruction or source-summary metric disagrees.
- Any expected evidence file cannot be generated or provenance is incomplete.
- Thresholds or denominators change after rescoring.

## Allowed claims

- `CTRL01_RUNTIME_QUALIFIED_R6`
- `CTRL01_FALSE_POSITIVE_CONFIRMED_R6`

Claims outside these codes are forbidden even if a metric looks favorable.

This successor recovers provenance and scoring only; it makes no new-inference claim.
