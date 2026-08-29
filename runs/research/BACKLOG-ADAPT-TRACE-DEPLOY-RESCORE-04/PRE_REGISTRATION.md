# BACKLOG-ADAPT-TRACE-DEPLOY-RESCORE-04 preregistration

Task: Final trace finalist aggregation from promoted blind labels
Evidence class: `distillation`

## Hypothesis

With the independently promoted blind labelset, full-trace retains a strictly positive paired-bootstrap gain over answer-only and reaches at least 40% accuracy on the frozen 256-task third panel.

## Frozen inputs

- `runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-03/raw/receipt.json`
- `runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-03/raw/sealed_scored_labels.jsonl`
- `runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-03/REVIEW.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/receipt.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/actual_scores.json`

The promoted label receipt/review/scored rows and original finalist receipt/retention metrics are frozen by SHA-256 in `tools/research/run_trace_deploy_rescore_r4.py`.

## Command

```powershell
python tools/research/run_trace_deploy_rescore_r4.py
```

## Factors

Offline aggregation only: 256 paired answer-only/full-trace rows, 20,000 paired-bootstrap replicates, seed 2026082814. No label change, inference, training, seed selection, GPU or service mutation.

## Acceptance gates

- `source_integrity`: `promoted_blind_labels_verified eq True`
- `evaluation_coverage`: `trace_labeled_rows eq 512`
- `finalist_gain`: `paired_bootstrap_95ci_lower_trace_minus_answer gt 0.0`
- `finalist_absolute`: `trace_accuracy ge 0.4`
- `protected_retention`: `imported_selected_seed_qa_regression le 0.05`

## Abort conditions

Abort on source mismatch, non-promoted label receipt, missing/duplicate pair, any row outside trace source, recomputation mismatch, or incomplete provenance. A scientific gate failure is recorded and ends this rescore family.

## Allowed claims

- `TRACE_DISTILLATION_DEPLOYMENT_FINALIST_CONFIRMED_R4`
- `TRACE_DISTILLATION_DEPLOYMENT_FINALIST_NOT_CONFIRMED_R4`

Claims outside these codes are forbidden even if a metric looks favorable.
