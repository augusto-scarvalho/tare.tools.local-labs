# BACKLOG-ADAPT-TRACE-DEPLOY-RESCORE-03 preregistration

Task: Conservative rescore of all frozen trace finalist outputs after independent scorer audit
Evidence class: `distillation`

## Hypothesis

After removing same-unit incidental-number selection, the frozen full-trace arm retains a strictly positive paired accuracy gain over answer-only and at least 40% absolute accuracy.

## Frozen inputs

- `runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/receipt.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/student_samples.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/actual_scores.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-RESCORE-02/REVIEW.json`
- `tools/analysis/final_numeric_answer_v2.py`
- `tests/fixtures/final_numeric_answer_v2_cases.json`
- `tests/test_final_numeric_answer_v2.py`

Every listed input is frozen by SHA-256 in `tools/research/run_trace_deploy_rescore_r3.py`; any mismatch aborts before writing evidence.

## Command

```powershell
python tools/research/run_trace_deploy_rescore_r3.py
```

## Factors

Offline, gold-blind rescore only: 256 paired tasks per arm (512 retained outputs), no generation, training, panel selection, or GPU use. Paired bootstrap: 20,000 replicates, seed 2026082811.

## Acceptance gates

- `source_integrity`: `frozen_deployment_sources_verified eq True`
- `fixture_validation`: `external_fixture_pass_rate eq 1.0`
- `fixture_coverage`: `external_fixture_cases ge 15`
- `retained_regressions`: `retained_regression_pass_rate eq 1.0`
- `evaluation_coverage`: `rescored_generations eq 512`
- `finalist_gain`: `paired_bootstrap_95ci_lower_trace_minus_answer gt 0.0`
- `finalist_absolute`: `trace_third_panel_accuracy ge 0.4`
- `protected_retention`: `imported_selected_seed_qa_regression le 0.05`
- `scorer_blinding`: `scorer_does_not_receive_gold eq True`

## Abort conditions

Abort on any input hash mismatch, non-empty raw directory, duplicate/missing paired task, fixture/regression failure, non-gold-blind scorer signature, or incomplete provenance. A failed scientific gate is recorded, not converted into an execution error.

## Allowed claims

- `TRACE_DISTILLATION_DEPLOYMENT_FINALIST_RESCORED_R3`
- `TRACE_DISTILLATION_DEPLOYMENT_FINALIST_NOT_CONFIRMED_R3`

Claims outside these codes are forbidden even if a metric looks favorable.
