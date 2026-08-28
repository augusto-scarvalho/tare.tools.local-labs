# BACKLOG-ADAPT-TRACE-DEPLOY-RESCORE-02 preregistration

Task: Rescore the frozen trace deployment finalist with an external fixture-validated extractor
Evidence class: `distillation`

## Hypothesis

The R1 absolute-accuracy failure is a scorer false negative. Applied once to
the same 512 frozen outputs, a generic question-aware extractor validated on
external fixtures will retain a positive paired trace gain and raise trace
accuracy to at least 40%, without seed reselection or another panel.

## Frozen inputs

- `runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/receipt.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/student_samples.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/actual_scores.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/dataset_hashes.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/checkpoint_hashes.json`
- `tools/analysis/final_numeric_answer.py`
- `tests/fixtures/final_numeric_answer_cases.json`
- `tests/test_final_numeric_answer.py`

- Admission: `6fc64b6edaa63656509d795e83d732603cd02f996b28cb1ba984d9165a87550f`.
- R1 receipt: `b4fc924a1542e4913c3c1d70fdf77f8bb9be0e2662b8757d0d06f82b60d3f521`.
- R1 student samples: `288270e4faa780bbd905b593193bf9c9edc595d84bf41cc2ef3fd72ba53663c9`.
- R1 scores: `c2ba817d9919d7c58e4d9ca33f6ce3105c9f25e3ccf105e96d94631944f3a18a`.
- R1 dataset hashes: `f3bd82ee0aef9b7eb7669d7ed6bc5549412c8a5ebdec33bb4496bb869c95661c`.
- R1 checkpoint hashes: `d13a971c7e27509142dfd0f15a61574bc5f7ebf7f32742b2df008643987014c2`.
- Scorer: `16fa779e53b6ceaa9a26507aa42c780171ba748003ffe9c8049bab310ae47905`.
- External fixtures: `414c86c25be9b49483475ea784297d9e6c693903fb0843d01fcbf40bbc595975`.
- Fixture test: `c1f59bbe0782d6395fbf9d0c282e02a6a00ab3bb4254161ddec66aea283272e4`.

## Command

```powershell
python tools/research/run_trace_deploy_rescore_r2.py --outdir runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-RESCORE-02
```

## Factors

- Exactly 512 existing outputs: answer-only and full-trace checkpoints for
  seed 20260832 on the frozen 256-task third panel.
- The scorer sees only prompt and generated text; gold is used afterward solely
  by `numeric_equal`.
- External fixtures include trailing contextual numbers, currency, time,
  distance, fractions, equations and missing answers.
- Paired bootstrap remains 20,000 task resamples at seed 2026082711.

## Acceptance gates

- `source_integrity`: `frozen_deployment_sources_verified eq True`
- `fixture_validation`: `external_fixture_pass_rate eq 1.0`
- `fixture_coverage`: `external_fixture_cases ge 18`
- `selection_reproducibility`: `selected_seed eq 20260832`
- `panel_isolation`: `third_panel_disjoint_from_training_and_prior_panels eq True`
- `evaluation_coverage`: `rescored_generations eq 512`
- `finalist_gain`: `paired_bootstrap_95ci_lower_trace_minus_answer gt 0.0`
- `finalist_absolute`: `trace_third_panel_accuracy ge 0.4`
- `protected_retention`: `imported_selected_seed_qa_regression le 0.05`
- `scorer_blinding`: `scorer_does_not_receive_gold eq True`

## Abort conditions

Abort on source hash drift, fixture failure, arm/task duplication, missing
coverage, panel/seed mismatch or scorer access to gold. This successor performs
no inference, training, service mutation, seed selection or panel selection.

## Allowed claims

- `TRACE_DISTILLATION_DEPLOYMENT_FINALIST_RESCORED_R2`
- `TRACE_DISTILLATION_DEPLOYMENT_FINALIST_NOT_CONFIRMED_R2`

Claims outside these codes are forbidden even if a metric looks favorable.
