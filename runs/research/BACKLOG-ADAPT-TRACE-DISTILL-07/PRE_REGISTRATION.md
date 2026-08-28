# BACKLOG-ADAPT-TRACE-DISTILL-07 preregistration

Task: Complete the corrected seven-seed trace confirmation after bounded shutdown timeout  
Evidence class: `distillation`  
Executor: Codex executor  
Date: 2026-08-27

## Hypothesis

The R6 scientific design is adopted without change: nine hash-bound R5 workers receive only their 38 missing QA tasks, five absent arms are trained fresh, and all seven paired seeds are scored on 256 teacher-disjoint GSM8K plus the actual 48 QA tasks. R7 changes only the service-stop timeout from the inherited 60 seconds to a bounded 180 seconds.

R6 produced no worker, receipt, result, or observed model score. It aborted while `systemctl stop` waited for Qwen3.8 to exit. Therefore no scientific factor or threshold is changed in response to evidence.

## Frozen inputs

- Admission: 2,898 bytes, SHA-256 `c4412e7ccea811ace51e91e92c3e285bc451b1a2f4b4474419a045bcde789f45`.
- R6 preregistration: 4,419 bytes, SHA-256 `bbf8cf1c5f7a952460da014c942cada1e74a7646d4101fcd30324e56b8e51a78`.
- Continuation ledger: 2,952 bytes, SHA-256 `817d595739eff09e3b7d2a78f82b331f8a411d0874b54abe8d43055a2d3066fc`.
- R6 timeout log: 1,729 bytes, SHA-256 `3eb3f1c1ce4d2ca2809df24133097bb9301cfa952bb9e8eb53247a7be5307836`.
- R6 watcher final: 925 bytes, SHA-256 `6459899935b76ec1a17d6a3f0948f901a340a53412375f132a7aa9f577b4400b`.
- All corpus, model, worker, checkpoint, training-manifest, seven-seed, 504-step, math-panel, actual-QA-panel, bootstrap, and threshold identities are exactly those frozen in R6 and its continuation ledger.

## Command

```powershell
python tools/research/run_trace_distillation_confirmation_r7.py --outdir runs/research/BACKLOG-ADAPT-TRACE-DISTILL-07
```

## Factors

- Scientific implementation: `run_trace_distillation_confirmation_r6.py`, unchanged and hash-bound in the R7 implementation digest.
- Operational wrapper: replaces only `systemctl stop/start` invocation with direct WSL argv and a 180-second timeout; all other R6 functions and constants remain unchanged.
- Imported workers: nine. Fresh workers: five. Missing QA augmentation: 38 per imported checkpoint.
- Actual QA order remains `f01..f10`, `m01..m10`, `r01..r08`, `i01..i08`, `c01..c06`, `s01..s06`.

## Acceptance gates

- `continuation_integrity`: `hash_verified_imported_workers eq 9`
- `treatment_materiality`: `matched_distinct_training_targets_verified eq True`
- `training_budget`: `training_steps_per_arm_per_seed eq 504`
- `seed_coverage`: `completed_paired_seeds eq 7`
- `math_coverage`: `heldout_math_samples_per_arm_per_seed eq 256`
- `qa_coverage`: `protected_qa_samples_per_arm_per_seed eq 48`
- `confirmed_gain`: `hierarchical_bootstrap_95ci_lower_trace_math_gain gt 0.0`
- `directional_repeatability`: `seeds_with_positive_trace_math_gain ge 5`
- `protected_retention`: `mean_protected_qa_regression_vs_answer_only le 0.05`
- `service_recovery`: `service_and_embedding_restored eq True`

## Abort conditions

- Any R6 scientific input or continuation hash differs.
- Service stop exceeds 180 seconds, embedding becomes unhealthy, or baseline cannot be restored.
- Any imported/fresh worker violates the R6 dimensions or a worker exits nonzero.

## Allowed claims

- `TRACE_DISTILLATION_CONFIRMED_R7`
- `TRACE_DISTILLATION_NOT_CONFIRMED_R7`

No broader claim is permitted. The executor stops at `EXECUTED`; independent review remains mandatory.
