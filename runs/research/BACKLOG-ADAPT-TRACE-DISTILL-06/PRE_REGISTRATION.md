# BACKLOG-ADAPT-TRACE-DISTILL-06 preregistration

Task: Complete the seven-seed trace confirmation with the actual 48-task QA panel  
Evidence class: `distillation`  
Executor: Codex executor  
Date: 2026-08-27

## Hypothesis

The R5 hypothesis and all math-side factors remain frozen: full teacher traces produce a positive, repeatable GSM8K gain over matched answer-only SFT across seven seeds. R6 corrects only a material implementation error in the protected panel: `f01..f48` selected 10 existing tasks rather than all 48 corpus rows.

R6 imports nine completed R5 workers and checkpoints without changing their training or 256 math outputs, evaluates the 38 QA tasks missing from each, and trains only the five absent arms. All 14 final workers must contain exactly 504 training steps, 256 math samples, and the actual 48 QA samples.

## Frozen inputs

- Admission: SHA-256 `31d8cb59f17bc01a1eafcd7550372cc19f3c7faa47ca3b3f106149d69c950fa3`.
- Continuation ledger: `CONTINUATION_SOURCES.json`, 2,952 bytes, SHA-256 `817d595739eff09e3b7d2a78f82b331f8a411d0874b54abe8d43055a2d3066fc`; it binds nine worker JSONs and their adapter configs/weights individually.
- R5 preregistration: SHA-256 `30b87154ef906703bc04f35eea67ba110b9968ba6e2c0b2bb2264526bfdd86e1`.
- R5 training manifest: SHA-256 `5c3f0d5fd80d97351839bca1e38685e5e21b3357dfa56077f44f02b857bfe4cc`.
- R5 restored-service record: SHA-256 `0831e29cf2e138eb90ed663c07ab1252e0a82ae4a024c5db6df4649f0df49825`.
- Complete QA IDs, in corpus order: `f01..f10`, `m01..m10`, `r01..r08`, `i01..i08`, `c01..c06`, `s01..s06`; canonical compact-JSON SHA-256 `5377ee57e27a3480fdad26c05cc7cc13b7e177c69abdda77795f898d43df45f3`.
- GSM8K, teacher, and QA corpus hashes remain `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`, `dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e`, and `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`.
- Base model weights/config/tokenizer remain bound by hashes `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`, `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`, and `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`.

## Command

```powershell
python tools/research/run_trace_distillation_confirmation_r6.py --outdir runs/research/BACKLOG-ADAPT-TRACE-DISTILL-06
```

## Factors

- Seeds, epoch orders, arms, optimizer, 504-step budget, LoRA configuration, math panel, decoding, bootstrap procedure, thresholds, and arm order remain identical to R5.
- Imported workers: both arms for seeds 20260830-20260833 plus answer-only seed 20260834.
- Fresh workers: full-trace seed 20260834 and both arms for seeds 20260835-20260836.
- Imported checkpoints receive only the missing 38 QA prompts. Their existing 256 math and 10 QA samples remain immutable and are hash-verified before augmentation.
- Fresh workers receive all 256 math and all 48 QA tasks.
- No partial R5 score changes any seed, threshold, treatment, or estimand.

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

- Any continuation, corpus, panel, model, or training-manifest hash differs.
- An imported worker differs from 504 training steps, 256 math samples, or exactly the existing ten `f01..f10` QA samples.
- Augmentation regenerates math or existing QA outputs instead of adding only the 38 missing IDs.
- A fresh worker produces fewer than 256 math or 48 QA samples.
- Service/embedding restoration fails or any worker exits nonzero.

## Allowed claims

- `TRACE_DISTILLATION_CONFIRMED_R6`
- `TRACE_DISTILLATION_NOT_CONFIRMED_R6`

No teacher noninferiority, production promotion, or broader generalization claim is allowed. The executor stops at `EXECUTED`; independent review remains mandatory.
