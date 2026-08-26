# BACKLOG-ADAPT-TRACE-DISTILL-03 preregistration

Task: Complete the frozen trace-distillation false-negative test after host log-decoder abort  
Evidence class: `distillation`  
Executor: Codex executor  
Date: 2026-08-25

## Hypothesis

The hypothesis, estimand, seeds, treatments, training recipe and gates are unchanged from the preregistered `BACKLOG-ADAPT-TRACE-DISTILL-02` design: full correct teacher traces must produce strictly positive mean held-out math gain over answer-only SFT, nonnegative direction in at least two of three seeds, and no more than five percentage points of mean protected-QA regression.

This successor exists only because the Windows host decoder aborted after four workers had completed. The failure was outside GPU training and evaluation. No seed, threshold, target, example order or hyperparameter is changed after seeing the partial results.

## Frozen inputs

- Admission specification: `config/research_backlog_admissions/BACKLOG-ADAPT-TRACE-DISTILL-03.json`, 3,132 bytes, SHA-256 `17915c826fe75ba2ede1a1be151824f9ae8c276f6e6047a0d9f3713cdf94dbed`.
- Aborted-at-host record: `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-02/ABORTED.md`, SHA-256 `648970adb100d68f8a63523b5ef5f7a0eb9bf30228e3161f7bb0b7e4aeb770a9`.
- Exact training manifest: `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-02/raw/training_pairs.json`, SHA-256 `e6dd3bb9d86b0c8d34f89f68d07768e6d0f53451295e70a02fafc9d6a0748966`.
- Restored-service evidence from the abort: `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-02/raw/service_maintenance.json`, SHA-256 `029892b166778978911c22f6847ade1f155019f632b11b3d5b0ae9a3608bbbbc`.

Frozen completed workers:

| Worker | Worker JSON SHA-256 | Adapter weight SHA-256 |
|---|---|---|
| seed 20260824 answer-only | `be30915b4b5c8b402a98953ecd0aba829afbe9fa58be5116616546614ccde79c` | `ef5bec8822e856883eaec930d2b851892bb6b681bde1fda5f76005667adbf1a2` |
| seed 20260824 full-trace | `c389a8290effa80b45685516375499a0f0c41a79348b3490e489608d27eaa7db` | `174832aa1bd25cbc5ed7f0ff717ad253ec94e2c23edc82e6f828ceadeed566b7` |
| seed 20260825 answer-only | `a8fcd0a8bb782176840882513e60d9670550e13c1493f994b09f409c55a2ef36` | `56ff9be8c5ac0876389cf12fe23a2ac301eac7c99cef977fa455b76f5817a2e6` |
| seed 20260825 full-trace | `0220a38e5ae1c74695ac91b25807c85a907fa8d0c949306c59e6be6e35f872f0` | `dc696b7553cf8e4d920f8554ec4e3dee484a04da374ef0d54bcb48160044050a` |

The corresponding adapter configs are also frozen: seed-20260824 answer `5acceba987552a5aa7f128d3840b9c465225345f60eb585ab0e9d7b7742e5e14`, seed-20260824 trace `4872214e344ce266b0b10a1f70db33775224441ff1d78dbe0c7fb4dd70aeac84`, and both seed-20260825 configs `091ddc225ff85380e6815963fad17391f4d8e89fb38523ddc2b798086c82ecb1`.

Base model and original corpus identities remain those frozen in `BACKLOG-ADAPT-TRACE-DISTILL-02/PRE_REGISTRATION.md`. The successor must recompute them before launching seed 20260826.

## Command

```powershell
python tools/research/run_trace_distillation_training_r3.py --outdir runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03
```

## Factors

- Import and verify exactly four frozen completed workers; do not regenerate or edit them.
- Run only seed 20260826 in its already frozen order: answer-only then full-trace.
- Both new workers use the exact 128 rows and order already stored under manifest seed 20260826.
- Training remains LoRA MLP `r=8`, alpha 16, dropout 0; AdamW `1e-4`, weight decay 0.01; 128 steps; bfloat16; completion-only loss; maximum sequence length 512.
- Evaluation remains 32 frozen math plus 16 protected-QA prompts with greedy decoding.
- The only implementation correction is explicit `encoding="utf-8", errors="replace"` for host capture of WSL stdout/stderr.
- All six workers enter the unchanged three-seed aggregate. The first four retain their original PIDs, outputs and checkpoints.

## Acceptance gates

- `continuation_integrity`: `frozen_partial_workers_verified eq 4`
- `treatment_materiality`: `matched_distinct_training_targets_verified eq True`
- `clean_base`: `fresh_base_workers eq 6`
- `paired_training`: `matched_pairs_per_arm_per_seed eq 128`
- `seed_coverage`: `completed_paired_seeds eq 3`
- `heldout_gain`: `mean_trace_math_gain_over_answer_only gt 0.0`
- `directional_repeatability`: `seeds_with_nonnegative_trace_math_gain ge 2`
- `protected_regression`: `mean_protected_qa_regression_vs_answer_only le 0.05`
- `service_recovery`: `service_and_embedding_restored eq True`

## Abort conditions

- Any frozen partial-worker, checkpoint, manifest, base-model or corpus hash differs.
- Any imported worker is edited or regenerated.
- Either seed-20260826 worker receives different task order, targets or hyperparameters.
- A new worker reports pre-existing PEFT modules, fails, diverges or produces fewer than 48 held-out samples.
- Port 8081 becomes unhealthy or the persistent serving tuple cannot be restored.

## Allowed claims

- `TRACE_DISTILLATION_FALSE_NEGATIVE_CONFIRMED_R3`
- `TRACE_DISTILLATION_REJECTED_R3`

Claims outside these codes are forbidden even if a metric looks favorable.

Partial observed scores are not acceptance gates and did not alter this continuation. The executor stops at `EXECUTED`; AGY must review the combined evidence.
