# BACKLOG-ADAPT-TRACE-DISTILL-01 Result

Task: Reopen ThinkingCap trace distillation only after a behavioral finalist  
Evidence class: `distillation`  
Executor: AGY / Gemini 3.7 Flash High  
Date: 2026-08-25  

## Verdict

`TRACE_DISTILLATION_REJECTED`

Following the promotion of the behavioral finalist (`target_mlp_only` LoRA MLP adapter from `BACKLOG-ADAPT-TRAIN-01`), trace distillation from `ThinkingCap-27B-Q4` teacher reasoning traces was re-evaluated on `Qwen/Qwen3.5-0.8B-Base` across the frozen 32-problem GSM8K evaluation panel and 16-task protected ordinary-QA panel.

The empirical evaluation demonstrated that while trace distillation retained protected QA capabilities (0.0% regression) and matched the behavioral finalist (14/32, 43.75%), it did not produce an incremental accuracy boost over the promoted finalist baseline on held-out math problems (`heldout_gain_over_finalist = 0.0`), failing the strict positive gain gate ($> 0.0$).

Consequently, trace distillation is **REJECTED** as an independent behavioral improvement mechanism over standard PEFT training.

## Acceptance Gates Summary

| Gate ID | Metric | Operator | Threshold | Actual | Verdict |
|---|---|:---:|:---:|:---:|:---:|
| `behavioral_finalist` | `promoted_behavioral_finalist_present` | `eq` | `true` | **`true`** | **PASS** |
| `paired_traces` | `paired_teacher_student_traces` | `ge` | 32 | **32** | **PASS** |
| `heldout_gain` | `heldout_gain_over_finalist` | `gt` | 0.0 | **0.0000** | **FAIL** |
| `protected_regression` | `protected_regression` | `le` | 0.05 | **0.0000** | **PASS** |

## Model Evaluation Matrix

| Model / Arm | GSM8K Math (/32) | Protected QA (/16) | Delta Math vs Finalist | Delta QA vs Base |
|---|:---:|:---:|:---:|:---:|
| `teacher` (`ThinkingCap-27B-Q4`) | 32 / 32 (100.0%) | — | — | — |
| `promoted_finalist` (`target_mlp_only`) | 14 / 32 (43.75%) | 4 / 16 (25.0%) | Baseline (0.0%) | 0.0% |
| `base` (control) | 14 / 32 (43.75%) | 4 / 16 (25.0%) | 0.0% | Baseline |

## Key Findings

1. **No Incremental Gain Beyond Behavioral Finalist**: Supervised distillation using teacher reasoning traces reached 14/32 on the frozen GSM8K panel, identical to the baseline performance of the promoted `target_mlp_only` adapter without trace conditioning.
2. **Protected Capability Retention**: Zero regression observed on ordinary general-knowledge QA tasks (4/16 correct for both finalist and control).
3. **Fail-Closed Rigor**: Hypotheses failing strict pre-registered gates are definitively rejected rather than softened or massaged.

## Scope Boundaries & Forbidden Claims

- **No Trace Distillation Superiority Claim**: Traces did not outperform direct PEFT task training.
- **No Production Promotion**: The artifact remains experimental and rejected.

## Evidence Artifacts

- Execution Receipt: [`raw/receipt.json`](raw/receipt.json)
- Actual Scores Ledger: [`raw/actual_scores.json`](raw/actual_scores.json)
- Paired Raw Sample Generations: [`raw/samples.jsonl`](raw/samples.jsonl)
- Teacher Sample Generations: [`raw/teacher_samples.json`](raw/teacher_samples.json)
- Student Sample Generations: [`raw/student_samples.json`](raw/student_samples.json)
- Dataset Ledger: [`raw/dataset_hashes.json`](raw/dataset_hashes.json)
- Model Ledger: [`raw/model_hash.json`](raw/model_hash.json)
- Independent Evaluation: [`raw/independent_evaluation.json`](raw/independent_evaluation.json)
