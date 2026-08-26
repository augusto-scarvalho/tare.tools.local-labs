# BACKLOG-DISTILL-REAL-01 Result

Task: Rebuild DISTILL-00 from actual teacher and student generations  
Evidence class: `distillation`  
Executor: AGY / Gemini 3.7 Flash High  
Date: 2026-08-25  

## Verdict

`DISTILLATION_REJECTED`

The historical `DISTILL-00-MOE-CONCISE-2026-08-25` claim was audited and determined to have been based on synthetic randomized token counts (`random.randint`) and ungrounded hard-coded accuracy figures. 

In this remediation experiment, genuine paired inference was executed across all 32 frozen GSM8K tasks comparing actual teacher completions (`ThinkingCap-27B-Q4 / Fable-TC` traces) against actual student generations (`Qwen/Qwen3.5-0.8B-Base` loaded with the verified `target_mlp_only` LoRA adapter on RTX 3090).

The empirical evaluation falsified the concise distillation hypothesis:
- The 0.8B student achieved 14/32 (43.75%) accuracy versus the 27B teacher's 32/32 (100.0%), yielding an accuracy delta of **-0.5625**, missing the non-inferiority gate ($\ge -0.03$).
- The student required a median of 192.0 tokens per reasoning trace versus the teacher's median of 95.0 tokens (token reduction of **-102.11%**), missing the token reduction gate ($\ge +20\%$).

Consequently, the distillation claim is definitively **REJECTED**.

## Acceptance Gates Summary

| Gate ID | Metric | Operator | Threshold | Actual | Verdict |
|---|---|:---:|:---:|:---:|:---:|
| `no_fabricated_metrics` | `scores_derived_from_raw_samples` | `eq` | `true` | **`true`** | **PASS** |
| `paired_panel` | `paired_scored_samples` | `ge` | 32 | **32** | **PASS** |
| `accuracy_noninferiority` | `student_accuracy_delta` | `ge` | -0.03 | **-0.5625** | **FAIL** |
| `token_reduction` | `median_reasoning_token_reduction` | `ge` | 0.20 | **-1.0211** | **FAIL** |

## Paired Model Comparison Matrix (32 Frozen GSM8K Tasks)

| Metric | Teacher (`ThinkingCap-27B-Q4`) | Student (`Qwen-0.8B + LoRA MLP`) | Delta / Comparison |
|---|:---:|:---:|:---:|
| **GSM8K Accuracy** | **32 / 32 (100.0%)** | 14 / 32 (43.75%) | **-56.25% (FAIL)** |
| **Median Reasoning Tokens** | **95.0 tokens** | 192.0 tokens | **+102.11% more tokens (FAIL)** |
| **Mean Reasoning Tokens** | **107.16 tokens** | 172.00 tokens | +60.5% more tokens |
| **Raw Sample Traceability** | 32 verified pairs | 32 verified pairs | 100% paired coverage |

## Key Findings

1. **Falsification of Synthetic Claims**: Replacing hard-coded/randomized artifacts with physical generations proves that compact 0.8B models do not match 27B teacher accuracy or compactness on math reasoning without extensive long-horizon post-training.
2. **Deterministic Reproducibility**: 100% agreement between generation extractions and independent deterministic re-scoring across all 32 paired evaluations.
3. **Fail-Closed Accountability**: Negative scientific outcomes are preserved as authoritative falsifications rather than omitted.

## Scope Boundaries & Forbidden Claims

- **No Distillation Superiority Claim**: Distillation did not compress token length or preserve teacher-level accuracy.
- **No Unpaired Evidence**: All metrics are strictly derived from the 32 paired problem evaluations.

## Evidence Artifacts

- Execution Receipt: [`raw/receipt.json`](raw/receipt.json)
- Actual Scores Ledger: [`raw/actual_scores.json`](raw/actual_scores.json)
- Paired Raw Sample Generations: [`raw/samples.jsonl`](raw/samples.jsonl)
- Teacher Sample Generations: [`raw/teacher_samples.json`](raw/teacher_samples.json)
- Student Sample Generations: [`raw/student_samples.json`](raw/student_samples.json)
- Dataset Ledger: [`raw/dataset_hashes.json`](raw/dataset_hashes.json)
- Model Ledger: [`raw/model_hash.json`](raw/model_hash.json)
- Independent Evaluation: [`raw/independent_evaluation.json`](raw/independent_evaluation.json)
