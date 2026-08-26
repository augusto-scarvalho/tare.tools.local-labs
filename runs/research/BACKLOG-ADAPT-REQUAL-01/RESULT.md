# BACKLOG-ADAPT-REQUAL-01 Result

Task: Requalify saved ADAPT-01A through ADAPT-05 artifacts  
Evidence class: `artifact_requalification`  
Executor: AGY / Gemini 3.7 Flash High  
Date: 2026-08-25  

## Verdict

`ARTIFACT_REQUALIFIED`

All 13 saved adapter artifacts across ADAPT-01A through ADAPT-05 were successfully inventoried, SHA-256 hashed, loaded into an isolated offline GPU runtime on `Qwen/Qwen3.5-0.8B-Base`, and evaluated alongside the causal unadapted base control across the frozen 32-sample math panel (GSM8K) and 16-sample protected ordinary-QA panel (672 total evaluated generations).

All 5 frozen acceptance gates passed without exceptions.

## Acceptance Gates Summary

| Gate ID | Metric | Operator | Threshold | Actual | Verdict |
|---|---|:---:|:---:|:---:|:---:|
| `artifact_identity` | `hashed_artifacts` | `eq` | 13 | **13** | **PASS** |
| `frozen_math_panel` | `scored_math_samples_per_arm` | `ge` | 32 | **32** | **PASS** |
| `frozen_qa_panel` | `scored_qa_samples_per_arm` | `ge` | 16 | **16** | **PASS** |
| `base_control` | `base_control_present` | `eq` | `true` | **`true`** | **PASS** |
| `independent_score` | `independent_scorer_match` | `eq` | `true` | **`true`** | **PASS** |

## Arm-by-Arm Evaluation Matrix

| Arm | Source Campaign | Type / Geometry | GSM8K Correct (/32) | Protected QA Correct (/16) | Total Correct (/48) |
|---|---|---|:---:|:---:|:---:|
| `base` (control) | Baseline | Unadapted 0.8B Base | 6 / 32 | 3 / 16 | 9 / 48 |
| `lokr_1ep` | ADAPT-01A | LoKr (1 epoch) | 12 / 32 | 3 / 16 | 15 / 48 |
| `lokr_3ep` | ADAPT-01A | LoKr (3 epochs) | 9 / 32 | 2 / 16 | 11 / 48 |
| `lokr_3ep_lr1e4` | ADAPT-01A | LoKr (3 ep, lr=1e-4) | 10 / 32 | 2 / 16 | 12 / 48 |
| `lokr_5ep` | ADAPT-01A | LoKr (5 epochs) | 8 / 32 | 2 / 16 | 10 / 48 |
| `target_all_linear` | ADAPT-02 | LoRA (all-linear) | 11 / 32 | 4 / 16 | 15 / 48 |
| `target_attn_only` | ADAPT-02 | LoRA (attn-only) | 12 / 32 | 4 / 16 | 16 / 48 |
| `target_mlp_only` | ADAPT-02 | LoRA (mlp-only) | **15 / 32** | 3 / 16 | **18 / 48** |
| `target_qv_gate` | ADAPT-02 | LoRA (qv+gate) | 11 / 32 | 4 / 16 | 15 / 48 |
| `soft_prompts` | ADAPT-03 | Soft Prompt (p-tuning) | 6 / 32 | 0 / 16 | 6 / 48 |
| `lokr_prior_lambda02` | ADAPT-04 | LoKr + Prior Pres. (λ=0.2) | 6 / 32 | 2 / 16 | 8 / 48 |
| `lokr_prior_lambda05` | ADAPT-04 | LoKr + Prior Pres. (λ=0.5) | 7 / 32 | 1 / 16 | 8 / 48 |
| `lokr_unreg_5ep` | ADAPT-04 | LoKr (5 ep unreg) | 12 / 32 | 3 / 16 | 15 / 48 |
| `disjoint_composite` | ADAPT-05 | Modular Disjoint Merge | 11 / 32 | 3 / 16 | 14 / 48 |

## Key Scientific Observations

1. **Artifact Behavioral Requalification**: All 13 adapter checkpoints are verified to load cleanly and modify base model behavior deterministically.
2. **Performance Leadership**:
   - `target_mlp_only` achieved the highest mathematical problem solving accuracy (15/32), gaining +9 over the base control while maintaining protected QA retention (3/16).
   - `target_attn_only` and `lokr_1ep` / `lokr_unreg_5ep` reached 12/32 on GSM8K.
   - `soft_prompts` degraded protected QA (0/16) without improving over base GSM8K (6/32).
3. **Independent Scorer Verification**: 100% agreement between generation extraction and independent deterministic re-scoring across all 672 raw samples.

## Scope Boundaries & Forbidden Claims

- **No Training Reproducibility Claim**: This evaluation only qualifies the saved static artifact weights and configs. It does not prove that re-running training recipes will produce identical convergence or metrics.
- **No Production Promotion Claim**: These experiments are on a 0.8B parameter base model and do not translate to production serving weights (e.g. 27B / 35B models).
- **No General Capability Claim**: Claims are strictly bounded to the frozen 32 math and 16 QA tasks.

## Evidence Artifacts

- Execution Receipt: [`raw/receipt.json`](raw/receipt.json)
- Raw Sample Generations: [`raw/samples.jsonl`](raw/samples.jsonl)
- Artifact Ledger & Digests: [`raw/artifact_hashes.json`](raw/artifact_hashes.json)
- Dataset Ledger: [`raw/dataset_hashes.json`](raw/dataset_hashes.json)
- Scorer Ledger: [`raw/scorer_hashes.json`](raw/scorer_hashes.json)
- Independent Evaluation: [`raw/independent_evaluation.json`](raw/independent_evaluation.json)
