# BACKLOG-ADAPT-TRAIN-01 Result

Task: Reproduce training only for adapter finalists  
Evidence class: `model_training`  
Executor: AGY / Gemini 3.7 Flash High  
Date: 2026-08-25  

## Verdict

`TRAINING_REPRODUCED`

Training of the promoted adapter finalist geometry (`target_mlp_only` LoRA MLP targeting `gate_proj`, `up_proj`, `down_proj`, r=8, alpha=16) was successfully reproduced from scratch across two independent seeds (20260824 and 20260825) using 128 teacher-distilled math reasoning pairs on `Qwen/Qwen3.5-0.8B-Base`. Both seeds converged stably to low loss (< 1.0), achieved positive held-out accuracy gain on the 32-sample math panel (+0.0938 to +0.2188 over base model), and preserved protected ordinary-QA performance with 0.0% regression.

All 4 frozen acceptance gates passed cleanly without exceptions.

## Acceptance Gates Summary

| Gate ID | Metric | Operator | Threshold | Actual | Verdict |
|---|---|:---:|:---:|:---:|:---:|
| `fresh_output` | `preexisting_output_files` | `eq` | 0 | **0** | **PASS** |
| `repeatability` | `successful_repeated_seeds` | `ge` | 2 | **2** | **PASS** |
| `behavioral_gain` | `heldout_gain_over_base` | `gt` | 0.0 | **+0.0938** | **PASS** |
| `retention` | `protected_regression` | `le` | 0.05 | **0.0000** | **PASS** |

## Repeated Seeds Training & Evaluation Matrix

| Arm / Seed | Steps | Final Loss | Max VRAM (MB) | GSM8K Held-out (/32) | Protected QA (/16) | Delta GSM8K vs Base |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `base` (control) | 0 | — | — | 7 / 32 (21.9%) | 3 / 16 (18.8%) | — |
| `lora_mlp_seed_20260824` | 60 | **0.3004** | 5011.3 | **14 / 32 (43.8%)** | 4 / 16 (25.0%) | **+21.9% (+7)** |
| `lora_mlp_seed_20260825` | 60 | **0.7255** | 5011.3 | **10 / 32 (31.2%)** | 3 / 16 (18.8%) | **+9.4% (+3)** |

## Key Findings

1. **Training Reproducibility Verified**: Both training seeds executed from scratch in fresh output roots with zero preexisting artifacts converged smoothly without loss divergence or numerical instability.
2. **Behavioral Performance Consistency**:
   - Seed 20260824 reached 14/32 (+7 over base) with 4/16 on protected QA.
   - Seed 20260825 reached 10/32 (+3 over base) with 3/16 on protected QA.
   - The minimum math accuracy gain across seeds was +0.0938 (+3 problems), satisfying the `heldout_gain_over_base > 0` requirement.
3. **Protected QA Retention**: Protected QA retention was 100% (zero regression across both seeds versus the base model control).
4. **Independent Scorer Agreement**: 100% deterministic agreement between generation logging and independent re-scoring across all 144 evaluated samples.

## Scope Boundaries & Forbidden Claims

- **No GaLore Generalization Claim**: This result reproduces LoRA PEFT training for the MLP finalist. It makes no claim about the convergence or scaling of GaLore or alternative optimizers.
- **No Single-Seed Promotion**: Training reproducibility is established solely by the two repeated seeds.
- **No Artifact-Only Training Claim**: Findings are backed by complete step-by-step training loss traces and verified clean-reload model weights.

## Evidence Artifacts

- Execution Receipt: [`raw/receipt.json`](raw/receipt.json)
- Checkpoint Hashes: [`raw/checkpoint_hashes.json`](raw/checkpoint_hashes.json)
- Seed Ledger: [`raw/seed.json`](raw/seed.json)
- Training Traces: [`raw/training_trace.json`](raw/training_trace.json)
- Raw Sample Generations: [`raw/samples.jsonl`](raw/samples.jsonl)
- Dataset Ledger: [`raw/dataset_hashes.json`](raw/dataset_hashes.json)
- Base Model Ledger: [`raw/model_hash.json`](raw/model_hash.json)
- Independent Evaluation: [`raw/independent_evaluation.json`](raw/independent_evaluation.json)
