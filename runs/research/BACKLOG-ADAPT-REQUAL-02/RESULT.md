# BACKLOG-ADAPT-REQUAL-02 result

Status: `EXECUTED` — independent review pending  
Execution window: 2026-08-25 21:21:36Z to 22:13:13Z  
Hardware: NVIDIA RTX 3090  
Receipt: `raw/receipt.json`, SHA-256 `8bc38d1f2cb5ef60f53ddb989e5c0aa1104b81359efbc5a2c8e4bbd0d92bc876`

## Outcome

The process-isolated requalification completed all 17 preregistered workers and produced the complete 14-arm evaluation set: one clean base control plus 13 saved adapters. All seven preregistered acceptance gates passed. The packet remains at `EXECUTED`; this document does not perform independent review, select a finalist, or claim promotion.

## Isolation result

- Every worker started a distinct WSL Python process.
- All 17 workers reported zero pre-existing PEFT/tuner modules on the freshly loaded base model.
- Forward smoke order: `base`, `lokr_1ep`, `target_mlp_only`.
- Reverse smoke order: `target_mlp_only`, `lokr_1ep`, `base`.
- The semantic projections were byte-identical for all three smoke arms in both orders (`order_invariant = true`).
- Thirteen adapter identities and all 26 corresponding config/weight hashes matched the frozen ledger.

## Descriptive scores

Each arm contains 32 unique frozen GSM8K tasks and 16 unique protected-QA tasks. These values are descriptive only; no performance-selection rule was preregistered.

| Arm | Math | Math rate | Protected QA | QA rate |
|---|---:|---:|---:|---:|
| `base` | 6/32 | 18.75% | 3/16 | 18.75% |
| `lokr_1ep` | 12/32 | 37.50% | 3/16 | 18.75% |
| `lokr_3ep` | 9/32 | 28.12% | 2/16 | 12.50% |
| `lokr_3ep_lr1e4` | 10/32 | 31.25% | 2/16 | 12.50% |
| `lokr_5ep` | 8/32 | 25.00% | 2/16 | 12.50% |
| `target_all_linear` | 10/32 | 31.25% | 3/16 | 18.75% |
| `target_attn_only` | 9/32 | 28.12% | 5/16 | 31.25% |
| `target_mlp_only` | 10/32 | 31.25% | 3/16 | 18.75% |
| `target_qv_gate` | 10/32 | 31.25% | 3/16 | 18.75% |
| `soft_prompts` | 7/32 | 21.88% | 0/16 | 0.00% |
| `lokr_prior_lambda02` | 6/32 | 18.75% | 2/16 | 12.50% |
| `lokr_prior_lambda05` | 7/32 | 21.88% | 1/16 | 6.25% |
| `lokr_unreg_5ep` | 12/32 | 37.50% | 3/16 | 18.75% |
| `disjoint_composite` | 12/32 | 37.50% | 5/16 | 31.25% |

The clean result does not reproduce the earlier `target_mlp_only` score of 15/32 math and 4/16 QA. Under process isolation it scored 10/32 and 3/16. This materially invalidates the earlier finalist ranking as evidence for downstream training, but this packet itself is not preregistered to select a replacement.

## Gate results

| Gate | Actual | Pass |
|---|---:|:---:|
| `artifact_identity` | 13 hashed adapters | yes |
| `clean_base_per_arm` | 17/17 clean workers | yes |
| `isolation_smoke` | order invariant | yes |
| `frozen_math_panel` | 32 scored samples per arm | yes |
| `frozen_qa_panel` | 16 scored samples per arm | yes |
| `base_control` | present | yes |
| `independent_score` | host recomputation matched worker flags | yes |

The host-side structural recount separately found 672 sample rows, 14 distinct arms, 32 unique math task IDs per arm and 16 unique QA task IDs per arm. The repository test suite passed 87/87 tests and `backlog_pipeline.py gate` returned `PASS` after execution.

## Service restoration

`llm-inference.service` was stopped through systemd to free VRAM, while the embedding endpoint on port 8081 remained healthy. After execution:

- inference port 8080 returned `{"status":"ok"}`;
- embedding port 8081 returned `{"status":"ok"}`;
- `llm-inference.service` was `active/running` with `NRestarts=0`;
- the final executable and argument vector matched the initial serving command;
- the process PID changed from 11434 to 26576, as expected after a controlled stop/start.

The raw maintenance record contains `exec_start_restored = false` because the implementation compared the complete `systemctl show` rendering, which embeds volatile start time, stop time and PID fields. The executable and `argv[]` portions are unchanged. Independent review should assess the captured initial/final records directly rather than treating that raw boolean as a command mismatch.

## Claim boundary

No `REVIEW.json` was authored and no `VERIFIED`, `PROMOTED` or `REJECTED` transition was performed by the executor. The next admissible action is review by an independent actor. Only after that review may the packet receive one of its allowed claim codes:

- `ARTIFACT_REQUALIFIED_R2`
- `ARTIFACT_REQUALIFICATION_R2_REJECTED`

The prior `target_mlp_only` finalist designation must not be used to authorize `BACKLOG-ADAPT-TRAIN-01`, `BACKLOG-DISTILL-REAL-01` or `BACKLOG-ADAPT-TRACE-DISTILL-01` without a separately preregistered selection decision.
