# BACKLOG-ADAPT01-640-EVAL-01 preregistration

Task: Evaluate the fresh ADAPT-01 640-step LoKr arm excluded by the historical probe
Evidence class: `artifact_requalification`

## Hypothesis

The fresh seed-20260827 640-step LoKr adapter omitted by the historical driver can be evaluated on the same frozen 32-math plus 16-QA panel. It is promoted only if it independently satisfies every original behavioral gate; its failed internal training gate is not treated as a substitute for behavioral evidence.

## Frozen inputs

- `runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/receipt.json`
- `runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/mechanisms/adapt01/lokr_5ep/adapter/adapter_model.safetensors`
- `runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/mechanisms/adapt01/lokr_5ep/metrics.json`
- `workloads/gsm8k.jsonl`
- `runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl`

- The packet freezes the parent receipt, fresh adapter tensor/config, training metrics, math workload and protected QA tasks listed above.
- The base comparator is the seed-20260827 base result in the immutable parent receipt: 8/32 math and 3/16 QA.
- The official local model is `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe`.

## Command

```powershell
python tools/research/run_adapt01_640_eval.py --outdir runs/research/BACKLOG-ADAPT01-640-EVAL-01
```

## Factors

- Load only the fresh 640-step adapter; do not retrain or modify it.
- Generate greedily for the same 32 held-out math tasks and 16 protected QA tasks selected by seed 20260827.
- Recompute every counter from raw rows and compare with the worker summary.
- Leave the active inference and embedding services untouched; require identical MainPID and zero restarts.

## Acceptance gates

- `panel_coverage`: `scored_generations eq 48`
- `target_absolute`: `target_correct ge 16`
- `target_gain`: `target_gain_over_base ge 3`
- `qa_retention`: `protected_pass ge 2`
- `natural_eos`: `natural_eos ge 40`
- `length_control`: `target_teacher_length_ratio le 1.25`
- `independent_score`: `independent_score_match eq 1`
- `runtime_unchanged`: `serving_process_unchanged eq 1`

## Abort conditions

- Abort on source hash drift, missing adapter, fewer than 48 rows, scorer disagreement, CUDA failure, or any service change.
- No threshold, prompt, seed, maximum generation length or scorer may change after observation.

## Allowed claims

- `ADAPT01_640_ARM_PROMOTED_R1`
- `ADAPT01_640_ARM_REJECTED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.

This packet may claim artifact behavior only, not fresh training, cross-seed repeatability, production readiness or out-of-panel capability.
