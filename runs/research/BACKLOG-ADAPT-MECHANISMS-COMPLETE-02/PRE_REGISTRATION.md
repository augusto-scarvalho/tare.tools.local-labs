# BACKLOG-ADAPT-MECHANISMS-COMPLETE-02 preregistration

Task: Complete the five-mechanism ADAPT matrix from the frozen missing arm
Evidence class: `artifact_requalification`

## Hypothesis

The incomplete R1 matrix can be completed without retraining: its exact missing
`adapt01/lokr_5ep` checkpoint was independently evaluated on the same frozen
32-math/16-QA panel. Joining those 48 rows to the 720 R1 rows will yield 16
complete arm instances, 768 independently reproducible scores and a normalized
historical service restoration match.

## Frozen inputs

- `runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/receipt.json`
- `runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/samples.jsonl`
- `runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/training_trace.json`
- `runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/artifact_hashes.json`
- `runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/service_maintenance.json`
- `runs/research/BACKLOG-ADAPT01-640-EVAL-01/raw/receipt.json`
- `runs/research/BACKLOG-ADAPT01-640-EVAL-01/raw/samples.jsonl`
- `runs/research/BACKLOG-ADAPT01-640-EVAL-01/raw/service_maintenance.json`
- `tools/analysis/a2_stats.py`
- `tools/benchmarks/normal_qa_ab.py`
- `runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl`

- Admission: `2c850cc5990b0b6f86b5e3af576067748dc12bd0230e097984736ce72e5f1ff2`.
- R1 receipt/samples/training/artifacts/service: `dd975197993bab7943ea2407a664f20fda927bb6fc581714eba093f7e93be0c6`, `76089e6911d55a8cad5ba75162c898bdf6ad62e8fe561749553235f06de90bc6`, `332ccf6d14e8e45ef5c7edeb7cbff8e2d0654e1ded45509c5a8f72bd87314a48`, `7d357cd1928066e45eb519825a0c89dc7d9dfed631fc3afaf1fb9b1dfb39519d`, `db3b8f9f686a65543a5e268e5a6f5e5194412823911d6302ed963f42daa90dd3`.
- Missing-arm receipt/samples/service: `cb0180cc424db42e5a8a81fd6bf60c66b5b61c5a2c029631e909418276e8d170`, `8b5b5a870b3a6f20ea8adadca76f093dcae9e1a725cf98a4d37e70f38caa1d33`, `b4915cd2cee96eb8998627ae61f019da87639bf795eea196dbeb7930d042f497`.
- Math scorer, QA scorer and QA tasks: `d63e4c0e5fcb820d912c2492fa1e4f50c94b2488970c8fa1278c749e6b0bd459`, `b249c4efd4d2d52ed2da748dbaba30ceb53833e60de15fea79e4b41070d3f641`, `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`.

## Command

```powershell
python tools/research/run_adapt_mechanisms_complete_r2.py --outdir runs/research/BACKLOG-ADAPT-MECHANISMS-COMPLETE-02
```

## Factors

- Preserve all 720 R1 rows unchanged and add exactly the 48 frozen lokr_5ep rows.
- Independently rescore math from raw text/gold and protected QA from raw
  text/task using the frozen scorers.
- Require 32 math plus 16 QA rows for each mechanism/arm and no duplicate key.
- Normalize service identity to executable, argv, active state, restart count
  and endpoint health; ignore PID and start-time serialization.

## Acceptance gates

- `source_integrity`: `source_receipts_and_artifacts_verified eq True`
- `mechanism_coverage`: `fresh_mechanisms_completed eq 5`
- `training_coverage`: `fresh_training_arms eq 12`
- `evaluation_coverage`: `fresh_scored_generations eq 768`
- `arm_coverage`: `complete_arm_instances eq 16`
- `seed_control`: `fresh_seed_verified eq 20260827`
- `independent_aggregate`: `independent_score_match eq True`
- `artifact_identity`: `hashed_adapter_artifacts ge 13`
- `service_restore`: `normalized_original_service_restored eq True`
- `embedding_integrity`: `embedding_health eq 200`

## Abort conditions

Abort on any source/hash drift, duplicate or missing row, scorer mismatch,
checkpoint/training-arm inconsistency, or non-equivalent historical service
argv. This is a read-only synthesis: no GPU, training, inference or service
mutation is allowed.

## Allowed claims

- `ADAPT01_05_MECHANISMS_COMPLETED_R2`
- `ADAPT01_05_MECHANISMS_MIXED_R2`

Claims outside these codes are forbidden even if a metric looks favorable.
