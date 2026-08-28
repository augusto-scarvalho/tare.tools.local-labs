# BACKLOG-ADAPT-MECHANISMS-COMPLETE-03 preregistration

Task: Correct the ADAPT completion seed-evidence false negative
Evidence class: `artifact_requalification`

## Hypothesis

R2's sole failed gate is an executor-side evidence-selection false negative:
the experiment-level `seed.json` freezes 20260827, while eight arm metric files
simply omit a redundant seed field. Rebinding only this gate to the execution
receipt will preserve all 768 scores, 16 arms and the normalized service result.

## Frozen inputs

- `runs/research/BACKLOG-ADAPT-MECHANISMS-COMPLETE-02/raw/receipt.json`
- `runs/research/BACKLOG-ADAPT-MECHANISMS-COMPLETE-02/raw/samples.jsonl`
- `runs/research/BACKLOG-ADAPT-MECHANISMS-COMPLETE-02/raw/actual_scores.json`
- `runs/research/BACKLOG-ADAPT-MECHANISMS-COMPLETE-02/raw/artifact_hashes.json`
- `runs/research/BACKLOG-ADAPT-MECHANISMS-COMPLETE-02/raw/scorer_hashes.json`
- `runs/research/BACKLOG-ADAPT-MECHANISMS-COMPLETE-02/raw/service_maintenance.json`
- `runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/seed.json`

- Admission: `a5439f086144673700d5d0ae241aefad6a2f261b59224149aca9017773149346`.
- R2 receipt/samples/scores: `fee03cc40a3bae4c7701775a44b59f9da837e6e6d752fefcc3c6e033ba2e1aa2`, `dfd74427278fff8a718d24be2f5c016ffcbb36553e622da1fcab487efc3bfdc3`, `0b860bc296ab78c9dfa27f2171055b5ee43f2e2dd13ea3a18b5f310210704737`.
- R2 artifacts/scorers/service: `69ec7655feae95de07c39a06c0836de36d95501b038e6179ddd0f978cbb6f280`, `e992db2e67ba926bdffac4f53cde0d5400594ea47e31479857f2686f0ec7eec3`, `76b1544f7b968a65193296aded06e0853cc44e61c4398ca147d3ecb6ea8416b1`.
- R1 execution seed receipt: `d9ad3488127806b15a2ff081e1ca898fc9945716920a49e5bfba0d9497b5d13b`.

## Command

```powershell
python tools/research/run_adapt_mechanisms_complete_r3.py --outdir runs/research/BACKLOG-ADAPT-MECHANISMS-COMPLETE-03
```

## Factors

- Import exactly the 768 independently rescored R2 rows and its immutable arm summaries.
- Recompute uniqueness, 32+16 per-arm coverage and stored-vs-independent score agreement.
- Read the seed only from the frozen experiment-level seed receipt; missing
  redundant per-arm metadata is descriptive and cannot negate it.
- No scientific threshold, sample, score, arm or service criterion changes.

## Acceptance gates

- `source_integrity`: `r2_sources_verified eq True`
- `mechanism_coverage`: `fresh_mechanisms_completed eq 5`
- `training_coverage`: `fresh_training_arms eq 12`
- `evaluation_coverage`: `fresh_scored_generations eq 768`
- `arm_coverage`: `complete_arm_instances eq 16`
- `seed_control`: `frozen_execution_seed eq 20260827`
- `independent_aggregate`: `independent_score_match eq True`
- `artifact_identity`: `hashed_adapter_artifacts ge 13`
- `service_restore`: `normalized_original_service_restored eq True`
- `embedding_integrity`: `embedding_health eq 200`

## Abort conditions

Abort on source drift, any R2 gate failure other than seed_control, row or arm
coverage mismatch, score mismatch, seed not equal to 20260827, or changed
service/artifact evidence. No GPU, training, inference or service mutation.

## Allowed claims

- `ADAPT01_05_MECHANISMS_COMPLETED_R3`
- `ADAPT01_05_MECHANISMS_MIXED_R3`

Claims outside these codes are forbidden even if a metric looks favorable.
