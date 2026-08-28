# BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-04 preregistration

Task: Replicate the broad LoKr artifact gain on a second teacher-disjoint math panel
Evidence class: `artifact_requalification`

## Hypothesis

The immutable 384-step `lokr_3ep_lr1e4` artifact reproduces its positive
clean-base accuracy delta on a second frozen panel of 256 GSM8K tasks that is
disjoint from every teacher row and from all R3 math tasks. Confirmation
requires the paired-bootstrap 95% lower bound to remain above zero.

R3's complete 48-task protected-QA samples are imported only by hash to retain
the already measured safety boundary; all 512 second-panel math generations
are fresh. This tests panel replication, not training repeatability.

## Frozen inputs

- `runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-03/raw/merged_worker.json`
- `runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-03/raw/receipt.json`
- `runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/mechanisms/adapt01/lokr_3ep_lr1e4/adapter`
- `workloads/gsm8k.jsonl`
- `runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json`
- `runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl`

- Admission SHA-256: `2630de4c9f952b6fcfbdd54ea8c314d31465eb80bf81a99e0cc1491c57eecdcf`.
- R3 merged worker SHA-256: `7beecfc4a2970ce39f6e5a8343d4d6b23fd5e78fa8328b03195f3e61acdb6b2f`.
- R3 receipt file SHA-256: `3ca20a2dbad797cc6ce1629d501802c6fa06ee8d67c8ab948f2959c62b298f40`.
- R3 preregistration SHA-256: `a4fd1091d47bcaf2da78cf0e719a2cab64ba935703f6bdbd2a02504c80642c4b`.
- R3 implementation SHA-256: `4aee5af93a5585977ce0dfe5c7b020218b4bc34130a9d1419607663f3b85794f`.
- Adapter weights SHA-256: `7f6d082243f6b406259791dc15a65e4b092b48597fad9b68018d507872ad8fa7`;
  config SHA-256: `08cf4d254e2a6c9aba9d34ba6a0c76926b478d7cd0ad771062acefb71a31d934`.
- GSM8K SHA-256: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`;
  teacher SHA-256: `dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e`;
  QA SHA-256: `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`.
- R3 math panel SHA-256: `78a3b7ef26cdf932b79eb6f64dfb576d66770d30a5ce0fd3251536cb6e76901f`.
- Second ordered math panel SHA-256: `4c88bd4c27eb8fea9240e11503ef781e744313c5de7b1f0391bb680f7e3379bd`.
- Actual ordered 48-task QA-ID SHA-256: `5377ee57e27a3480fdad26c05cc7cc13b7e177c69abdda77795f898d43df45f3`.

## Command

```powershell
python tools/research/run_adapt01_broad_artifact_eval_r4.py --outdir runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-04
```

## Factors

- Fresh arms: clean base and immutable LoKr adapter, evaluated sequentially in
  one isolated worker with greedy decoding and identical prompts.
- Fresh panel: the second 256 lowest numeric GSM8K IDs absent from all teacher
  rows, after excluding the 256 IDs consumed by R3; 512 fresh generations.
- Primary uncertainty: R2's unchanged 20,000-replicate paired prompt bootstrap.
- Protected retention: imported immutable R3 outputs for all 48 actual QA IDs,
  independently rescored again in R4.
- Runtime: RTX 3090 under WSL. Port 8080 may be stopped only for VRAM and must
  be restored with stable service identity; port 8081 stays healthy.

## Acceptance gates

- `source_integrity`: `r3_source_hashes_verified eq True`
- `artifact_identity`: `artifact_hashes_verified eq True`
- `panel_isolation`: `second_panel_disjoint_from_teacher_and_r3 eq True`
- `evaluation_coverage`: `fresh_paired_math_generations eq 512`
- `replicated_gain`: `paired_bootstrap_95ci_lower_math_gain gt 0.0`
- `directional_consistency`: `r3_and_r4_math_gain_positive eq True`
- `protected_retention`: `imported_r3_protected_qa_regression le 0.05`
- `independent_score`: `independent_rescore_match eq True`
- `service_recovery`: `service_and_embedding_restored eq True`

## Abort conditions

- Any source hash or frozen panel hash differs.
- The second panel overlaps the teacher set or the R3 panel, or has fewer than 256 tasks.
- Either fresh arm has other than 256 math and zero freshly generated QA samples.
- Imported R3 QA does not contain exactly the same ordered 48 IDs per arm.
- Independent math or QA rescoring disagrees with recorded correctness.
- Embedding becomes unhealthy, VRAM remains insufficient after bounded
  maintenance, or the qualified serving baseline cannot be restored.
- No receipt is emitted from a partial worker or incomplete evidence map.

## Allowed claims

- `ADAPT01_384_ARTIFACT_SECOND_PANEL_GAIN_R4`
- `ADAPT01_384_ARTIFACT_SECOND_PANEL_GAIN_NOT_CONFIRMED_R4`

Claims outside these codes are forbidden even if a metric looks favorable.
