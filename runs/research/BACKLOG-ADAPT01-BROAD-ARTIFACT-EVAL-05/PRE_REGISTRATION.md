# BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-05 preregistration

Task: Resolve LoKr panel uncertainty with one final third panel and a frozen three-panel synthesis
Evidence class: `artifact_requalification`

## Hypothesis

Across exactly three equal-sized, pairwise-disjoint and teacher-disjoint GSM8K
panels, the immutable 384-step `lokr_3ep_lr1e4` artifact has a positive
clean-base accuracy delta whose equal-panel stratified-bootstrap 95% lower
bound is above zero. The newly generated third panel must also have a positive
point delta. This is the final panel extension regardless of outcome.

R3 and R4 math outputs are imported by hash. Only the third panel's 512 paired
math responses are generated fresh. R3's 48-task protected QA remains the
frozen retention control.

## Frozen inputs

- `runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-03/raw/merged_worker.json`
- `runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-03/raw/receipt.json`
- `runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-04/raw/combined_worker.json`
- `runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-04/raw/receipt.json`
- `runs/research/BACKLOG-ADAPT-MECHANISMS-RERUN-01/raw/mechanisms/adapt01/lokr_3ep_lr1e4/adapter`
- `workloads/gsm8k.jsonl`
- `runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json`
- `runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl`

- Admission SHA-256: `f0a4cd0de4a1192637eb288f84ed896f0cc13f21d403f41d64bd1ea1cbf6aeee`.
- R3 merged worker SHA-256: `7beecfc4a2970ce39f6e5a8343d4d6b23fd5e78fa8328b03195f3e61acdb6b2f`;
  receipt file SHA-256: `3ca20a2dbad797cc6ce1629d501802c6fa06ee8d67c8ab948f2959c62b298f40`.
- R4 combined worker SHA-256: `9556cffa5c23554a0e47184b8ef4af0a59114c575ef053954ccdbf674529e5b7`;
  receipt file SHA-256: `185935ee19679509b780150d683707ef294818dd6711b7076b09ac5cb835f8e8`.
- R4 preregistration SHA-256: `73cfe7178bee1770a55b41982836ac836d5e0f4b011e95e151cdce0d09db7d5c`;
  implementation SHA-256: `ac1f7453cb873fba1ba04d544f87776e975da6a41c27956a3e310d76c6e1c539`.
- Adapter weights SHA-256: `7f6d082243f6b406259791dc15a65e4b092b48597fad9b68018d507872ad8fa7`;
  config SHA-256: `08cf4d254e2a6c9aba9d34ba6a0c76926b478d7cd0ad771062acefb71a31d934`.
- GSM8K SHA-256: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`;
  teacher SHA-256: `dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e`;
  QA SHA-256: `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`.
- Ordered panel hashes: R3 `78a3b7ef26cdf932b79eb6f64dfb576d66770d30a5ce0fd3251536cb6e76901f`;
  R4 `4c88bd4c27eb8fea9240e11503ef781e744313c5de7b1f0391bb680f7e3379bd`;
  R5 `73024245450c6158c150d654243e30ef26e562027dcc6514abd096862d0a69fe`.
- Actual ordered QA-ID SHA-256: `5377ee57e27a3480fdad26c05cc7cc13b7e177c69abdda77795f898d43df45f3`.

## Command

```powershell
python tools/research/run_adapt01_broad_artifact_eval_r5.py --outdir runs/research/BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-05
```

## Factors

- Fresh arms: clean base and immutable LoKr adapter, greedy decoding, identical
  prompts and maximum 192 new tokens on the third 256-task panel.
- Third panel: the next 256 lowest numeric GSM8K IDs after R3 and R4 among IDs
  absent from every teacher row; 512 fresh generations.
- Primary estimator: mean paired accuracy delta across all 768 tasks with equal
  256-task panel weights. Bootstrap uses seed `2026082705`, 20,000 replicates,
  and independently resamples 256 prompt differences with replacement inside
  each of the three panels before averaging the three panel means.
- Secondary fixed checks: third-panel point delta above zero and at least two
  of three panel point deltas positive.
- Protected retention imports and independently rescores R3's complete 48-task
  QA samples. Serving may stop only for VRAM; 8080 and 8081 must recover.

## Acceptance gates

- `source_integrity`: `r3_r4_source_hashes_verified eq True`
- `artifact_identity`: `artifact_hashes_verified eq True`
- `panel_isolation`: `three_panels_pairwise_and_teacher_disjoint eq True`
- `evaluation_coverage`: `fresh_third_panel_paired_math_generations eq 512`
- `pooled_gain`: `stratified_bootstrap_95ci_lower_three_panel_math_gain gt 0.0`
- `third_panel_direction`: `third_panel_math_gain gt 0.0`
- `panel_repeatability`: `panels_with_positive_math_gain ge 2`
- `protected_retention`: `imported_r3_protected_qa_regression le 0.05`
- `independent_score`: `independent_rescore_match eq True`
- `service_recovery`: `service_and_embedding_restored eq True`

## Abort conditions

- Any frozen source or panel hash differs, or panels overlap each other/teacher.
- R3/R4 do not each contain exactly 256 paired math rows with the frozen IDs.
- The fresh worker does not contain exactly 256 math and zero QA rows per arm.
- Independent rescoring disagrees with any imported or fresh correctness bit.
- Embedding becomes unhealthy, VRAM remains insufficient after maintenance, or
  the qualified serving baseline cannot be restored.
- No further panel successor may be opened to rescue the result; no receipt is
  emitted from partial output or an incomplete evidence map.

## Allowed claims

- `ADAPT01_384_ARTIFACT_THREE_PANEL_GAIN_R5`
- `ADAPT01_384_ARTIFACT_THREE_PANEL_GAIN_NOT_CONFIRMED_R5`

Claims outside these codes are forbidden even if a metric looks favorable.
