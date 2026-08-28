# BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01 preregistration

Task: Select one trace-distilled deployment finalist and validate it on a third untouched panel
Evidence class: `distillation`

## Hypothesis

Seed `20260832`, selected before this run by zero R7 protected-QA regression
and the highest combined trace score across the two prior panels, will retain a
strict positive trace-over-answer-only gain on a third untouched 256-task
panel. Its paired-bootstrap lower 95% bound must exceed zero and trace accuracy
must reach at least 40%.

## Frozen inputs

- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-07/raw/checkpoint_hashes.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-07/raw/actual_scores.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-07/raw/student_samples.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-07/raw/training_pairs.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-07/raw/receipt.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-08/raw/actual_scores.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-08/raw/dataset_hashes.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-08/raw/receipt.json`
- `tools/research/run_trace_distillation_replication_r8.py`
- `workloads/gsm8k.jsonl`

- Admission: `220160c63e606ce905f80c0c07b41d163c1b73da1e3b57805fb209b2c1357beb`.
- R7 checkpoint ledger: `57364aaba37c39771aaf216950bfff6df1282735b641b0c682ded89ffa8aaf4c`.
- R7 scores: `0171dcfcd70334a780a16337469f200656ec3b1d7c567889393c419cad9bae1e`.
- R7 student samples: `5283e8e1a66227d71d7a0c5847bd2c147397f580cfaa9c22520edfc65128e19b`.
- R7 training pairs: `5c3f0d5fd80d97351839bca1e38685e5e21b3357dfa56077f44f02b857bfe4cc`.
- R7 receipt: `782d9e58a97c5ac55dd6ebc2d62c67f9e003af415fb62e14ac8124718ea93b3a`.
- R8 scores: `15d359d9701a10ed449b8b325c1be93bb002a1ba685ec1bb97c21f3f30efed45`.
- R8 dataset ledger: `934dfa0b0e45ce73deab24a2e1ee5684a7a223843290296755e8eb83f5e19171`.
- R8 receipt: `eb4cff3c9d5022887f2621bdf0c303b4aca807e9449e8c600860bd72a046b990`.
- R8 runner: `0ad1f687c8ed1b9f0d923a61fa853a47ece9b35dbaaaa0b72b483e9793cbcbec`.
- GSM8K source: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`.

## Command

```powershell
python tools/research/run_trace_distillation_deploy_finalist.py --outdir runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01
```

## Factors

- Frozen selection: among R7 seeds with QA regression <=0, maximize the sum of
  trace-correct counts on R7 and R8. Candidates score `220, 223, 210`, selecting
  seed `20260832` without using the third panel.
- Arms: immutable `seed_20260832_answer_only` and `seed_20260832_full_trace`.
- Third panel: the next 256 teacher-disjoint GSM8K IDs after the two prior
  panels; ID-list SHA-256
  `73024245450c6158c150d654243e30ef26e562027dcc6514abd096862d0a69fe`.
- The panel is disjoint from teacher traces, training pool, R7 and R8 panels.
- 512 fresh deterministic generations on RTX 3090; same base, evaluator,
  prompt, 192-token cap and paired-bootstrap procedure as R8.
- R7 selected-seed QA regression is imported by hash; no new QA selection.

## Acceptance gates

- `source_integrity`: `r7_r8_sources_and_checkpoints_verified eq True`
- `selection_reproducibility`: `selected_seed eq 20260832`
- `panel_isolation`: `third_panel_disjoint_from_training_and_prior_panels eq True`
- `checkpoint_coverage`: `immutable_checkpoints_evaluated eq 2`
- `evaluation_coverage`: `fresh_third_panel_generations eq 512`
- `finalist_gain`: `paired_bootstrap_95ci_lower_trace_minus_answer gt 0.0`
- `finalist_absolute`: `trace_third_panel_accuracy ge 0.4`
- `protected_retention`: `imported_selected_seed_qa_regression le 0.05`
- `independent_score`: `independent_rescore_match eq True`
- `service_recovery`: `service_and_embedding_restored eq True`

## Abort conditions

- Abort on any source/checkpoint/base-model mismatch, seed-selection mismatch,
  panel overlap/hash mismatch, incomplete worker, insufficient VRAM, scorer
  disagreement, embedding failure or service restoration failure.
- Wrong answers and a negative gain remain evidence and do not abort.
- No fourth panel or post-result seed selection is allowed.

## Allowed claims

- `TRACE_DISTILLATION_DEPLOYMENT_FINALIST_CONFIRMED_R1`
- `TRACE_DISTILLATION_DEPLOYMENT_FINALIST_NOT_CONFIRMED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.
