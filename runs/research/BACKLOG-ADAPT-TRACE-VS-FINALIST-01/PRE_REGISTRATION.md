# BACKLOG-ADAPT-TRACE-VS-FINALIST-01 preregistration

Task: Compare replicated full-trace checkpoints against the reproduced behavioral finalist
Evidence class: `distillation`

## Hypothesis

As an end-to-end checkpoint family, the seven immutable full-trace SFT models
from R7/R8 will have higher mean math accuracy than the two immutable
behavioral-finalist reproductions from ADAPT-TRAIN-01 across the same two
teacher-disjoint 256-task panels. The hierarchical-bootstrap 95% lower bound
of trace minus behavioral accuracy must exceed zero, and the point difference
must be positive on both panels. Mean protected-QA accuracy may be at most five
percentage points worse.

This is a practical artifact comparison, not a training-budget-controlled
causal comparison: the trace family used 504 steps while the historical
behavioral finalists used 60. Trace outputs are imported by hash; only the two
previously unevaluated behavioral checkpoints generate fresh broad-panel data.

## Frozen inputs

- `runs/research/BACKLOG-ADAPT-TRAIN-01/raw/checkpoint_seed_20260824`
- `runs/research/BACKLOG-ADAPT-TRAIN-01/raw/checkpoint_seed_20260825`
- `runs/research/BACKLOG-ADAPT-TRAIN-01/raw/receipt.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-07/raw/student_samples.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-07/raw/receipt.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-08/raw/student_samples.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-08/raw/receipt.json`
- `workloads/gsm8k.jsonl`
- `runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json`
- `runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl`

- Admission SHA-256: `759be8cb8abc548c52f4201fa75b03b193ac0c0ba0ac0f23fbedc51c4d11cac9`.
- ADAPT-TRAIN-01 receipt SHA-256: `903c723f3d63130cf06a5e501498451beee0cee34a8aa71d6f9de36faeb602b8`.
- Behavioral seed 20260824 config/weights SHA-256:
  `4872214e344ce266b0b10a1f70db33775224441ff1d78dbe0c7fb4dd70aeac84` /
  `05b80090d2d1ba751d48a5032cddec82819a79b9bfb5bd8b05306b85d6ef0122`.
- Behavioral seed 20260825 config/weights SHA-256:
  `4872214e344ce266b0b10a1f70db33775224441ff1d78dbe0c7fb4dd70aeac84` /
  `433978a1b942b4a6d8150e40ca067d2615f811ab8ad2ff880e9a161c655c5646`.
- R7 student samples/receipt SHA-256:
  `5283e8e1a66227d71d7a0c5847bd2c147397f580cfaa9c22520edfc65128e19b` /
  `782d9e58a97c5ac55dd6ebc2d62c67f9e003af415fb62e14ac8124718ea93b3a`.
- R8 student samples/receipt SHA-256:
  `178affbf232d8dd7ad6a021d02e4494756e477f121c6506aeae0868a8cc8069d` /
  `eb4cff3c9d5022887f2621bdf0c303b4aca807e9449e8c600860bd72a046b990`.
- GSM8K/teacher/QA SHA-256:
  `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77` /
  `dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e` /
  `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`.
- Ordered panel hashes: R7
  `78a3b7ef26cdf932b79eb6f64dfb576d66770d30a5ce0fd3251536cb6e76901f`;
  R8 `4c88bd4c27eb8fea9240e11503ef781e744313c5de7b1f0391bb680f7e3379bd`.

## Command

```powershell
python tools/research/run_trace_vs_behavioral_finalist.py --outdir runs/research/BACKLOG-ADAPT-TRACE-VS-FINALIST-01
```

## Factors

- Trace family: full-trace arms for seeds 20260830..20260836, imported from R7
  (panel 1 and QA) and R8 (panel 2).
- Behavioral family: ADAPT-TRAIN-01 seeds 20260824 and 20260825, freshly
  evaluated on both panels and all 48 QA tasks with greedy decoding,
  `max_new_tokens=192` math and 128 QA.
- Required trace coverage is 3,920 immutable rows: 7 x (512 math + 48 QA).
- Required fresh behavioral coverage is 1,120 rows: 2 x (512 math + 48 QA).
- Primary bootstrap: 20,000 replicates, seed `2026082711`; independently
  resample seven trace seeds and two behavioral seeds with replacement, then
  resample prompts within each fixed panel and average equal panel weights.
- Serving on 8080 may stop only for VRAM. Embedding on 8081 must remain healthy
  and the initial serving identity must be restored afterward.

## Acceptance gates

- `source_integrity`: `all_source_and_checkpoint_hashes_verified eq True`
- `panel_isolation`: `two_panels_disjoint_from_teacher_training_and_each_other eq True`
- `trace_import_coverage`: `imported_full_trace_generations eq 3920`
- `behavioral_checkpoint_coverage`: `behavioral_checkpoints_evaluated eq 2`
- `fresh_evaluation_coverage`: `fresh_behavioral_generations eq 1120`
- `practical_superiority`: `hierarchical_bootstrap_95ci_lower_trace_minus_behavioral_math gt 0.0`
- `panel_consistency`: `panels_with_positive_trace_minus_behavioral_math eq 2`
- `protected_retention`: `mean_trace_minus_behavioral_qa_accuracy ge -0.05`
- `independent_score`: `independent_rescore_match eq True`
- `service_recovery`: `service_and_embedding_restored eq True`

## Abort conditions

- Any frozen source, artifact, dataset, panel or preregistration hash differs.
- Either panel overlaps teacher tasks, training tasks or the other panel.
- Imported trace samples are incomplete, out of order or fail independent
  rescoring; fresh behavioral output is not exactly 560 rows per checkpoint.
- Embedding becomes unhealthy, VRAM remains insufficient after bounded service
  maintenance, or the initial gateway service cannot be restored.
- No receipt is emitted from partial output. This result cannot be described as
  causal evidence that traces outperform equally budgeted direct SFT.

## Allowed claims

- `TRACE_DISTILLATION_PRACTICALLY_SUPERIOR_TO_BEHAVIORAL_FINALIST_R1`
- `TRACE_DISTILLATION_NOT_PRACTICALLY_SUPERIOR_TO_BEHAVIORAL_FINALIST_R1`

Claims outside these codes are forbidden even if a metric looks favorable.
