# BACKLOG-ADAPT-TRACE-DISTILL-08 preregistration

Task: Replicate the seven-seed trace-distillation gain on a second teacher-disjoint math panel
Evidence class: `distillation`

## Hypothesis

For the fourteen immutable R7 checkpoints (seven paired seeds), full-trace SFT
will retain a positive math-accuracy effect over matched answer-only SFT on a
second, disjoint 256-task GSM8K panel. The hierarchical bootstrap 95% lower
bound over paired seed-by-prompt differences must exceed zero and at least five
of seven seed deltas must be positive.

This is an out-of-panel replication only. No checkpoint is retrained or
selected after seeing the second panel. The original 48-task protected-QA
result is imported from R7 by hash because retention is a property of the same
immutable checkpoints and was already measured completely. This family stops
after R8 regardless of outcome.

## Frozen inputs

- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-07/raw/checkpoints`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-07/raw/checkpoint_hashes.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-07/raw/actual_scores.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-07/raw/student_samples.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-07/raw/training_pairs.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-07/raw/receipt.json`
- `workloads/gsm8k.jsonl`
- `runs/a2/market-r0__thinkingcap-27b-q4__gsm8k.json`

- Admission SHA-256: `d0e8e967ca3885ce39349201edd1f162795058af2e19e630b2780bc36c607059`.
- R7 checkpoint ledger SHA-256: `57364aaba37c39771aaf216950bfff6df1282735b641b0c682ded89ffa8aaf4c`.
- R7 scores SHA-256: `0171dcfcd70334a780a16337469f200656ec3b1d7c567889393c419cad9bae1e`.
- R7 student samples SHA-256: `5283e8e1a66227d71d7a0c5847bd2c147397f580cfaa9c22520edfc65128e19b`.
- R7 training pairs SHA-256: `5c3f0d5fd80d97351839bca1e38685e5e21b3357dfa56077f44f02b857bfe4cc`.
- R7 execution receipt SHA-256: `782d9e58a97c5ac55dd6ebc2d62c67f9e003af415fb62e14ac8124718ea93b3a`.
- GSM8K SHA-256: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`.
- Teacher rows SHA-256: `dc5cabe44c92e48b0e832881ef27ebad4047b140928c9a12678e0c0c6660006e`.
- Ordered R7 panel hash: `78a3b7ef26cdf932b79eb6f64dfb576d66770d30a5ce0fd3251536cb6e76901f`.
- Ordered second-panel hash: `4c88bd4c27eb8fea9240e11503ef781e744313c5de7b1f0391bb680f7e3379bd`.
- Each checkpoint's `adapter_model.safetensors` and `adapter_config.json` must
  match the fourteen entries in the frozen R7 checkpoint ledger.

## Command

```powershell
python tools/research/run_trace_distillation_replication_r8.py --outdir runs/research/BACKLOG-ADAPT-TRACE-DISTILL-08
```

## Factors

- Treatment is fixed by R7: answer-only versus full-trace SFT, matched within
  each of seeds `20260830` through `20260836`.
- The second panel is the next 256 lowest numeric GSM8K IDs after the R7 panel
  among IDs absent from every teacher row. It is disjoint from the 168 training
  IDs, all teacher rows, and all R7 held-out IDs.
- All 14 checkpoints are loaded independently on the RTX 3090 and decoded
  greedily with the unchanged prompt template and `max_new_tokens=192`.
- Exactly 3,584 fresh generations are required: 7 seeds x 2 arms x 256 tasks.
- Primary estimator is the mean paired accuracy delta with a hierarchical
  bootstrap over seeds and prompts, 20,000 replicates, seed `2026082708`.
- Directional repeatability requires at least five positive seed deltas.
- R7 protected-QA is imported and independently recomputed from its raw sample
  text; its mean regression must remain at most 0.05.
- Serving on 8080 may stop only to release VRAM; embedding on 8081 must remain
  healthy and the initial serving identity must be restored.

## Acceptance gates

- `source_integrity`: `r7_source_and_checkpoint_hashes_verified eq True`
- `panel_isolation`: `second_panel_disjoint_from_teacher_training_and_r7_panel eq True`
- `checkpoint_coverage`: `immutable_checkpoints_evaluated eq 14`
- `evaluation_coverage`: `fresh_second_panel_generations eq 3584`
- `replicated_gain`: `hierarchical_bootstrap_95ci_lower_trace_math_gain gt 0.0`
- `directional_repeatability`: `seeds_with_positive_trace_math_gain ge 5`
- `protected_retention`: `imported_r7_mean_protected_qa_regression le 0.05`
- `independent_score`: `independent_rescore_match eq True`
- `service_recovery`: `service_and_embedding_restored eq True`

## Abort conditions

- Any source, checkpoint, dataset, panel or preregistration hash differs.
- The second panel overlaps R7, the training pool or any teacher task.
- Any checkpoint is missing, changed, or produces fewer than 256 rows.
- Independent rescoring disagrees with stored correctness for any fresh row or
  imported protected-QA row.
- Embedding becomes unhealthy, the GPU cannot fit after bounded service
  maintenance, or the qualified serving baseline cannot be restored.
- No receipt is emitted from partial output; no R9 panel may be opened to rescue
  a negative result.

## Allowed claims

- `TRACE_DISTILLATION_SECOND_PANEL_REPLICATED_R8`
- `TRACE_DISTILLATION_SECOND_PANEL_NOT_REPLICATED_R8`

Claims outside these codes are forbidden even if a metric looks favorable.
