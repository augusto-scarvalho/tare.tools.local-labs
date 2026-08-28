# BACKLOG-ADAPT-TRACE-VS-FINALIST-02 preregistration

Task: Compare the selected trace deployment candidate with behavioral finalists on the third panel
Evidence class: `distillation`

## Hypothesis

The preselected trace seed `20260832` will outperform the mean of both
reproduced behavioral-finalist checkpoints on the untouched third panel, with
a hierarchical-bootstrap lower 95% bound above zero and protected-QA delta no
worse than -5 percentage points.

## Frozen inputs

- `runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/receipt.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/student_samples.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/dataset_hashes.json`
- `runs/research/BACKLOG-ADAPT-TRACE-VS-FINALIST-01/raw/receipt.json`
- `runs/research/BACKLOG-ADAPT-TRACE-VS-FINALIST-01/raw/student_samples.json`
- `runs/research/BACKLOG-ADAPT-TRAIN-01/raw/receipt.json`
- `tools/research/run_trace_vs_behavioral_finalist.py`
- `tools/research/run_trace_distillation_replication_r8.py`
- `workloads/gsm8k.jsonl`

- Admission: `360c5e626bb363860d0dd675611c90519c10345b864d5051b52225f29122c8d1`.
- Selected-trace receipt/samples/dataset: `b4fc924a1542e4913c3c1d70fdf77f8bb9be0e2662b8757d0d06f82b60d3f521`, `288270e4faa780bbd905b593193bf9c9edc595d84bf41cc2ef3fd72ba53663c9`, `f3bd82ee0aef9b7eb7669d7ed6bc5549412c8a5ebdec33bb4496bb869c95661c`.
- Prior practical-comparison receipt/samples: `54ada49af2a437513f47b766dc2f6fd9a71b93e88d06e48ae896b6cca88a1487`, `b94d98cd6f5f356e7011ca9ca11186d15cb40923059ef3ceaf4ba93492d3abd5`.
- Behavioral training receipt: `903c723f3d63130cf06a5e501498451beee0cee34a8aa71d6f9de36faeb602b8`.
- Prior practical runner: `a1cdb8766699108effcd17e53b681667bbe7757ca0e3371f712c9d3f8d7b6ff6`.
- R8 worker runner: `0ad1f687c8ed1b9f0d923a61fa853a47ece9b35dbaaaa0b72b483e9793cbcbec`.
- Behavioral seed 20260824 config/weights: `4872214e344ce266b0b10a1f70db33775224441ff1d78dbe0c7fb4dd70aeac84`, `05b80090d2d1ba751d48a5032cddec82819a79b9bfb5bd8b05306b85d6ef0122`.
- Behavioral seed 20260825 config/weights: `4872214e344ce266b0b10a1f70db33775224441ff1d78dbe0c7fb4dd70aeac84`, `433978a1b942b4a6d8150e40ca067d2615f811ab8ad2ff880e9a161c655c5646`.
- GSM8K: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`.

## Command

```powershell
python tools/research/run_trace_vs_behavioral_finalist_r2.py --outdir runs/research/BACKLOG-ADAPT-TRACE-VS-FINALIST-02
```

## Factors

- Import exactly the 256 immutable third-panel full-trace samples for selected
  seed `20260832`; do not regenerate or reselect trace evidence.
- Freshly evaluate behavioral seeds `20260824` and `20260825` on the identical
  third-panel ID list, producing 512 new deterministic generations.
- Same frozen Qwen3.5-0.8B base, PEFT loader, prompt, 192-token cap, greedy
  decoding and independent GSM8K scorer as the trace worker.
- Bootstrap samples the two behavioral seeds and 256 prompts for 20,000
  deterministic replicates while holding the single selected trace artifact
  fixed.
- QA imports the already hash-bound trace seed score `11/48` and behavioral
  scores `12/48` and `10/48`; mean QA delta is therefore frozen at zero.

## Acceptance gates

- `source_integrity`: `all_source_and_checkpoint_hashes_verified eq True`
- `panel_isolation`: `third_panel_disjoint_from_training_and_prior_panels eq True`
- `trace_import_coverage`: `imported_selected_trace_generations eq 256`
- `behavioral_checkpoint_coverage`: `behavioral_checkpoints_evaluated eq 2`
- `fresh_evaluation_coverage`: `fresh_behavioral_generations eq 512`
- `practical_superiority`: `hierarchical_bootstrap_95ci_lower_trace_minus_behavioral_math gt 0.0`
- `point_superiority`: `selected_trace_minus_behavioral_mean_math gt 0.0`
- `protected_retention`: `selected_trace_minus_behavioral_mean_qa ge -0.05`
- `independent_score`: `independent_rescore_match eq True`
- `service_recovery`: `service_and_embedding_restored eq True`

## Abort conditions

- Abort on source/checkpoint/base mismatch, third-panel mismatch, incomplete
  worker, scorer disagreement, insufficient VRAM, embedding failure or service
  restoration failure. Wrong answers or non-superiority are evidence.
- No fourth panel, new seed or post-result selection is allowed.

## Allowed claims

- `SELECTED_TRACE_PRACTICALLY_SUPERIOR_TO_BEHAVIORAL_FINALISTS_R2`
- `SELECTED_TRACE_NOT_PRACTICALLY_SUPERIOR_TO_BEHAVIORAL_FINALISTS_R2`

Claims outside these codes are forbidden even if a metric looks favorable.
