# BACKLOG-FLEET-HUMANEVALPLUS-01 preregistration

Task: Requalify the coding alias on the full HumanEval+ benchmark across the text fleet
Evidence class: `serving_runtime`

## Hypothesis

On all 164 HumanEval+ tasks scored by the official EvalPlus executor, the
qualified `hauhaucs` coding route will achieve Plus pass@1 of at least 0.80 and
will be non-inferior to `qwen38`: the paired-bootstrap 95% lower bound of
HauhauCS minus Qwen3.8 must be greater than -0.05.

All four qualified text routes are evaluated on the identical complete panel.
The prior 200-task MBPP+ result is context only and is not pooled into this
benchmark's gate.

## Frozen inputs

- `config/qualified_model_fleet.json`
- `workloads/humaneval_plus.jsonl`
- `tools/analysis/a2_score_humaneval.py`
- `benchmark_harness_qa.py`
- `runs/research/BACKLOG-FLEET-MBPPPLUS-02/raw/receipt.json`

- Admission SHA-256: `5404ba3bc2e1232eddd9c36608036375df31c2c7f6b8f012864642598f80f99d`.
- Fleet registry SHA-256: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`.
- HumanEval+ JSONL SHA-256:
  `08d3df5c27a5f9a40176c27592b2d81e931b55d8d9edb7b1ffc28f2ccbdba735`.
- Official per-task scorer SHA-256:
  `cdbefde3f0e12b0dbf8697aff154ea88ec07d7fafb2a959a3c88882f63f1aa0d`.
- Shared extraction/benchmark library SHA-256:
  `60af3eac1e119047e3b0d767c52ee8295ac44abbfbaa44b1c42eee45945336c6`.
- MBPP+ R2 receipt SHA-256:
  `48fbe796e40c9f31e8b783a8fc92af40137be5847798d70e1f1ef6497c45c9fc`.
- Ordered full-panel task-ID SHA-256:
  `8c4a9413be6b6b793de94b702ab733ca734db2f5d6bca361605f1d7f71dd9ebe`.
- Each GGUF path, byte count and SHA-256 must match the frozen fleet registry.

## Command

```powershell
python tools/research/run_fleet_humanevalplus.py --outdir runs/research/BACKLOG-FLEET-HUMANEVALPLUS-01
```

## Factors

- Routes: `qwen38`, `hauhaucs`, `fable-tc`, `qwen36-moe`, explicitly switched
  and identity-checked through the port-8080 qualified-model gateway.
- Panel: all 164 rows in the frozen HumanEval+ JSONL; exactly 656 generations.
- Generation: greedy, temperature 0, top-k 1, seed 20260827,
  `max_tokens=768`, thinking disabled, cache off, complete-function prompt.
- Code extraction uses the frozen shared helper. Generated code executes only
  inside the existing WSL EvalPlus 0.3.1 environment.
- Primary comparison resamples the 164 paired Plus pass/fail differences with
  replacement for 20,000 replicates under seed `2026082714`.
- Four consecutive request failures abort a route. Initial gateway route and
  embedding service must be restored after completion or exception.

## Acceptance gates

- `artifact_identity`: `qualified_model_artifacts_verified eq 4`
- `route_coverage`: `verified_text_routes_completed eq 4`
- `generation_coverage`: `fresh_humaneval_generations eq 656`
- `response_integrity`: `successful_nonempty_responses ge 650`
- `official_score_coverage`: `models_scored_by_evalplus eq 4`
- `coding_alias_absolute`: `hauhaucs_humaneval_plus_pass_at_1 ge 0.8`
- `coding_alias_noninferiority`: `paired_bootstrap_95ci_lower_hauhaucs_minus_qwen38 gt -0.05`
- `service_recovery`: `initial_route_and_services_restored eq True`

## Abort conditions

- Any input, model artifact, full-panel hash or preregistration hash differs.
- Gateway identity differs from the requested alias or embedding is unhealthy
  at a route boundary.
- A route closes fewer than 164 ordered rows, four consecutive requests fail,
  or official EvalPlus produces fewer than 164 per-task verdicts.
- Initial gateway route cannot be restored. No receipt is emitted from partial
  output and generated code is never executed on the Windows host.

## Allowed claims

- `HAUHAUCS_FULL_HUMANEVALPLUS_CODING_ALIAS_RETAINED_R1`
- `HAUHAUCS_FULL_HUMANEVALPLUS_CODING_ALIAS_NOT_RETAINED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.
