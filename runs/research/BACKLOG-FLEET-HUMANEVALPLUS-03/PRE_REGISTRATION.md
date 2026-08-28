# BACKLOG-FLEET-HUMANEVALPLUS-03 preregistration

Task: Rescore full HumanEval+ after bounded regeneration of objectively truncated rows
Evidence class: `serving_runtime`

## Hypothesis

The R2 comparison contains thirteen objectively censored responses whose source
runtime stopped at the common 768-token cap. Regenerating exactly those rows at
1536 tokens, without looking at their correctness and without touching any other
row, will leave at most two targeted rows truncated and will preserve the
`hauhaucs` coding alias gates: HumanEval+ pass@1 >= 0.80 and paired-bootstrap
lower 95% bound versus `qwen38` > -0.05.

## Frozen inputs

- `runs/research/BACKLOG-FLEET-HUMANEVALPLUS-02/raw/samples.jsonl`
- `runs/research/BACKLOG-FLEET-HUMANEVALPLUS-02/raw/official_scores.json`
- `runs/research/BACKLOG-FLEET-HUMANEVALPLUS-02/raw/receipt.json`
- `runs/research/BACKLOG-FLEET-HUMANEVALPLUS-02/raw/dataset_hashes.json`
- `runs/research/BACKLOG-FLEET-HUMANEVALPLUS-02/PRE_REGISTRATION.md`
- `tools/research/run_fleet_humanevalplus_r2.py`
- `config/qualified_model_fleet.json`
- `workloads/humaneval_plus.jsonl`
- `tools/analysis/a2_score_humaneval.py`
- `benchmark_harness_qa.py`

- `runs/research/BACKLOG-FLEET-HUMANEVALPLUS-02/raw/samples.jsonl`: `3f2d5d2df02e2443e05324436db32ba8b4b1f7e6c7c5ac02032fad5f58bd8da2`
- `runs/research/BACKLOG-FLEET-HUMANEVALPLUS-02/raw/official_scores.json`: `035107400cca9ae9393b73b82df05657aa22cc245b461c0d62bf431adbae3159`
- `runs/research/BACKLOG-FLEET-HUMANEVALPLUS-02/raw/receipt.json`: `b6cce633af34db44f92d76f345e19b3c3b0a5e8ccc0c9b756904f545be9f615a`
- `runs/research/BACKLOG-FLEET-HUMANEVALPLUS-02/raw/dataset_hashes.json`: `79540c786ea1777478fc88b46e879568c01c204d204ee2c88c36469bdc35edc6`
- `runs/research/BACKLOG-FLEET-HUMANEVALPLUS-02/PRE_REGISTRATION.md`: `54d727d209264b11853f07ed59137414f2db10673cf5ec6a154473171d05f862`
- `tools/research/run_fleet_humanevalplus_r2.py`: `139e5596bcd5cb8fda1cbcf0813aaee87959513232885a146dd83b1913203fe2`
- `config/qualified_model_fleet.json`: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`
- `workloads/humaneval_plus.jsonl`: `08d3df5c27a5f9a40176c27592b2d81e931b55d8d9edb7b1ffc28f2ccbdba735`
- `tools/analysis/a2_score_humaneval.py`: `cdbefde3f0e12b0dbf8697aff154ea88ec07d7fafb2a959a3c88882f63f1aa0d`
- `benchmark_harness_qa.py`: `60af3eac1e119047e3b0d767c52ee8295ac44abbfbaa44b1c42eee45945336c6`
- Admission spec: `5d49ba3f3831b7923cad55f073715ab29f827f710afa9b8599dc1381b13e188b`

## Command

```powershell
python tools/research/run_fleet_humanevalplus_r3.py --outdir runs/research/BACKLOG-FLEET-HUMANEVALPLUS-03
```

## Factors

- Frozen matrix: four aliases by all 164 HumanEval+ tasks (656 rows).
- Objective selection predicate: source R2 field `truncated == true` only.
- Frozen targets (13): `qwen38` 129/130/147; `hauhaucs`
  32/116/129/134/147; `qwen36-moe` 116/129/130/132/134; none for
  `fable-tc`.
- Treatment: regenerate only the thirteen targets with the R2 decoding settings,
  seed and prompt, changing only `max_tokens` from 768 to 1536.
- Controls: all 643 non-target rows are imported byte-for-byte at the record
  level; all four complete matrices are rescored by the official EvalPlus
  harness.
- Statistic: paired task bootstrap, 20,000 deterministic replicates, Hauhaucs
  minus Qwen38. Hardware/runtime identity is recorded live on the RTX 3090.
- This is the final cap-correction round; no R4 or post-result cap extension is
  allowed.

## Acceptance gates

- `source_integrity`: `r2_sources_and_model_artifacts_verified eq True`
- `objective_selection`: `source_rows_selected_only_by_truncation eq 13`
- `fresh_correction_coverage`: `fresh_1536_token_regenerations eq 13`
- `official_score_coverage`: `models_rescored_by_evalplus eq 4`
- `truncation_recovery`: `remaining_truncated_target_rows le 2`
- `coding_alias_absolute`: `corrected_hauhaucs_humaneval_plus_pass_at_1 ge 0.8`
- `coding_alias_noninferiority`: `corrected_paired_bootstrap_95ci_lower_hauhaucs_minus_qwen38 gt -0.05`
- `service_recovery`: `initial_route_and_services_restored eq True`

## Abort conditions

- Abort fail-closed on any frozen hash mismatch, model-artifact mismatch, target
  set/count mismatch, duplicate/missing task, non-target mutation, route identity
  mismatch, official scorer failure, missing receipt, or failure to restore the
  initial 8080 route and healthy 8081 embedding service.
- Do not substitute proxy or synthetic generations and do not select rows using
  R2 correctness.

## Allowed claims

- `HAUHAUCS_HUMANEVALPLUS_CAP_CORRECTED_ALIAS_RETAINED_R3`
- `HAUHAUCS_HUMANEVALPLUS_CAP_CORRECTED_ALIAS_NOT_RETAINED_R3`

Claims outside these codes are forbidden even if a metric looks favorable.
