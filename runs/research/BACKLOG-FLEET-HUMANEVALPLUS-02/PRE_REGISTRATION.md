# BACKLOG-FLEET-HUMANEVALPLUS-02 preregistration

Task: Continue full HumanEval+ fleet evaluation after scorer import abort
Evidence class: `serving_runtime`

## Hypothesis

Unchanged from R1: on all 164 HumanEval+ tasks, `hauhaucs` will achieve Plus
pass@1 of at least 0.80 and a paired-bootstrap 95% lower bound versus `qwen38`
greater than -0.05. R1 produced 164 complete Qwen3.8 responses but aborted
before scoring because its direct-file scorer lacked repository `PYTHONPATH`.

Those Qwen3.8 generations are immutable and imported by hash. R2 changes only
the scorer bootstrap and generates the three absent routes; the scientific
panel, prompts, decoding contract, gates and bootstrap remain unchanged.

## Frozen inputs

- `runs/research/BACKLOG-FLEET-HUMANEVALPLUS-01/raw/samples.jsonl`
- `runs/research/BACKLOG-FLEET-HUMANEVALPLUS-01/raw/qwen38.samples.jsonl`
- `runs/research/BACKLOG-FLEET-HUMANEVALPLUS-01/raw/recovery_state.json`
- `runs/research/BACKLOG-FLEET-HUMANEVALPLUS-01/PRE_REGISTRATION.md`
- `runs/research/BACKLOG-FLEET-HUMANEVALPLUS-01/runner.stderr.log`
- `runs/autonomous/EXPERIMENT-WATCH-2026-08-27-FLEET-HUMANEVAL-R1/FINAL.json`
- `tools/research/run_fleet_humanevalplus.py`
- `config/qualified_model_fleet.json`
- `workloads/humaneval_plus.jsonl`
- `tools/analysis/a2_score_humaneval.py`
- `benchmark_harness_qa.py`

- Admission SHA-256: `dc6440ecc190b69e35af792b15cf87c37699a2b564f26b12d76e827158ca1c61`.
- R1 full response/sample JSONL SHA-256:
  `a243f7e842ad837521084ba2e881c7708491028c0d3bac778c0536cbfc402b1f` /
  `a325dcfa96bd92e018c8f7fbd76a836e2d3e0e46185ccc026fb5cfc9896b55d3`.
- R1 recovery-state SHA-256:
  `7f9104d994b58c54a653fa17aabd2ae2a93b0543de89d660b02d4c05f8c98be6`.
- R1 preregistration/implementation SHA-256:
  `bb93658ec154eaaa136074ec418ddfbc6ce657957670e45c4c497b0d90b1b0fd` /
  `e495a2094097d1e616709f6064850e8b3d2fd0778b81c9f4ef05944934cb6cfe`.
- R1 stderr/watcher-final SHA-256:
  `648b63739f42c7f42f6858c48fa8003d31071c5b5582d386d207b29ee925e0f9` /
  `7d6437de24949e8fd40c358ad6837f284cd7d74f2aed35587926efdb91c6471a`.
- Fleet/HumanEval+/scorer/shared-helper SHA-256:
  `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82` /
  `08d3df5c27a5f9a40176c27592b2d81e931b55d8d9edb7b1ffc28f2ccbdba735` /
  `cdbefde3f0e12b0dbf8697aff154ea88ec07d7fafb2a959a3c88882f63f1aa0d` /
  `60af3eac1e119047e3b0d767c52ee8295ac44abbfbaa44b1c42eee45945336c6`.
- Ordered full-panel hash:
  `8c4a9413be6b6b793de94b702ab733ca734db2f5d6bca361605f1d7f71dd9ebe`.
- All four GGUF identities remain bound to the frozen fleet registry.

## Command

```powershell
python tools/research/run_fleet_humanevalplus_r2.py --outdir runs/research/BACKLOG-FLEET-HUMANEVALPLUS-02
```

## Factors

- Import exactly 164 ordered Qwen3.8 records and solution rows from R1; verify
  nonempty count, truncation flags, task IDs and both file hashes.
- Generate exactly 164 fresh rows for each of `hauhaucs`, `fable-tc` and
  `qwen36-moe` under the unchanged greedy 768-token contract (492 fresh rows).
- Score all four solution files with EvalPlus 0.3.1 while setting `PYTHONPATH`
  to the WSL repository root; code remains inside the WSL executor.
- Primary comparison remains 20,000 paired bootstrap replicates under seed
  `2026082714` over all 164 task verdicts.
- Route snapshots include imported Qwen3.8 identity plus the three live route
  switches. Initial gateway route and embedding service must be restored.

## Acceptance gates

- `source_integrity`: `r1_partial_sources_and_model_artifacts_verified eq True`
- `imported_coverage`: `imported_qwen38_generations eq 164`
- `fresh_coverage`: `fresh_remaining_generations eq 492`
- `route_coverage`: `verified_text_routes_completed eq 4`
- `response_integrity`: `successful_nonempty_responses ge 650`
- `official_score_coverage`: `models_scored_by_evalplus eq 4`
- `coding_alias_absolute`: `hauhaucs_humaneval_plus_pass_at_1 ge 0.8`
- `coding_alias_noninferiority`: `paired_bootstrap_95ci_lower_hauhaucs_minus_qwen38 gt -0.05`
- `service_recovery`: `initial_route_and_services_restored eq True`

## Abort conditions

- Any frozen R1 source, input, model artifact, panel or preregistration hash
  differs; imported rows are incomplete or out of order.
- Qwen3.8 is regenerated, any missing route closes fewer than 164 rows, four
  consecutive requests fail, or an official score lacks 164 verdicts.
- Embedding becomes unhealthy or the initial route cannot be restored.
- No receipt is emitted from partial output; the R1 import failure is strictly
  operational evidence and cannot affect model scores.

## Allowed claims

- `HAUHAUCS_FULL_HUMANEVALPLUS_CODING_ALIAS_RETAINED_R2`
- `HAUHAUCS_FULL_HUMANEVALPLUS_CODING_ALIAS_NOT_RETAINED_R2`

Claims outside these codes are forbidden even if a metric looks favorable.
