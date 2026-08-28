# BACKLOG-FLEET-MBPPPLUS-01 preregistration

Task: Requalify the coding alias on a 100-task MBPP+ screen across the text fleet
Evidence class: `serving_runtime`

## Hypothesis

On a fixed 100-task MBPP+ subset scored by the official EvalPlus executor, the
qualified `hauhaucs` coding route will achieve Plus pass@1 of at least 0.70 and
will be non-inferior to the `qwen38` general route: the paired-bootstrap 95%
lower bound of HauhauCS minus Qwen3.8 must be greater than -0.05.

All four qualified text routes are evaluated descriptively on the same tasks.
Only the existing coding-alias claim is gated; this experiment does not select
a new broad default or make latency claims.

## Frozen inputs

- `config/qualified_model_fleet.json`
- `workloads/mbpp_plus.jsonl`
- `workloads/mbpp_plus.identity.json`
- `tools/analysis/score_mbpp_subset.py`
- `tools/benchmarks/mbpp_plus_bench.py`
- `benchmark_harness_qa.py`
- `runs/research/BACKLOG-FLEET-REGRESSION-SCREEN-01/raw/receipt.json`

- Admission SHA-256: `b164711c6cb1851d0cd669a4007b5e9c208d0623227bf7cd5a1ead8203546f78`.
- Fleet registry SHA-256: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`.
- MBPP+ JSONL/identity SHA-256:
  `0b3f47548d2e7676fac6b9bcbbc074b3a8df49606c6949343c716e46f1df9264` /
  `d35e1146d622b27c3f7e0a8d776834404ff1dc8cd2137f7ad30fae5a5c5f62d0`.
- Official subset scorer SHA-256:
  `c0c80841b92756cda807a1d13c28dbcfe82df014ce13c32fb257fa823e7d9f6e`.
- Generation harness/reference SHA-256:
  `0c50677f26fe92e9dfc14b4db2dacd3c8b17e5824b1a11a688bc731ced34cacf`.
- Shared benchmark QA library SHA-256:
  `60af3eac1e119047e3b0d767c52ee8295ac44abbfbaa44b1c42eee45945336c6`.
- Prior fleet receipt SHA-256:
  `d303490b152babcc8f590b0b840fb14c4da02c865430f8398bdca7023a0eeb94`.
- Ordered 100-task subset SHA-256:
  `288ceec36e29dea4f39822770e8a61bd93df8f57e8fc5b8cdf9f0fc58a238e91`.
- Each GGUF path, byte count and SHA-256 must match its entry in the frozen
  fleet registry before its route can generate evidence.

## Command

```powershell
python tools/research/run_fleet_mbppplus.py --outdir runs/research/BACKLOG-FLEET-MBPPPLUS-01
```

## Factors

- Routes: `qwen38`, `hauhaucs`, `fable-tc`, `qwen36-moe`, each explicitly
  switched and verified through the qualified-model gateway on port 8080.
- Panel: first 100 indices of the single deterministic shuffle of all 378
  MBPP+ rows under seed `20260726`; identical ordered task IDs for every route.
- Generation: one greedy response per task, temperature 0, top-k 1,
  `max_tokens=768`, thinking disabled, no prompt cache. Exactly 400 requests.
- Code is extracted with the frozen shared helper. Each model's 100 solutions
  are executed by EvalPlus 0.3.1 in the existing WSL evalplus environment.
- Primary comparison uses paired Plus pass/fail differences on the 100 common
  tasks, 20,000 bootstrap replicates with seed `2026082712`.
- HTTP failures remain failed rows. Four consecutive transport/server failures
  abort the active route. Initial gateway route and embedding service must be
  restored after the final model or any exception.

## Acceptance gates

- `artifact_identity`: `qualified_model_artifacts_verified eq 4`
- `route_coverage`: `verified_text_routes_completed eq 4`
- `generation_coverage`: `fresh_mbpp_generations eq 400`
- `response_integrity`: `successful_nonempty_responses ge 396`
- `official_score_coverage`: `models_scored_by_evalplus eq 4`
- `coding_alias_absolute`: `hauhaucs_mbpp_plus_pass_at_1 ge 0.7`
- `coding_alias_noninferiority`: `paired_bootstrap_95ci_lower_hauhaucs_minus_qwen38 gt -0.05`
- `service_recovery`: `initial_route_and_services_restored eq True`

## Abort conditions

- Any host input, GGUF size/hash, subset hash or preregistration hash differs.
- Gateway reports a model other than the requested alias after switching.
- Embedding on 8081 becomes unhealthy at a route boundary.
- Four consecutive request failures occur, fewer than 100 rows close for a
  model, or official EvalPlus produces no complete 100-task score.
- Initial gateway route cannot be restored. No receipt is emitted from partial
  output and raw model code is never executed outside the existing EvalPlus
  WSL executor.

## Allowed claims

- `HAUHAUCS_MBPPPLUS_CODING_ALIAS_RETAINED_R1`
- `HAUHAUCS_MBPPPLUS_CODING_ALIAS_NOT_RETAINED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.
