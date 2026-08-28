# BACKLOG-FLEET-MBPPPLUS-02 preregistration

Task: Resolve coding-alias uncertainty with a final disjoint MBPP+ panel
Evidence class: `serving_runtime`

## Hypothesis

Across exactly two equal, disjoint 100-task MBPP+ panels, the `hauhaucs` coding
route will retain at least 0.70 combined Plus pass@1 and remain non-inferior to
`qwen38`: the equal-panel stratified-bootstrap 95% lower bound of HauhauCS minus
Qwen3.8 must be greater than -0.05. This is the final MBPP panel extension
regardless of outcome.

R1 generations and official scores are imported by hash. All four routes
generate only the second fixed panel, which was frozen before those outputs.

## Frozen inputs

- `runs/research/BACKLOG-FLEET-MBPPPLUS-01/raw/samples.jsonl`
- `runs/research/BACKLOG-FLEET-MBPPPLUS-01/raw/official_scores.json`
- `runs/research/BACKLOG-FLEET-MBPPPLUS-01/raw/receipt.json`
- `config/qualified_model_fleet.json`
- `workloads/mbpp_plus.jsonl`
- `workloads/mbpp_plus.identity.json`
- `tools/analysis/score_mbpp_subset.py`
- `tools/research/run_fleet_mbppplus.py`
- `benchmark_harness_qa.py`

- Admission SHA-256: `84c0fbb21eba0cc380ece1be59bc921ea333900dbba8793d4febb81870f60b63`.
- R1 samples/scores/receipt SHA-256:
  `357a3b29558867559d87c2083bcb08ac4bf0ba26e0f5f054efc9513840e5e46e` /
  `5c44c3741a380b0869658630ca886d5e364351965a59c1b894f5cb8ca2951280` /
  `cbbd3ea566581b152c71d992f8cf29bf2131eabb8d7ccd09de551ddc51205b5e`.
- R1 preregistration/implementation SHA-256:
  `3be7bdf941f7e9f903b7b49a78f60b5c09db1306f3dd6f140f210ea443e2cab8` /
  `bab01e2ca100cba1f280947322b0c22b137cb35a2181b490a0c1e44387ef3a39`.
- Fleet registry SHA-256: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`.
- MBPP+ JSONL/identity SHA-256:
  `0b3f47548d2e7676fac6b9bcbbc074b3a8df49606c6949343c716e46f1df9264` /
  `d35e1146d622b27c3f7e0a8d776834404ff1dc8cd2137f7ad30fae5a5c5f62d0`.
- Scorer/shared extractor SHA-256:
  `c0c80841b92756cda807a1d13c28dbcfe82df014ce13c32fb257fa823e7d9f6e` /
  `60af3eac1e119047e3b0d767c52ee8295ac44abbfbaa44b1c42eee45945336c6`.
- Ordered R1/R2 panel hashes:
  `288ceec36e29dea4f39822770e8a61bd93df8f57e8fc5b8cdf9f0fc58a238e91` /
  `27374bbc1040b08ee5c4b4ecd518ed03e00d826a3a5cf9a5b4c86e45ea380ef0`.
- All four model artifact identities remain bound to the frozen fleet registry.

## Command

```powershell
python tools/research/run_fleet_mbppplus_r2.py --outdir runs/research/BACKLOG-FLEET-MBPPPLUS-02
```

## Factors

- R2 panel: indices 101..200 of the same seed-20260726 deterministic shuffle,
  sorted back to dataset order; it is disjoint from R1's first 100 indices.
- Routes and generation contract are unchanged from R1: four text aliases,
  greedy/top-k 1, thinking off, 768-token cap, one response per task.
- EvalPlus 0.3.1 officially executes each route's second-panel solutions.
- Combined score is total Plus passes across the 200 unique tasks divided by
  200, with descriptive per-panel and per-model values retained.
- Primary uncertainty: 20,000 replicates, seed `2026082713`; independently
  resample 100 paired outcomes inside each panel and average equal panel deltas.
- Initial route and services must be restored. Four consecutive request
  failures abort the active route.

## Acceptance gates

- `source_integrity`: `r1_sources_and_model_artifacts_verified eq True`
- `panel_isolation`: `two_mbpp_panels_disjoint eq True`
- `route_coverage`: `verified_text_routes_completed eq 4`
- `fresh_generation_coverage`: `fresh_second_panel_generations eq 400`
- `combined_score_coverage`: `official_model_panel_scores eq 8`
- `coding_alias_absolute`: `combined_hauhaucs_mbpp_plus_pass_at_1 ge 0.7`
- `coding_alias_noninferiority`: `stratified_bootstrap_95ci_lower_hauhaucs_minus_qwen38 gt -0.05`
- `service_recovery`: `initial_route_and_services_restored eq True`

## Abort conditions

- Any R1 source, model artifact, dataset, scorer, panel or preregistration hash
  differs, or R1/R2 task IDs overlap.
- Any route returns fewer than 100 ordered rows or lacks a complete official
  100-task EvalPlus score.
- Embedding becomes unhealthy at a route boundary, four consecutive requests
  fail, or the initial gateway route cannot be restored.
- No receipt is emitted from partial output. No R3 MBPP panel may be admitted to
  rescue this result.

## Allowed claims

- `HAUHAUCS_MBPPPLUS_200_CODING_ALIAS_RETAINED_R2`
- `HAUHAUCS_MBPPPLUS_200_CODING_ALIAS_NOT_RETAINED_R2`

Claims outside these codes are forbidden even if a metric looks favorable.
