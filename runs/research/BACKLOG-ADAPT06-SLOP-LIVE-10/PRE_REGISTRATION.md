# BACKLOG-ADAPT06-SLOP-LIVE-10 preregistration

Task: Complete the retained two-adapter forensic rebind with import-safe execution
Evidence class: `artifact_requalification`

## Hypothesis

The immutable R7 physical rows support the bounded client-ordering claim after
the R9 launcher-only import failure is corrected. No scorer, scientific factor,
acceptance threshold, source row or service state is changed.

## Frozen inputs

- Admission: `a42b31f138d66867a5958df7b5727a74cda8f3a128ba738d144240ad748e4600`.
- R9 blocked pipeline and wrapper:
  `6d9fec5661df7eaeb6b92a08ca3a06c71b18dbaaf92f10ea64eb654c674fe1f4`,
  `c972779c9fb3d00b92431ef932fe1b940679b023a9a47e298cfdc5c40c1aadeb`.
- R9 immutable launch failure stderr and worker exit:
  `69a5b43b1e682368f5dc4b935bff8f762d10e8b59782d5b96751744a986616dd`,
  `715343282c22a85c9fbd7e3aafdf28d26b14b87e6bf0ad2296c2bf7abab7c21d`.
- R7 terminal, review and live rows:
  `5437e2bcded18d2b5f32a0dbc83812cbcfb38bddd54cd2f10dd0d44a5e1b23cf`,
  `f39cd7aef72da06b7abcf5c1fdddd7af26799f24cbfc02db41ca7b8f8bb3f36a`,
  `43ef65d35e1ff79cea1e926f0486e442fb94e70bf4e80d10675cf3053a302830`.

## Command

```powershell
python tools/research/run_adapt06_forensic_rebind_r10.py --outdir runs/research/BACKLOG-ADAPT06-SLOP-LIVE-10
```

## Factors

- Seed the repository root in `sys.path` before importing the frozen R9/R8
  delegates; this is the sole execution repair.
- Recompute all 36 baseline and 72 routed UTF-8 rows, route controls,
  materiality, schedule parity and switch reduction.
- Use the harness terminal and retain 108 scored samples. No new inference,
  service mutation, cache or latency measurement is permitted.
- Bind the R7, R8, R9 and R10 identities in provenance.

## Acceptance gates

- `sealed_source`: `sealed_r7_source eq True`
- `baseline_hashes`: `recomputed_baseline_hashes eq 36`
- `routed_hashes`: `recomputed_routed_hashes eq 72`
- `route_counterfactual`: `route_correct_counterfactual_match_rate eq 1.0`
- `route_controls`: `physical_route_control_match_rate eq 1.0`
- `materiality`: `prompts_with_distinct_route_outputs ge 4`
- `schedule_parity`: `schedule_semantic_parity eq 1.0`
- `switch_reduction`: `requested_route_switch_reduction ge 0.8`
- `wrapper_binding`: `r8_and_r7_wrappers_provenance_bound eq True`
- `service_health`: `gateway_and_embedding_healthy eq True`

## Abort conditions

Abort without scientific interpretation if direct file invocation cannot import
the delegate, any frozen source hash drifts, terminal verification fails, the
108 source rows cannot be exactly reconstructed, provenance.script is not the
R10 wrapper, or either service health endpoint is unavailable.

## Allowed claims

- `ADAPT06_CLIENT_AFFINITY_FORENSIC_REBOUND_R10`
- `ADAPT06_CLIENT_AFFINITY_REJECTED_R10`

Claims outside these codes are forbidden. No speed, cache-isolation,
server-native scheduling, fused execution, production or generalization claim
is permitted.
