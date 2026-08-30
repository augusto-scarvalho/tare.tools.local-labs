# BACKLOG-ADAPT06-SLOP-LIVE-09 preregistration

Task: Complete the retained two-adapter forensic rebind with verified terminal manifest access
Evidence class: `artifact_requalification`

## Hypothesis

The immutable R7 physical rows support the same bounded client-ordering claim
as R8 after reading the sealed file count from `run.terminal.json` rather than
from the compact `verify_run` projection. No scientific factor changes.

## Frozen inputs

- `runs/research/BACKLOG-ADAPT06-SLOP-LIVE-08/PIPELINE.json`
- `tools/research/run_adapt06_forensic_rebind_r8.py`
- `runs/research/BACKLOG-ADAPT06-SLOP-LIVE-07/raw/run.terminal.json`
- `runs/research/BACKLOG-ADAPT06-SLOP-LIVE-07/REVIEW.json`
- `runs/research/BACKLOG-ADAPT06-SLOP-LIVE-07/raw/physical_r5/raw/live_rows.json`

- Admission: `77bc0a61a7d94f815162acbbf65704ba928f561d06b68f4ab65a415dd129b9b8`.
- R8 blocked pipeline and delegated scorer:
  `b9e9a2758ee6e65820de8a9b26ed88f9e0f446a4303c74c71bb7998ea500e903`,
  `c4efef978abfa35b0cbc384506ece6be5ed3c2cd2612f94925b17d5891c18607`.
- R7 terminal, review and live rows retain their frozen identities:
  `5437e2bcded18d2b5f32a0dbc83812cbcfb38bddd54cd2f10dd0d44a5e1b23cf`,
  `f39cd7aef72da06b7abcf5c1fdddd7af26799f24cbfc02db41ca7b8f8bb3f36a`,
  `43ef65d35e1ff79cea1e926f0486e442fb94e70bf4e80d10675cf3053a302830`.

## Command

```powershell
python tools/research/run_adapt06_forensic_rebind_r9.py --outdir runs/research/BACKLOG-ADAPT06-SLOP-LIVE-09
```

## Factors

- Reuse the complete frozen R8 scorer without modifying it.
- Override only task identity, frozen-input map, executing-wrapper identity and
  the `verify_run` projection so `manifest` is populated from the already
  hashed R7 `run.terminal.json`.
- Recompute 36 baseline and 72 routed UTF-8 rows, route controls, materiality,
  schedule parity and switch reduction. No inference or service mutation.
- Bind R7, R8 and R9 wrappers in the new provenance receipt.

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

All R8 abort conditions remain. Additionally abort if the terminal JSON lacks
a manifest, the projected manifest differs from the terminal bytes, delegated
script identity drifts or provenance.script is not the R9 wrapper.

## Allowed claims

- `ADAPT06_CLIENT_AFFINITY_FORENSIC_REBOUND_R9`
- `ADAPT06_CLIENT_AFFINITY_REJECTED_R9`

Claims outside these codes are forbidden even if a metric looks favorable.
No speed, cache, server-native scheduler, fused execution, production or
generalization claim is permitted.
