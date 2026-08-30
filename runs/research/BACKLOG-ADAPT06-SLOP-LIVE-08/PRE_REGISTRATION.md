# BACKLOG-ADAPT06-SLOP-LIVE-08 preregistration

Task: Forensically rebind the retained two-adapter gateway affinity evidence
Evidence class: `artifact_requalification`

## Hypothesis

The immutable R7 physical rows support a bounded two-adapter client-ordering
claim after every baseline/routed content digest is recomputed from its exact
UTF-8 text and both the R7 execution wrapper and R8 forensic wrapper are bound
in provenance. Any route, control, schedule or binding failure rejects it.

## Frozen inputs

- `runs/research/BACKLOG-ADAPT06-SLOP-LIVE-07/raw/run.terminal.json`
- `runs/research/BACKLOG-ADAPT06-SLOP-LIVE-07/REVIEW.json`
- `runs/research/BACKLOG-ADAPT06-SLOP-LIVE-07/raw/physical_r5/raw/live_rows.json`
- `runs/autonomous/POST-AUDIT-VALUE-WAVE-2026-08-29/watchers/002-BACKLOG-ADAPT06-SLOP-LIVE-07/FINAL.json`
- `tools/research/run_adapt06_slop_live_r7.py`

- Admission: `988e7aeb2919536827f5ffbdcc89d2409af6685678c7d893e8a8b4cb38ffeaa1`.
- R7 terminal and independent HOLD review:
  `5437e2bcded18d2b5f32a0dbc83812cbcfb38bddd54cd2f10dd0d44a5e1b23cf`,
  `f39cd7aef72da06b7abcf5c1fdddd7af26799f24cbfc02db41ca7b8f8bb3f36a`.
- Immutable physical live rows:
  `43ef65d35e1ff79cea1e926f0486e442fb94e70bf4e80d10675cf3053a302830`.
- R7 watcher FINAL, LAUNCH and WORKER_EXIT:
  `156f8a7628f4bea9fd320a77415217fd004d7c84b0a12354399fa5f54dab3487`,
  `28af4decb884a652445a4ef9d7109288b8b3e55ea033ad3f094e1b5833f3fdd3`,
  `328c7d9aca047000b77272b9fcaae5144542beafd9cb44c9a42dc96673ea9655`.
- R7 PIPELINE and wrapper:
  `70239662df0a37dae7ae428aefeda4f9e084a1a5896d4150d06e07470f979132`,
  `a443b1bb7df3f42dafafd45b67c01605a5515297264a587a01463fa20d9b27c8`.

## Command

```powershell
python tools/research/run_adapt06_forensic_rebind_r8.py --outdir runs/research/BACKLOG-ADAPT06-SLOP-LIVE-08
```

## Factors

- No new inference or service mutation. Reopen exactly 36 isolated baselines,
  72 routed rows, three cache records and the two 30-row schedules from R7.
- Hash exact Python Unicode text as UTF-8 bytes for every baseline and routed
  row; retain all 108 corrected digest rows rather than reusing R7 scalars.
- For each routed row, compare full text against the isolated baseline for the
  same `(route,index)` and verify request and response LoRA vectors against the
  frozen base/MLP/attention controls.
- Materiality counts prompts with at least two distinct isolated route texts.
  Schedule parity compares `(route,index)` cells; switch reduction is
  `1 - grouped_switches / alternating_switches`.
- Cache rows remain descriptive and cannot support cache-isolation claims.
  Wall times remain descriptive and cannot support a speedup claim.
- Verify current 8080/8081 health without changing the active service.

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

Abort on any source hash mismatch, invalid R7 terminal/watcher identity,
missing/duplicate route-index cell, row-count drift, route-control ambiguity,
non-UTF-8 input, nonfinite schedule fact, unhealthy 8080/8081, incomplete
harness seal or incomplete provenance. Never repair or rewrite R7 evidence.

## Allowed claims

- `ADAPT06_CLIENT_AFFINITY_FORENSIC_REBOUND_R8`
- `ADAPT06_CLIENT_AFFINITY_REJECTED_R8`

Claims outside these codes are forbidden even if a metric looks favorable.
No server-native scheduling, cache isolation, latency speedup, fused execution,
production or generalization claim is permitted.
