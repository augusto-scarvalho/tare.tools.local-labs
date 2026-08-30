# BACKLOG-ADAPT06-SLOP-LIVE-06 preregistration

Task: Qualify two-adapter client affinity with bound schedules and route counterfactuals
Evidence class: `serving_runtime`

## Hypothesis

On the frozen matching Qwen3.5 0.8B base and two physical LoRA adapters, grouping
requests by requested adapter will reduce route switches by at least 90% while
every output remains identical to a fresh isolated route-correct counterfactual.
All schedule, cache and route-log evidence must be inside the sealed terminal.

## Frozen inputs

- `runs/research/BACKLOG-ADAPT06-SLOP-LIVE-05/raw/receipt.json`: `871fd8aeb94ff4b2e4eeb6432ba10305591c01b6270462686af0a116ec8d3a28`
- `runs/research/BACKLOG-ADAPT06-SLOP-LIVE-05/raw/live_rows.json`: `9ad728eef78827899ad27920a81a8273c99ac75808517286529a0538fbca30e0`
- `docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md`: `a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04`
- `tools/research/run_adapt06_slop_live_r5.py`: `7fc57290cf59fb826f18306e3e861a1019680206027b771203fb5e5441269922`
- behavioral LoRA: `05b80090d2d1ba751d48a5032cddec82819a79b9bfb5bd8b05306b85d6ef0122`
- trace LoRA: `174832aa1bd25cbc5ed7f0ff717ad253ec94e2c23edc82e6f828ceadeed566b7`

## Command

```powershell
python tools/research/run_adapt06_slop_live_r6.py --outdir runs/research/BACKLOG-ADAPT06-SLOP-LIVE-06
```

## Factors

- RTX 3090; frozen Qwen3.5 0.8B base; deterministic greedy decoding.
- Three routes: base, behavioral LoRA and trace LoRA; 12 frozen prompts.
- 36 isolated route-correct baselines and 72 routed repeats.
- Alternating and grouped 30-request schedules over the same route/prompt cells.
- Cache switch/return sequences on three fixed slots.
- The harness owns the raw directory, records each measured row, requires
  restoration, and seals schedules, cache rows and server logs atomically.

## Acceptance gates

- `adapter_conversion`: `converted_adapters eq 2`
- `adapter_loading`: `loaded_adapters eq 2`
- `behavioral_materiality`: `prompts_with_distinct_route_outputs ge 4`
- `bound_live_rows`: `digest_bound_live_rows eq True`
- `isolated_counterfactuals`: `route_correct_counterfactual_match_rate eq 1.0`
- `route_isolation`: `routed_exact_match_rate eq 1.0`
- `cross_route_isolation`: `cross_route_contamination_count eq 0`
- `affinity_switch_reduction`: `requested_route_switch_reduction ge 0.9`
- `affinity_parity`: `schedule_semantic_parity eq 1.0`
- `service_restore`: `original_service_restored eq 1`
- `embedding_integrity`: `embedding_health eq 200`

## Abort conditions

Abort on any frozen-input mismatch, conversion/loading failure, missing route
log, incomplete schedule/cache/counterfactual row, request failure, service
identity mismatch, port collision, embedding health loss, restoration failure,
nonfinite metric or unsealed auxiliary evidence. Never infer server-native
scheduling from client request order.

## Allowed claims

- `ADAPT06_TWO_ADAPTER_CLIENT_AFFINITY_QUALIFIED_R6`
- `ADAPT06_TWO_ADAPTER_CLIENT_AFFINITY_REJECTED_R6`

Claims outside these codes are forbidden even if a metric looks favorable.
Server-native affinity, fused multi-adapter GEMM, production deployment and
generalization beyond the frozen two-adapter treatment remain forbidden.
