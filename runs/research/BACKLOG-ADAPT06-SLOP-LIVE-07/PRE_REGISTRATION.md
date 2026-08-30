# BACKLOG-ADAPT06-SLOP-LIVE-07 preregistration

Task: Qualify two-adapter client affinity behind the qualified-model gateway
Evidence class: `serving_runtime`

## Hypothesis

Resolving the temporary physical llama-server from the frozen fleet registry,
rather than assuming port 8080 systemd ExecStart is a model server, will allow
the complete R6 treatment to run behind the qualified-model gateway. Grouping
must reduce requested route switches by at least 90% with exact route-correct
counterfactual parity and exact restoration of the gateway command.

## Frozen inputs

- R6 aborted terminal: `e7e83bb158ced87e973b71e1fa0b18252ab8d13a585e21a78a89ab09c2a841b0`
- R6 runner: `4f1c94b180366123e65e94e2ccd395e29108bbf92813417d6a02f4ad39d80e20`
- R5 physical runner: `7fc57290cf59fb826f18306e3e861a1019680206027b771203fb5e5441269922`
- qualified fleet registry: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`
- behavioral LoRA: `05b80090d2d1ba751d48a5032cddec82819a79b9bfb5bd8b05306b85d6ef0122`
- trace LoRA: `174832aa1bd25cbc5ed7f0ff717ad253ec94e2c23edc82e6f828ceadeed566b7`

## Command

```powershell
python tools/research/run_adapt06_slop_live_r7.py --outdir runs/research/BACKLOG-ADAPT06-SLOP-LIVE-07
```

## Factors

- R6 factors, prompts, routes, counterfactuals, schedules, seed and gates remain unchanged.
- Sole execution correction: obtain the temporary llama-server path from
  `models.qwen38.runtime.binary` and bind the real gateway ExecStart/MainPID
  before and after the controlled systemd handoff.
- The compatibility view exists only inside the historical R5 helper; retained
  service evidence preserves both real gateway identity and temporary physical binary.
- Harness owns and seals all raw evidence; watcher controls completion.

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

Abort on any R6 abort condition, fleet-registry or binary mismatch, missing
gateway identity, gateway command drift, service collision, incomplete physical
rows/logs, embedding health loss or restoration failure. Preserve the R6 abort.

## Allowed claims

- `ADAPT06_GATEWAY_BOUND_CLIENT_AFFINITY_QUALIFIED_R7`
- `ADAPT06_GATEWAY_BOUND_CLIENT_AFFINITY_REJECTED_R7`

Claims outside these codes are forbidden. Server-native scheduling, fused GEMM,
production deployment and generalization beyond the frozen routes remain forbidden.
