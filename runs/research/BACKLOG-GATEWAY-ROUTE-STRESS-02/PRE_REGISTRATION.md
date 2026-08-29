# BACKLOG-GATEWAY-ROUTE-STRESS-02 preregistration

Task: Rebind retained gateway route stress with semantic success separation
Evidence class: `serving_runtime`

## Hypothesis

The retained 30-switch/120-request campaign demonstrates transport and effective-route integrity across all six qualified aliases, while semantic non-empty-content eligibility holds only for the four text aliases. Recomputing these estimands separately will pass every frozen gate without claiming vision-task success.

## Frozen inputs

- `runs/research/BACKLOG-GATEWAY-ROUTE-STRESS-01/raw/receipt.json`
- `runs/research/BACKLOG-GATEWAY-ROUTE-STRESS-01/raw/samples.jsonl`
- `runs/research/BACKLOG-GATEWAY-ROUTE-STRESS-01/raw/switches.jsonl`
- `runs/research/BACKLOG-GATEWAY-ROUTE-STRESS-01/raw/effective_route.json`
- `runs/research/BACKLOG-GATEWAY-ROUTE-STRESS-01/raw/service_identity.json`
- `docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md`
- `config/qualified_model_fleet.json`

- receipt: `527e308b2aa54fe96bb641f1d5380b04b42e7871245d173ee107cec0dabbfe41`
- samples: `b94e55da69cdd0b40e209bf9fdd554be65727e6d483a35a09f1a8b08f6c8f865`
- switches: `a99a7295d3c9a756f8852eafe72c22dfa16fba1ca3b28435443413f1bedd6f60`
- effective route: `fb5b0d0ceff9e1caccc85f921a8fefa795dfee5e7254ac623283250a46ae41d9`
- service identity: `5ac6ebe947d8b27989b8ea19af75711c7c9d3283d9532711e164fb210e21d154`
- recovery state: `9ba2650f2a2b94de2aedde5106e76a2e6d7669562cb4a3fc6925e29dab2c838d`
- audit ledger: `a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04`
- fleet registry: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`

## Command

```powershell
python tools/research/run_retained_fleet_rebind.py --task-id BACKLOG-GATEWAY-ROUTE-STRESS-02 --outdir runs/research/BACKLOG-GATEWAY-ROUTE-STRESS-02
```

## Factors

No new inference and no stochastic seed. Recompute five cycles across six aliases, one switch per alias per cycle, and four fixed probes per switch: 30 switches and 120 requests. Transport success and response alias are evaluated over all rows; non-empty content is evaluated only over the 80 eligible text-model rows. Exact cycle repeatability compares cycles 1-4 to cycle 0. Service restart, embedding boundary and restoration evidence are recomputed from retained physical state.

## Acceptance gates

- `source_receipt`: `source_receipt_digest_verified eq True`
- `immutable_sources`: `final_source_set_immutable eq True`
- `switch_coverage`: `recomputed_switches eq 30`
- `request_coverage`: `recomputed_requests eq 120`
- `transport_integrity`: `http_transport_success_rate eq 1.0`
- `route_identity`: `route_alias_match_rate eq 1.0`
- `text_content_integrity`: `eligible_text_nonempty_content_rate eq 1.0`
- `cycle_repeatability`: `exact_cycle_repeat_rate ge 0.95`
- `artifact_table`: `verified_distinct_model_artifacts eq 6`
- `gateway_integrity`: `gateway_service_restarts eq 0`
- `embedding_integrity`: `embedding_boundary_successes eq 30`
- `service_recovery`: `initial_model_restored eq True`

## Abort conditions

Abort before sealing on a frozen-input or receipt-fingerprint mismatch, preregistration mismatch, malformed response/switch row, missing baseline key, alias outside the registry, incomplete provenance, missing recovery evidence, or harness exception. Never reinterpret empty vision chat content as semantic failure or success beyond the transport-only claim.

## Allowed claims

- `QUALIFIED_GATEWAY_ROUTE_TRANSPORT_STRESS_REBOUND_R2`
- `QUALIFIED_GATEWAY_ROUTE_TRANSPORT_STRESS_NOT_CONFIRMED_R2`

Claims outside these codes are forbidden even if a metric looks favorable.
