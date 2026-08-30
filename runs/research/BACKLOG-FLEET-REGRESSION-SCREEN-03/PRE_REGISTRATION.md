# BACKLOG-FLEET-REGRESSION-SCREEN-03 preregistration

Task: Longitudinal replication of the qualified text-fleet regression screen
Evidence class: `serving_runtime`

## Hypothesis

A fresh execution of the complete qualified-fleet screen will retain 100%
transport success, at least 95% exact greedy repeatability and correct physical
route identity across all four aliases without restarting the gateway or
damaging the embedding service.

## Frozen inputs

- Admission: `8746a25d050af19e49c5e8e68499a72cc31e3b8f130ea143681e85c0530cd0c6`.
- Promoted R2 receipt and review:
  `23f3f94634e3e9be5451bd0daafcb7e28dd7d338ddf9bddbe6b81b15a267e598`,
  `e0fd0c20f510bb70b50694a828450b6e47bd9fd86ae798059e34a3ae5836be27`.
- Frozen R2 wrapper and base screen:
  `7f577e3e3ab9054b7dd48f0279852b033fd4ba323c53b9d3ee06cdb710303fd7`,
  `7cbf942375c120970ac78fb40f7f15050ebddb7ec5372cb40bd721009faf1de3`.
- Fleet registry, math panel, QA panel and agent suite:
  `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`,
  `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`,
  `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`,
  `14d0a1b76d4d729228678f215ecefa3254aef214eb65ac9d8d7061bccc0dc59e`.
- The registry freezes each model artifact, runtime binary and argument list;
  all four WSL artifact hashes are reverified before requests begin.

## Command

```powershell
python tools/research/run_fleet_regression_screen_r3.py --outdir runs/research/BACKLOG-FLEET-REGRESSION-SCREEN-03
```

## Factors

- Four physical routes: `qwen38`, `hauhaucs`, `fable-tc`, `qwen36-moe`.
- Two greedy repeats per case over 32 GSM8K, 16 protected QA and eight agent
  tool cases: 448 independently issued requests in total.
- Temperature 0, seed 20260826, prompt cache disabled and complete request and
  response retention.
- Route switches are observed through the gateway physical status. MainPID,
  restart count, embedding health and the initially resident route are captured
  before and after.
- The outer experiment harness seals the complete final state after restoration;
  the predecessor's post-receipt runner-state mutation cannot recur.

## Acceptance gates

- promoted R2 binding equals true;
- four routes and exactly 448 retained requests;
- HTTP success rate equals 1.0 and physical route identity equals true;
- exact semantic repeat rate is at least 0.95;
- all 448 request payloads are retained;
- gateway restart delta equals zero, embedding health equals 200 and the
  initially resident route is restored.

## Abort conditions

Abort on any frozen artifact drift, unhealthy gateway/embedding preflight,
unverified route identity, fewer than 512 MiB free after a switch, three
consecutive request failures, malformed evidence or restoration failure. The
harness must emit an aborted terminal rather than a scientific verdict.

## Allowed claims

- `QUALIFIED_TEXT_FLEET_LONGITUDINAL_REPLICATION_R3`
- `QUALIFIED_TEXT_FLEET_LONGITUDINAL_REJECTED_R3`

The claim covers only this bounded transport, routing and repeatability screen.
It does not promote model quality, cross-model superiority, production SLOs or
semantic correctness from HTTP success.
