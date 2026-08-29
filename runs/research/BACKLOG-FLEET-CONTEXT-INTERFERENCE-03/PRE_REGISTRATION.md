# BACKLOG-FLEET-CONTEXT-INTERFERENCE-03 preregistration

Task: Correct retained context-interference recomputation after recursive wrapper abort
Evidence class: `serving_runtime`

## Hypothesis

A non-recursive versioned wrapper using the canonical JSON prompt digest will reconstruct all 72 retained physical 31-decoy prompts and preserve at least 90% exact recall per route plus 80% in the weakest position bucket. Any failed gate yields the negative claim.

## Frozen inputs

- `runs/research/BACKLOG-FLEET-CONTEXT-INTERFERENCE-01/raw/receipt.json`
- `runs/research/BACKLOG-FLEET-CONTEXT-INTERFERENCE-01/raw/samples.jsonl`
- `runs/research/BACKLOG-FLEET-CONTEXT-INTERFERENCE-01/raw/case_manifest.json`
- `runs/research/BACKLOG-FLEET-CONTEXT-INTERFERENCE-01/raw/artifact_hashes.json`
- `runs/research/BACKLOG-FLEET-CONTEXT-INTERFERENCE-02/raw/run.terminal.json`
- `runs/research/BACKLOG-FLEET-CONTEXT-INTERFERENCE-02/runner.stderr.log`
- `docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md`
- `config/qualified_model_fleet.json`

- source receipt: `96c57a33a1539c8f0e6d3eac1dca20352cc8bfd274e5a72a8ccb00c95667d8de`
- source samples: `539196301bc0c8f5cccaebcd9fc0f730ae0a381775925db1625141786f5b3e97`
- source cases: `86f8cf00668dac9b832e03f0f42537784e02593113c26c087fda5e265e1a6616`
- source artifacts: `304d121331249e78edff482a08c53e75a0317e604ddff3f0d1bc3005fbeedb0f`
- source recovery: `0b1515faee24eb1079aad969c0001cd050c897d45ea6a336b6b71678d5ae2f67`
- source service identity: `1ce9a25296787e14caed231878e34b835e283d5cb444c6f1e473ea97d7027dc9`
- preserved R2 aborted terminal: `5f423740344e35dc9105893331cea302ffa8282af858995b51e78a0ed9a25778`
- preserved R2 traceback: `8da89e242c912622680e297a7ee4922683f47b7eda4e4fe6b6474c52a4608e42`
- admission: `d3a76aa42247106a0eb9b7b6c786645af52cb321147bb73dc630db87b51c2c66`
- audit ledger: `a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04`
- fleet registry: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`

## Command

```powershell
python tools/research/run_retained_fleet_rebind_r3.py --task-id BACKLOG-FLEET-CONTEXT-INTERFERENCE-03 --outdir runs/research/BACKLOG-FLEET-CONTEXT-INTERFERENCE-03
```

## Factors

No new inference and no stochastic seed. Recompute exactly 72 retained rows across four text routes, three bounded lengths, three positions and two replicates. Every frozen case contains one exact target and 31 near-label decoys. Controls are canonical prompt reconstruction, construct identity, artifact identity, route alias, slot fit, HTTP success and exact target recall. The non-recursive implementation is fixture-tested against both physical source sets.

## Acceptance gates

- `source_receipt`: `source_receipt_digest_verified eq True`
- `immutable_sources`: `final_source_set_immutable eq True`
- `retained_rows`: `retained_rows_recomputed eq 72`
- `prompt_reconstruction`: `prompt_hash_reconstruction_rate eq 1.0`
- `decoy_construct`: `cases_with_exactly_31_verified_decoys eq 72`
- `artifact_identity`: `verified_model_artifacts eq 4`
- `route_identity`: `route_alias_match_rate eq 1.0`
- `context_fit`: `requests_within_route_slot_context eq 72`
- `request_integrity`: `successful_response_rate eq 1.0`
- `qwen38_recall`: `qwen38_exact_recall ge 0.9`
- `hauhaucs_recall`: `hauhaucs_exact_recall ge 0.9`
- `fable_recall`: `fable_tc_exact_recall ge 0.9`
- `qwen36_moe_recall`: `qwen36_moe_exact_recall ge 0.9`
- `position_robustness`: `minimum_position_bucket_recall ge 0.8`

## Abort conditions

Abort before sealing on any frozen hash mismatch, invalid source receipt, preregistration mismatch, construct other than 31 decoys, incomplete/duplicate join, prompt reconstruction below 100%, incomplete provenance or harness exception. Preserve the R2 aborted terminal and traceback unchanged.

## Allowed claims

- `QUALIFIED_TEXT_FLEET_CONTEXT_INTERFERENCE_REBOUND_R3`
- `QUALIFIED_TEXT_FLEET_CONTEXT_INTERFERENCE_NOT_CONFIRMED_R3`

Claims outside these codes are forbidden even if a metric looks favorable.
