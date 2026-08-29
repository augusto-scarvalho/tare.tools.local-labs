# BACKLOG-FLEET-CONTEXT-INTERFERENCE-02 preregistration

Task: Rebind and independently recompute retained fleet context interference
Evidence class: `serving_runtime`

## Hypothesis

The 72 physical hard-decoy responses retained by `BACKLOG-FLEET-CONTEXT-INTERFERENCE-01` preserve at least 90% exact recall per model and 80% in the weakest position bucket when recomputed from a fully immutable source set whose prompt construct contains exactly 31 near-label decoys per case.

## Frozen inputs

- `runs/research/BACKLOG-FLEET-CONTEXT-INTERFERENCE-01/raw/receipt.json`
- `runs/research/BACKLOG-FLEET-CONTEXT-INTERFERENCE-01/raw/samples.jsonl`
- `runs/research/BACKLOG-FLEET-CONTEXT-INTERFERENCE-01/raw/case_manifest.json`
- `runs/research/BACKLOG-FLEET-CONTEXT-INTERFERENCE-01/raw/artifact_hashes.json`
- `docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md`
- `config/qualified_model_fleet.json`

- receipt: `96c57a33a1539c8f0e6d3eac1dca20352cc8bfd274e5a72a8ccb00c95667d8de`
- samples: `539196301bc0c8f5cccaebcd9fc0f730ae0a381775925db1625141786f5b3e97`
- case manifest: `86f8cf00668dac9b832e03f0f42537784e02593113c26c087fda5e265e1a6616`
- artifact hashes: `304d121331249e78edff482a08c53e75a0317e604ddff3f0d1bc3005fbeedb0f`
- recovery state: `0b1515faee24eb1079aad969c0001cd050c897d45ea6a336b6b71678d5ae2f67`
- service identity: `1ce9a25296787e14caed231878e34b835e283d5cb444c6f1e473ea97d7027dc9`
- audit ledger: `a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04`
- fleet registry: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`

## Command

```powershell
python tools/research/run_retained_fleet_rebind.py --task-id BACKLOG-FLEET-CONTEXT-INTERFERENCE-02 --outdir runs/research/BACKLOG-FLEET-CONTEXT-INTERFERENCE-02
```

## Factors

No new inference and no stochastic seed. Recompute 72 retained rows over four text models, three bounded target lengths, three positions and two replicates. The frozen interference construct is one exact target label plus 31 near-label decoys. Controls are prompt reconstruction, artifact identity, route alias, slot fit, HTTP status and exact code recall. RTX 3090 timings remain retained metadata only.

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

Abort before sealing on any frozen-input hash mismatch, invalid receipt fingerprint, preregistration mismatch, construct other than exactly 31 decoys, incomplete or duplicate case joins, prompt reconstruction below 100%, malformed evidence, incomplete provenance, or harness exception. Scientific gate failures yield only the allowed negative claim.

## Allowed claims

- `QUALIFIED_TEXT_FLEET_CONTEXT_INTERFERENCE_REBOUND_R2`
- `QUALIFIED_TEXT_FLEET_CONTEXT_INTERFERENCE_NOT_CONFIRMED_R2`

Claims outside these codes are forbidden even if a metric looks favorable.
