# BACKLOG-FLEET-CONTEXT-ENVELOPE-04 preregistration

Task: Rebind and independently recompute the retained fleet context envelope
Evidence class: `serving_runtime`

## Hypothesis

The 72 physical responses retained by `BACKLOG-FLEET-CONTEXT-ENVELOPE-03` are sufficient to reproduce its bounded per-slot exact-recall conclusion when every final input is immutable and every prompt is independently reconstructed. Any hash mismatch, incomplete join, or failed manifest gate falsifies the rebound claim.

## Frozen inputs

- `runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/raw/receipt.json`
- `runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/raw/samples.jsonl`
- `runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/raw/case_manifest.json`
- `runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/raw/artifact_hashes.json`
- `docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md`
- `config/qualified_model_fleet.json`

- receipt: `17f0ec8b541f6d769dd5909ca4a44bc3f7c2813d13cc51ec8b206d74090d15d6`
- samples: `e7ac2131253fc930ff92db9ba3b54994aaf4758d7803fa7c4fc9497af51dea9a`
- case manifest: `63936f59148535a54ca54221d29dd669c387c87e07f0f28361899f18ee914111`
- artifact hashes: `4a696061f21ff8942ee27a38df3dde8227abc3f2f9c6e650ea851c4c7cc96513`
- recovery state: `8b8b43adc50290970a3efd04439d6bdfa01f15c9c26b129b27474f9ab3387903`
- service identity: `d96e87884324e337d49b0cb205eb023151c7c2523d22b99af79070afca702b16`
- audit ledger: `a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04`
- fleet registry: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`

## Command

```powershell
python tools/research/run_retained_fleet_rebind.py --task-id BACKLOG-FLEET-CONTEXT-ENVELOPE-04 --outdir runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-04
```

## Factors

No new inference and no stochastic seed. Recompute exactly 72 retained rows: four text models, three bounded target lengths per model, three target positions and two replicates. Controls are the stored prompt digests, model artifact identities, per-route slot limits, HTTP status, response alias and exact target code. Hardware timings are retained observations from the RTX 3090 run and are not reinterpreted as a new runtime measurement.

## Acceptance gates

- `source_receipt`: `source_receipt_digest_verified eq True`
- `immutable_sources`: `final_source_set_immutable eq True`
- `retained_rows`: `retained_rows_recomputed eq 72`
- `prompt_reconstruction`: `prompt_hash_reconstruction_rate eq 1.0`
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

Abort before sealing on any frozen-input hash mismatch, invalid source receipt fingerprint, preregistration mismatch, duplicate/missing case join, prompt reconstruction below 100%, malformed row, absent required evidence, incomplete provenance, or harness exception. A failed scientific gate produces the allowed negative claim; it must not be rewritten as success.

## Allowed claims

- `QUALIFIED_TEXT_FLEET_SLOT_CONTEXT_ENVELOPES_REBOUND_R4`
- `QUALIFIED_TEXT_FLEET_SLOT_CONTEXT_ENVELOPES_NOT_CONFIRMED_R4`

Claims outside these codes are forbidden even if a metric looks favorable.
