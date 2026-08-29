# BACKLOG-FLEET-CONTEXT-ENVELOPE-05 preregistration

Task: Correct retained context-envelope recomputation with the historical canonical prompt digest
Evidence class: `serving_runtime`

## Hypothesis

Using the predecessor's canonical JSON prompt digest will reconstruct all 72 retained physical prompts and preserve at least 90% exact recall per text route plus 80% in the weakest position bucket. Any source mismatch, incomplete join or failed gate yields the negative claim.

## Frozen inputs

- `runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/raw/receipt.json`
- `runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/raw/samples.jsonl`
- `runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/raw/case_manifest.json`
- `runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/raw/artifact_hashes.json`
- `runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-04/raw/receipt.json`
- `docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md`
- `config/qualified_model_fleet.json`

- source receipt: `17f0ec8b541f6d769dd5909ca4a44bc3f7c2813d13cc51ec8b206d74090d15d6`
- source samples: `e7ac2131253fc930ff92db9ba3b54994aaf4758d7803fa7c4fc9497af51dea9a`
- source cases: `63936f59148535a54ca54221d29dd669c387c87e07f0f28361899f18ee914111`
- source artifacts: `4a696061f21ff8942ee27a38df3dde8227abc3f2f9c6e650ea851c4c7cc96513`
- source recovery: `8b8b43adc50290970a3efd04439d6bdfa01f15c9c26b129b27474f9ab3387903`
- source service identity: `d96e87884324e337d49b0cb205eb023151c7c2523d22b99af79070afca702b16`
- preserved R4 false-negative receipt: `6559d46b6e02a935db4269eca50d1bad20532183efa789d7a2f7796c9d57c50f`
- admission: `a7c18505b6a37ee9cb5ec5d41034555487c9dadb1cf9d97b4d0bb180c66a1c39`
- audit ledger: `a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04`
- fleet registry: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`

## Command

```powershell
python tools/research/run_retained_fleet_rebind_r3.py --task-id BACKLOG-FLEET-CONTEXT-ENVELOPE-05 --outdir runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-05
```

## Factors

No new inference and no stochastic seed. Recompute exactly 72 retained rows across four text routes, three bounded target lengths, three positions and two replicates. The correction is limited to the historical `canonical_json_sha256(prompt)` contract. Artifact identity, route alias, slot fit, HTTP success, prompt digest and exact code recall remain controls. Retained RTX 3090 timings are not new runtime measurements.

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

Abort before sealing on any hash or receipt-fingerprint mismatch, preregistration mismatch, missing/duplicate join, prompt reconstruction below 100%, malformed evidence, incomplete provenance or harness exception. Never rewrite or delete the R4 false-negative packet.

## Allowed claims

- `QUALIFIED_TEXT_FLEET_SLOT_CONTEXT_ENVELOPES_REBOUND_R5`
- `QUALIFIED_TEXT_FLEET_SLOT_CONTEXT_ENVELOPES_NOT_CONFIRMED_R5`

Claims outside these codes are forbidden even if a metric looks favorable.
