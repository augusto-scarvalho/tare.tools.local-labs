# BACKLOG-MTP-PERSISTENCE-FUNCTIONAL-03 preregistration

Task: Test functional MTP slot reuse after save and restore
Evidence class: `serving_runtime`

## Hypothesis

MTP slot save/restore preserves a reusable prefix, not merely slot bytes. In
sixteen physical cycles the restored request will report positive `cache_n`,
reuse at least 80% of the prompt, reduce evaluated prefill tokens by at least
80%, and reproduce the cold greedy continuation exactly.

## Frozen inputs

- `runs/research/BACKLOG-MTP-PERSISTENCE-01/raw/receipt.json`
- `runs/research/BACKLOG-MTP-PERSISTENCE-01/raw/samples.jsonl`
- `runs/research/BACKLOG-MTP-PERSISTENCE-RESCORE-02/raw/receipt.json`
- `tools/research/run_mtp_persistence_first_instance.py`
- `config/qualified_model_fleet.json`

- Admission: `9a706365ebfbd9a46de47e99df643a824c12a0d4000f9283e2d25b53c77a5fa9`.
- R1 receipt: `6a52f8690b9f20c6583bdfa80e947ca4f7469e3297b24e046195a799c4116079`.
- R1 samples: `ae235a397007db32e6c5adc4228f26023fd0a7a5efa4a93cab8d7ba9e6b1b34d`.
- R2 receipt: `b7ab4e3567f48be73c8be933316af1126fcde644f3e0adca8450a0672d611021`.
- Physical runtime runner: `2a70897b1a9d73fb5fbf77159fa8c6706a41786c95bd428061e8963b8abde7b1`.
- Fleet registry: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`.
- Deployed binary/model identities are rechecked against the runner's frozen
  WSL byte sizes and SHA-256 values before service maintenance.

## Command

```powershell
python tools/research/run_mtp_persistence_functional_r3.py --outdir runs/research/BACKLOG-MTP-PERSISTENCE-FUNCTIONAL-03
```

## Factors

- One fresh MTP-enabled Qwen3.8 server process, sixteen independent slot files
  and sixteen unique deterministic long prompts.
- Each cycle is `erase → cold completion → save → erase → restore → same
  completion → erase` with prompt caching enabled and greedy seed 20260826.
- Cold evaluation is the paired baseline; warm restored evaluation is the
  treatment. File lifecycle, cache telemetry, prefill work and content parity
  are separately retained.

## Acceptance gates

- `source_integrity`: `sources_and_runtime_verified eq True`
- `cycle_coverage`: `completed_cycles eq 16`
- `lifecycle_integrity`: `successful_lifecycle_rate eq 1.0`
- `functional_cache_reuse`: `cycles_with_cache_n_positive eq 16`
- `material_cache_reuse`: `median_cache_fraction_of_prompt ge 0.8`
- `prefill_reduction`: `median_prefill_token_reduction ge 0.8`
- `continuation_parity`: `exact_continuation_rate eq 1.0`
- `service_recovery`: `service_and_embedding_restored eq True`

## Abort conditions

Abort on frozen identity drift, occupied transient endpoint, save/restore HTTP
failure, missing slot file, three consecutive request failures, unhealthy 8081,
or inability to restore qwen38/8080. Cache misses and parity failures are data,
not aborts. Stop the persistent service only through systemd.

## Allowed claims

- `MTP_PERSISTENCE_FUNCTIONAL_REUSE_CONFIRMED_R3`
- `MTP_PERSISTENCE_FUNCTIONAL_REUSE_REJECTED_R3`

Claims outside these codes are forbidden even if a metric looks favorable.
