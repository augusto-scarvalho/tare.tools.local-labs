# BACKLOG-NEGATIVE-KV-REAL-SCREEN-02 preregistration

Task: Repeat the frozen five-candidate real Qwen KV screen after pre-measurement model-binding abort
Evidence class: `proxy_realization`

## Hypothesis

The five hypotheses, thresholds, real-model tensor cells, prompt construction, layer set, seeds, baselines and aggregation are identical to the frozen `BACKLOG-NEGATIVE-KV-REAL-SCREEN-01` preregistration. That predecessor stopped before its first forward pass because of a model-object binding error. With the binding corrected to the object actually returned by Transformers 5.15.1, the unchanged screen will determine whether any of RSH-01, REP-03, RSH-03, RSH-04 or REP-06 was falsely rejected on synthetic tensors.

The sole scientific purpose remains detection of false-negative candidates. This proxy-realization screen cannot qualify packed memory, a native kernel or production serving.

## Frozen inputs

- `runs/research/BACKLOG-NEGATIVE-KV-REAL-SCREEN-01/PRE_REGISTRATION.md`
- `runs/research/BACKLOG-NEGATIVE-KV-REAL-SCREEN-01/ABORTED.md`
- `runs/research/BACKLOG-NEGATIVE-KV-REAL-SCREEN-01/raw/worker.stderr.log`
- `runs/research/BACKLOG-NEGATIVE-KV-REAL-SCREEN-01/raw/service_maintenance.json`
- `runs/research/RSH-01-FIBQUANT-2026-08-25/RESULT.md`
- `runs/research/REP-03-KVARN-OFFLINE-2026-08-25/RESULT.md`
- `runs/research/RSH-03-KVLINC-COMPENSATION-2026-08-25/RESULT.md`
- `runs/research/RSH-04-RABITQ-CACHE-2026-08-25/RESULT.md`
- `runs/research/REP-06-ENTROPY-PRECISION-2026-08-25/RESULT.md`

- Admission specification SHA-256: `903dad8498d0d54e9887349ca5477bdf8e531c025b2ed8abd6dd6261a9cf881a`.
- Frozen predecessor preregistration SHA-256: `dce466a9cdfc8a34b7baf1afe52f15752891bd063a32ac3d2f28450d7aebfe11`.
- Frozen predecessor abort record SHA-256: `c5cd9acb6070fb704ccf9f6b17230a2cfcad89b9bebb65daca9f988c8eb6590b`.
- Frozen predecessor stderr SHA-256: `61df470821dfd43c4c9a799c2b2a24e1fa5fa24f936a8eacff279c581d0ebbc4`.
- Frozen predecessor service-restoration record SHA-256: `d4bf245ae8cf5a77665c62aeaba87118be977805a7a036fb1749b1fd3d2f90e7`.
- All original AGY result, original probe, Qwen model/config/tokenizer and GSM8K identities are exactly those enumerated in the predecessor preregistration; they must be recomputed before execution.

## Command

```powershell
python tools/research/run_negative_kv_real_screen_r2.py --outdir runs/research/BACKLOG-NEGATIVE-KV-REAL-SCREEN-02
```

## Factors

- Import no score or tensor from the failed predecessor; it contains none.
- Verify the predecessor preregistration, abort, stderr and service-restoration hashes before loading the model.
- Retain three deterministic 4,096-token frozen GSM8K contexts; layers `3, 7, 11, 15, 19, 23`; 18 activation cells; 12 real 1,024 x 1,024 weight slices; the first 2,048 positions for REP-03/REP-06; all 4,096 for RSH-04; and projection seeds `20260824` through `20260826`.
- Retain all original candidate thresholds, block size 32, entropy boundaries, per-cell calculations and median aggregation from the predecessor.
- The only model-side correction is binding `language = model.model` instead of accessing nonexistent `model.model.language_model`. Host changes may only select the successor task ID, verify the frozen failure and record continuation evidence.
- Service isolation and restoration rules remain unchanged.

## Acceptance gates

- `continuation_integrity`: `frozen_failed_predecessor_verified eq True`
- `actual_activation_coverage`: `actual_model_activation_cells ge 18`
- `actual_weight_coverage`: `actual_model_weight_matrices ge 12`
- `candidate_coverage`: `candidate_hypotheses_evaluated eq 5`
- `no_synthetic_decisive_tensors`: `all_decisive_tensors_from_frozen_model eq True`
- `independent_recompute`: `independent_metric_recompute_match eq True`
- `rsh01_mse`: `rsh01_fib_mse_ratio_vs_uniform le 0.7`
- `rsh01_sqnr`: `rsh01_fib_sqnr_gain_db ge 2.5`
- `rsh01_cosine`: `rsh01_fib_cosine_similarity ge 0.995`
- `rep03_mse`: `rep03_hadamard_mse_reduction ge 0.5`
- `rep03_attention`: `rep03_hadamard_attention_cosine ge 0.99`
- `rsh03_recovery`: `rsh03_rank4_mse_recovery ge 0.5`
- `rsh03_cosine`: `rsh03_rank4_output_cosine ge 0.998`
- `rsh03_overhead`: `rsh03_rank4_parameter_overhead le 0.01`
- `rsh04_recall`: `rsh04_binary_top_block_recall ge 0.9`
- `rsh04_dram`: `rsh04_retained_fraction le 0.3`
- `rep06_bits`: `rep06_average_bits_per_element le 7.0`
- `rep06_attention`: `rep06_dynamic_attention_cosine ge 0.992`
- `rep06_beats_static`: `rep06_dynamic_beats_static_int4 eq True`
- `service_recovery`: `service_and_embedding_restored eq True`

## Abort conditions

- Any predecessor, model, tokenizer, corpus, AGY result or original-probe hash differs.
- Any experimental factor differs from the predecessor preregistration beyond the explicit model-object binding correction.
- Any decisive tensor is synthetic, or fewer than three contexts, 18 activation cells, 12 weight cells or five candidates complete.
- CUDA/model execution, independent aggregation, provenance or service restoration is incomplete.
- Port 8081 becomes unhealthy or the persistent serving executable/arguments and restart count cannot be restored.

## Allowed claims

- `NEGATIVE_KV_REAL_SCREEN_VERIFIED_R2`
- `RSH01_FALSE_NEGATIVE_CANDIDATE_R2`
- `REP03_FALSE_NEGATIVE_CANDIDATE_R2`
- `RSH03_FALSE_NEGATIVE_CANDIDATE_R2`
- `RSH04_FALSE_NEGATIVE_CANDIDATE_R2`
- `REP06_FALSE_NEGATIVE_CANDIDATE_R2`

Claims outside these codes are forbidden even if a metric looks favorable.
