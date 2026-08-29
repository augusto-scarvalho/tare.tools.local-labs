# BACKLOG-QWEN38-Q8-KV-CONCURRENCY-01 preregistration

Task: Confirm Qwen3.8 Q8 KV long-context utility under two-slot concurrency
Evidence class: `serving_runtime`

## Hypothesis

With two physically overlapping long-context requests and 32768 tokens allocated per slot, Q8_0 K/V remains noninferior to F16 on the promoted exact-retrieval construct, saves at least 1000 MiB, and retains at least 90% of F16 median concurrent batch output rate. This is a bounded concurrent-correctness and Q8/F16 comparison, not a concurrency-scaling claim.

## Frozen inputs

- `runs/research/BACKLOG-QWEN38-Q8-KV-LONGCONTEXT-02/raw/receipt.json` — SHA-256 `b5f2c691d270754857290fd43aa74c248698b3ad767026bd6dcc116aad0c8575`
- `runs/research/BACKLOG-QWEN38-Q8-KV-LONGCONTEXT-02/REVIEW.json` — SHA-256 `3f759898e5e58e1f5a7d305724be869974a4d75ca71e92274084c703df08a15a`
- `config/qualified_model_fleet.json` — SHA-256 `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`
- R1/R2 generator and chat-contract implementations — SHA-256 `4197afacf8c2bd9c4f3a96802612afcd41a4ca47ed4f1190284d170cf13419c8` and `4c5facc502b25aa1de4465328b18473be6ae74147f932b6e09d1cbfb6487e89a`
- Server, CUDA library and model identities remain exactly those promoted by R2: `efb2f06c...`, `ca185915...`, and `bee238bb...` respectively, with dereferenced sizes 17920, 75928784 and 17923394624 bytes.

## Command

```powershell
python tools/research/run_qwen38_q8_kv_concurrency.py
```

## Factors

- Preserve R2's four-process crossover `F16-Q8 | Q8-F16`, 24 unique paired cases, prompt generator, exact scorer, 8k/16k bands, cache controls, chat contract with thinking disabled, seeds and paired bootstrap.
- Change server shape only to `--parallel 2 --ctx-size 65536`, retaining 32768 tokens per slot.
- Each block executes its 12 cases as six batches of two simultaneous HTTP calls using a two-worker barrier and explicit requested slots 0 and 1. Thus there are 24 batches and 48 requests.
- A batch overlaps only when both measured client request intervals have positive intersection. Both calls must start from the same released barrier. No failed batch is retried or replaced.
- Batch end-to-end output rate is total generated tokens divided by the union wall interval. The Q8/F16 metric compares medians over 12 batches per arm; it is not compared to R2 single-slot throughput.
- Physical prompts must report 8141 or 16149 chat tokens within the existing bands. Nonempty/timing integrity, exact recall, paired bootstrap (20k, seed 2026082823), GPU snapshots and restoration are unchanged.

## Acceptance gates

- `binary_model_identity`: `binary_and_model_identity_verified eq True`
- `cache_treatment_identity`: `explicit_cache_controls_verified eq True`
- `concurrency_identity`: `explicit_two_slot_controls_verified eq True`
- `balanced_crossover`: `valid_crossover_blocks eq 4`
- `sample_size`: `recorded_requests eq 48`
- `concurrent_batches`: `recorded_concurrent_batches eq 24`
- `physical_overlap`: `overlapping_batch_rate eq 1.0`
- `physical_context`: `requests_within_target_token_bands eq 48`
- `request_integrity`: `successful_response_rate eq 1.0`
- `chat_content`: `nonempty_content_rate eq 1.0`
- `timing_integrity`: `nondegenerate_timing_rate eq 1.0`
- `f16_retrieval`: `f16_exact_recall ge 0.95`
- `q8_retrieval`: `q8_exact_recall ge 0.95`
- `paired_noninferiority`: `paired_bootstrap_ci95_low_q8_minus_f16 ge -0.05`
- `batch_throughput_nonregression`: `q8_vs_f16_median_batch_rate_ratio ge 0.9`
- `memory_saving`: `median_vram_saving_mib ge 1000.0`
- `service_recovery`: `service_gateway_embedding_restored eq True`

## Abort conditions

- Abort before requests on identity, service, process argv, cache, `parallel=2` or `ctx-size=65536` mismatch.
- Abort after three consecutive failed batches, missing overlap, malformed response, or embedding failure; retain partial evidence and restore services in `finally`.
- Do not modify cases, batches, slot assignments, thresholds or statistics after observing results.
- Executor stops at `EXECUTED`; independent review is mandatory.

## Allowed claims

- `QWEN38_Q8_KV_CONCURRENT_LONGCONTEXT_NONINFERIOR_R1`
- `QWEN38_Q8_KV_CONCURRENT_LONGCONTEXT_NOT_NONINFERIOR_R1`

No scaling, >2-slot, >16k-per-slot, broad reasoning, other-model or production claim is allowed.
