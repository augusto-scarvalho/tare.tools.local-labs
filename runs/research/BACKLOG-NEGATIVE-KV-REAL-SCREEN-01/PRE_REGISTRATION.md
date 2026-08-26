# BACKLOG-NEGATIVE-KV-REAL-SCREEN-01 preregistration

Task: Retest five AGY negative KV conclusions on real Qwen activations and weights
Evidence class: `proxy_realization`

## Hypothesis

At least one of the five AGY negative conclusions below may be a false negative because its decisive tensor was synthetic or randomly constructed rather than extracted from the named Qwen model. Replacing those tensors with actual weights and forward-pass activations from the frozen Qwen3.5-0.8B model will independently determine whether each original success rule is met:

- RSH-01: Fibonacci INT4 must beat uniform INT4 by at least 30% MSE, gain at least 2.5 dB SQNR and retain cosine at least 0.995.
- REP-03: Hadamard INT4 must reduce reconstruction MSE by at least 50% and retain attention cosine at least 0.99.
- RSH-03: a rank-4 residual must recover at least 50% output MSE, reach output cosine at least 0.998 and cost at most 1% of matrix parameters.
- RSH-04: 128-bit binary sketches retaining at most 30% of blocks must recall at least 90% of exact top-attention blocks.
- REP-06: entropy-guided precision must average at most 7 bits per element, reach attention cosine at least 0.992 and beat static INT4 fidelity.

This is a real-model mechanism screen, not a packed codec, native-kernel or serving qualification. A passed candidate rule identifies a false-negative candidate requiring its own physical successor; a failed rule retains the bounded negative on this frozen screen.

## Frozen inputs

- `runs/research/RSH-01-FIBQUANT-2026-08-25/RESULT.md`
- `runs/research/REP-03-KVARN-OFFLINE-2026-08-25/RESULT.md`
- `runs/research/RSH-03-KVLINC-COMPENSATION-2026-08-25/RESULT.md`
- `runs/research/RSH-04-RABITQ-CACHE-2026-08-25/RESULT.md`
- `runs/research/REP-06-ENTROPY-PRECISION-2026-08-25/RESULT.md`
- `tools/probes/rsh01_fibquant_simulation.py`
- `tools/probes/rep03_kvarn_codec.py`
- `tools/probes/rsh03_kvlinc_compensation.py`
- `tools/probes/rsh04_rabitq_cache.py`
- `tools/probes/rep06_dynamic_entropy_precision.py`

- Admission specification: 4,116 bytes, SHA-256 `16d845d3debc066fb9aa2852e9249605c7fc96d1005b089338846ae0b4dbff68`.
- Frozen GSM8K corpus `workloads/gsm8k.jsonl`: 389,701 bytes, SHA-256 `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`.
- Qwen model weights: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/model.safetensors-00001-of-00001.safetensors`, 1,746,942,600 bytes, SHA-256 `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`.
- Qwen config: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/config.json`, 2,907 bytes, SHA-256 `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`.
- Qwen tokenizer: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/tokenizer.json`, 12,807,196 bytes, SHA-256 `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`.

Original result and implementation identities:

| Candidate | Result SHA-256 | Original implementation SHA-256 |
|---|---|---|
| RSH-01 | `f68d18cd0113f0e36a3b1146e4840b2f4fb7cfff8913c21898d0d7da40626d7e` | `8a0cd28fe10d3c6b50a7bcf5df4740442a043b3717b1807099dfae4ec6bfa80d` |
| REP-03 | `3ec9a7f93659511d04a1254a05f9ad792a3ffd3ac543b244a40401305e0b68d1` | `b27ff1cd5df40e82f58c1f6719a9e992fec6df5948a307688f643690e26f1973` |
| RSH-03 | `4e6fb2bdd5eb6949fb1a7a57e3861c5517f34b97e7dec78f92f0bc0a9f0ee014` | `a77eee4852575149186d54eff35ff9afa7e937a5adea9c15482d37613068e346` |
| RSH-04 | `9873ce73c576e2a2a055c3dd29983c213736c510a1de1cfc2da212ba23147e48` | `ced47b5c945bc496d53eff40f12cf34bb36ad0826bf9a315a09dfe5e517926ea` |
| REP-06 | `98d45bad61c76787d585e61e1aa9fc7742922f8b2ebd27f0b935c0202795530b` | `709be19a12535435cde12aecac792f3175c132892cbba938f31b01f503af879d` |

## Command

```powershell
python tools/research/run_negative_kv_real_screen.py --outdir runs/research/BACKLOG-NEGATIVE-KV-REAL-SCREEN-01
```

## Factors

- Hardware/runtime: NVIDIA RTX 3090; WSL2 Ubuntu-24.04; `/home/augus/.venvs/adapt00-20260824`; PyTorch/Transformers versions captured at execution.
- Inputs: three deterministic 4,096-token contexts built only from ordered rows of the frozen GSM8K corpus. No generated random input tensor may affect a decisive metric.
- Activation cells: six actual full-attention layers (`3, 7, 11, 15, 19, 23`) x three contexts = 18. K projections use the model's actual two KV heads and 256-dimensional head size. REP-03 and REP-06 use the first 2,048 positions; RSH-04 uses all 4,096 positions.
- Weight cells: two non-overlapping 1,024 x 1,024 row slices of each actual full-attention `q_proj.weight`, yielding 12 real matrices. Their forward inputs are captured from the same model pass; no Gaussian surrogate is allowed.
- Determinism: algorithm seed `20260824` is used only for the RSH-04 projection matrix, which is part of the treatment rather than an input substitute. RSH-04 reports the median across projection seeds `20260824`, `20260825`, and `20260826` for every activation cell.
- Baselines: uniform symmetric INT4 block-32 for RSH-01; direct symmetric INT4 block-32 for REP-03/RSH-03/REP-06; exact FP32 attention/block rankings for REP-03/RSH-04/REP-06.
- Aggregation: every candidate decision metric is the median across its frozen cells. Ratios and gains are computed per cell before aggregation. The result must retain all individual cell rows.
- Entropy policy: REP-06 derives token entropy from the frozen model's own next-token logits, evaluated in bounded chunks, then applies the original thresholds `<0.8`, `[0.8,2.0)`, and `>=2.0` nats to assign INT2, INT4, and FP16.
- Service isolation: stop `llm-inference.service` through systemd only if VRAM is insufficient, keep embedding port 8081 healthy, and restore the exact serving executable/arguments and restart count afterward.
- Physical limitations: no claimed byte saving or latency is admissible because this screen does not implement packed storage or a native binary-sketch kernel.

## Acceptance gates

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

- Any frozen model, tokenizer, corpus, admission, predecessor result or predecessor implementation hash differs.
- Any decisive tensor is sampled from a synthetic distribution instead of the frozen model.
- Fewer than three complete contexts, 18 activation cells, 12 weight cells or five candidate evaluations are produced.
- Model loading, forward capture, scorer recomputation, CUDA execution or receipt provenance is incomplete.
- Port 8081 becomes unhealthy, or the original persistent inference service executable/arguments and restart count cannot be restored.
- Any threshold, prompt construction rule, layer set, position count, projection seed or aggregation changes after a score is observed.

## Allowed claims

- `NEGATIVE_KV_REAL_SCREEN_VERIFIED_R1`
- `RSH01_FALSE_NEGATIVE_CANDIDATE_R1`
- `REP03_FALSE_NEGATIVE_CANDIDATE_R1`
- `RSH03_FALSE_NEGATIVE_CANDIDATE_R1`
- `RSH04_FALSE_NEGATIVE_CANDIDATE_R1`
- `REP06_FALSE_NEGATIVE_CANDIDATE_R1`

Claims outside these codes are forbidden even if a metric looks favorable.
