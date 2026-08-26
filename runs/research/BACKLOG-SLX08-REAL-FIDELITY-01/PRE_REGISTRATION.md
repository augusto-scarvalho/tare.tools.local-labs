# BACKLOG-SLX08-REAL-FIDELITY-01 preregistration

Task: Retest the rejected SLX-08 fidelity gate with real Qwen QKV and the computed block indices
Evidence class: `proxy_realization`

## Hypothesis

The `SLX-08` fidelity rejection may be a false negative caused by an implementation error: its probe computes `selected_indices` but then ignores them and attends to the first half of the sequence. On actual Qwen3.5-0.8B QKV activations, gathering the computed top 50% blocks will achieve median last-token attention-context cosine at least 0.95 against dense attention across the frozen cells.

This successor retests only the rejected fidelity gate. It does not implement or claim TTFT acceleration.

## Frozen inputs

- `runs/research/SLX-08-SPECULATIVE-PREFILL-2026-08-25/PRE_REGISTRATION.md`
- `runs/research/SLX-08-SPECULATIVE-PREFILL-2026-08-25/RESULT.md`
- `runs/research/SLX-08-SPECULATIVE-PREFILL-2026-08-25/raw/receipt.json`
- `tools/probes/slx08_speculative_prefill_oracle.py`

- Admission specification: 2,156 bytes, SHA-256 `ac43e4dceb11ce8b5dc9eaae43ebb544ebe10057de5d756215789ad61e08d672`.
- Original preregistration: 1,733 bytes, SHA-256 `5004d124c4f7543a2542916f05c45ec52afced4be4758ff2c95fa386dd4c6212`.
- Original result: 2,499 bytes, SHA-256 `c3d87dd4624e1e2c851df96f6956efb44508df0423bf676657b8c11ae6ade0b7`.
- Original receipt: 899 bytes, SHA-256 `f19600ed451d5ed4ad3a24b5c29ef3fbcf2a95de06ffcab0aef9a7e9152cb78a`.
- Original probe: 5,635 bytes, SHA-256 `5b85dd266c3fc72ae47a7cabe6e5ae3246e4aab544e87e6ee7cd47eab81bdc37`.
- Frozen GSM8K corpus: 389,701 bytes, SHA-256 `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`.
- Qwen weights/config/tokenizer identities remain `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`, `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`, and `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`.

## Command

```powershell
python tools/research/run_slx08_real_fidelity.py --outdir runs/research/BACKLOG-SLX08-REAL-FIDELITY-01
```

## Factors

- Two deterministic 8,192-token contexts assembled from ordered frozen GSM8K rows.
- Six actual full-attention layers: `3, 7, 11, 15, 19, 23`, yielding 12 QKV cells.
- Qwen query projection is split exactly as the installed model implementation specifies; Q/K RMS normalization is applied. The treatment and dense baseline use the same pre-RoPE QKV tensors, so rotary position is held constant by omission in both arms.
- Dense baseline: exact last-token scaled dot-product attention over all keys/values.
- Corrected treatment: pool K in 256-token blocks, score every block with the last query, select the top 50%, gather those exact indices from K and V, and compute last-token attention.
- Legacy-bug control: attend to the first 50% of tokens, reproducing the old probe's ignored-index behavior.
- Primary metric: per-cell cosine between dense and corrected attention context; aggregate by median over 12 cells. Threshold remains the original `>=0.95` fidelity gate.
- Materiality requires the gathered corrected block set to differ from the legacy first-half set in every cell.
- No TTFT comparison is admissible because the treatment is an offline last-token mechanism screen, not an integrated prefill runtime.
- RTX 3090 / WSL environment and systemd restoration contract are the same as the preceding real-model screens.

## Acceptance gates

- `actual_qkv_coverage`: `actual_qkv_cells ge 12`
- `no_synthetic_decisive_tensors`: `all_decisive_tensors_from_frozen_model eq True`
- `computed_indices_used`: `computed_top_block_indices_materially_used eq True`
- `independent_recompute`: `independent_metric_recompute_match eq True`
- `fidelity`: `median_selected_block_context_cosine ge 0.95`
- `service_recovery`: `service_and_embedding_restored eq True`

## Abort conditions

- Any frozen artifact hash differs.
- Any decisive Q, K or V tensor is synthetic or fewer than 12 cells complete.
- The implementation computes indices but does not use them to gather both K and V.
- Corrected and legacy selected block sets are identical in any cell.
- Model/CUDA execution, independent aggregation, provenance or service restoration is incomplete.
- Port 8081 becomes unhealthy or the serving executable/arguments and restart count cannot be restored.

## Allowed claims

- `SLX08_FIDELITY_FALSE_NEGATIVE_CANDIDATE_R1`
- `SLX08_FIDELITY_NEGATIVE_RETAINED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.
