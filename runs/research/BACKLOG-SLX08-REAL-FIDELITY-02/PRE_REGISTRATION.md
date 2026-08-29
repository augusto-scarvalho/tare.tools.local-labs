# BACKLOG-SLX08-REAL-FIDELITY-02 preregistration

Task: Requalify corrected SLX-08 fidelity with retained context vectors
Evidence class: `proxy_realization`

## Hypothesis

The R1 fidelity pass is not an aggregation artifact. On the same two frozen
8,192-token contexts and six physical full-attention layers, the corrected
top-50% block gather will again achieve median last-token attention-context
cosine at least 0.95. A separately hashed scorer reopening retained dense,
corrected and legacy context vectors will reproduce all 12 cell projections
exactly. This tests only offline fidelity and the historical false-negative
candidate; it does not test an integrated prefill implementation or TTFT.

## Frozen inputs

- `runs/research/BACKLOG-SLX08-REAL-FIDELITY-01/raw/receipt.json`
- `docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md`
- `tools/research/run_slx08_real_fidelity.py`
- `tools/research/slx08_real_fidelity_worker.py`
- `workloads/gsm8k.jsonl`

- Admission specification: 2,614 bytes, SHA-256
  `f8bfd39378849d3e251e9e227187cabec8ef5153d2da4fa01b7425a9a8cc5369`.
- R1 receipt: 10,578 bytes, SHA-256
  `6e5212692ff4e8fa3ac50eab13e144a7e08cb933b2c44ecf3bf55568c9b4e660`.
- Independent audit ledger: 68,324 bytes, SHA-256
  `a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04`.
- R1 host runner: 11,005 bytes, SHA-256
  `0b8e7bc733d7bad7e51ee2edc3586f1259ce1bbcfe4d5f1357ebcc109564efa5`.
- R1 GPU worker: 7,576 bytes, SHA-256
  `2b628e2fdf864216d40e538ac60cb1f9ac09f6d325aa6c70225bc1c05fa8e1b7`.
- Frozen GSM8K corpus: 389,701 bytes, SHA-256
  `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`.
- Physical Qwen weights/config/tokenizer must match the R1 receipt before any
  forward. The new worker, scorer and host runner are frozen by the
  `PREREGISTERED -> IMPLEMENTED` transition before execution.

## Command

```powershell
python tools/research/run_slx08_real_fidelity_r2.py --outdir runs/research/BACKLOG-SLX08-REAL-FIDELITY-02
```

## Factors

- Frozen panel: two deterministic 8,192-token contexts assembled from the
  ordered GSM8K corpus and layers `3, 7, 11, 15, 19, 23` (12 cells).
- Dense control: last-token scaled dot-product attention over all normalized
  real-model K/V tensors.
- Corrected treatment: score 256-token K blocks, retain the top 16 of 32 per
  head, gather the exact selected K/V tokens, then recompute attention.
- Legacy control: first 50% of tokens, preserving the historical ignored-index
  behavior.
- Retained bundle: dense, corrected and legacy context vectors for every cell,
  with cell keys, selected indices, Q/K/V hashes and source projections in
  JSON. Full QKV retention is deliberately excluded to avoid roughly 1 GiB of
  redundant evidence; the retained vectors are the decisive inputs to every
  frozen cosine gate.
- Independent scorer: a separate process opens the bundle, requires exactly
  36 tensors (three arms times 12 cells), recomputes shape, finiteness, tensor
  hashes and all cell/median cosines, and compares them with worker projections.
- Runtime: the same RTX 3090/WSL model environment as R1. The inference service
  may be stopped for VRAM only under the existing restoration contract; port
  8081 must remain healthy.
- No seed or adaptive sampling is used. No new panel, threshold, layer or
  aggregation choice may be introduced after execution begins.

## Acceptance gates

- `actual_qkv_coverage`: `actual_qkv_cells ge 12`
- `no_synthetic_decisive_tensors`: `all_decisive_tensors_from_frozen_model eq True`
- `computed_indices_used`: `computed_top_block_indices_materially_used eq True`
- `context_bundle_coverage`: `retained_context_cells eq 12`
- `context_projection_match`: `recomputed_projection_match_rate eq 1.0`
- `fidelity`: `recomputed_median_selected_block_context_cosine ge 0.95`
- `service_recovery`: `service_and_embedding_restored eq True`

## Abort conditions

- Any frozen source or physical model identity differs.
- Fewer than 12 cells complete, a decisive tensor is synthetic, or corrected
  and legacy selected sets are identical in any cell.
- The context bundle lacks any arm/cell, contains an extra tensor, has a
  nonfinite value, or cannot be reopened by the separate scorer.
- Any recomputed cell projection or tensor hash differs from the worker record.
- GPU execution, receipt/provenance binding or service restoration is
  incomplete; port 8081 becoming unhealthy is an immediate abort.
- The result may be negative, but raw evidence must be preserved. R1 is never
  modified and no post-result gate relaxation is allowed.

## Allowed claims

- `SLX08_FIDELITY_FALSE_NEGATIVE_WITH_RETAINED_CONTEXTS_R2`
- `SLX08_FIDELITY_NEGATIVE_WITH_RETAINED_CONTEXTS_R2`

Claims outside these codes are forbidden even if a metric looks favorable.
In particular, neither outcome authorizes TTFT acceleration, runtime
integration, production qualification, quality claims, or causal attribution
solely to computed indices because the real-QKV substitution was also material.
