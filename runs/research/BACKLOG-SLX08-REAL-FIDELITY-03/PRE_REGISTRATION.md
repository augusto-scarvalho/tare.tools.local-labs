# BACKLOG-SLX08-REAL-FIDELITY-03 preregistration

Task: Canonically rescore retained SLX-08 fidelity contexts
Evidence class: `proxy_realization`

## Hypothesis

The R2 all-gates rejection is a numerical-comparator false negative. A
fixed-order float64 cosine computed with `math.fsum` directly from the immutable
R2 context vectors will produce the same canonical evaluation on two separate
process invocations and retain median corrected-context cosine at least 0.95.
This is a retained-evidence rescore, not new physical inference.

## Frozen inputs

- `runs/research/BACKLOG-SLX08-REAL-FIDELITY-02/raw/receipt.json`
- `runs/research/BACKLOG-SLX08-REAL-FIDELITY-02/REVIEW.json`
- `runs/research/BACKLOG-SLX08-REAL-FIDELITY-02/raw/context_vectors.safetensors`
- `runs/research/BACKLOG-SLX08-REAL-FIDELITY-02/raw/samples.jsonl`
- `tools/research/slx08_real_fidelity_worker_r2.py`
- `tools/research/slx08_context_scorer.py`

- Admission specification: 2,809 bytes, SHA-256
  `0b703704ce40191d1c6222778e4a2c52f2945dc160ad51c995f7ffcb763644a6`.
- R2 receipt: 13,981 bytes, SHA-256
  `9715c2fc0c82d7be475268eb3470127ad058e063a3041db34dc195935c1b9143`.
- R2 review: 5,061 bytes, SHA-256
  `e8947002265c7a2d8f33765a25e8a3fe49351a3d70341b0837682c87e8e0edf6`.
- R2 context bundle: 298,304 bytes, SHA-256
  `859ea9e3088de4e1f354a51a3c5502fd845ac1289dd0ad7b83d8c4f35b76cc58`.
- R2 samples: 15,453 bytes, SHA-256
  `e5e8bb64db6e3a1a680bc266601122caaaf0cc8f4d1b7903110aa8aa2f94017e`.
- R2 GPU worker: 7,063 bytes, SHA-256
  `5c4f966e95baf572e4e1942098b80a6be0a3d9c53de636ca43df37da41fd91ed`.
- R2 cross-device scorer: 4,220 bytes, SHA-256
  `37d5d808e317494293bcbb3933ec25107a72526dcb238d8f280ac9bf2ca8e995`.
- The new canonical scorer, runner and fixtures are frozen by the
  `PREREGISTERED -> IMPLEMENTED` transition before any rescore.

## Command

```powershell
python tools/research/run_slx08_canonical_rescore_r3.py --outdir runs/research/BACKLOG-SLX08-REAL-FIDELITY-03
```

## Factors

- Source population: the exact 36 retained R2 float32 tensors, arranged as 12
  cells by dense/corrected/legacy arms. No task, layer, vector or arm selection
  is allowed.
- Tensor identity: exact safetensors key set, shape `[8, 1, 256]`, dtype
  float32, per-tensor SHA-256 and bundle SHA-256.
- Canonical cosine: flatten in safetensors row-major order, convert each scalar
  to Python float, compute dot product and both squared norms with `math.fsum`,
  then divide by `sqrt(left_norm * right_norm)`. No device reduction, tolerance
  or comparison with the R2 CUDA scalar enters the fidelity gate.
- Repetition control: launch the frozen scorer twice in separate WSL processes;
  require canonical JSON equality of all 24 cell cosines, medians, hashes and
  counts. Output filenames are excluded from the canonical payload.
- Service control: read 8080/8081 process and health state before and after;
  never stop, restart, switch or issue inference to either service.
- Primary metric remains the preregistered R1/R2 median corrected-context cosine
  threshold `>=0.95`. The legacy median is descriptive. No new sample or GPU
  forward can rescue a failure.

## Acceptance gates

- `source_receipt_binding`: `source_receipt_bound eq True`
- `bundle_identity`: `bundle_sha256_match eq True`
- `bundle_key_contract`: `retained_context_tensors eq 36`
- `cell_coverage`: `retained_context_cells eq 12`
- `tensor_identity`: `tensor_hash_match_rate eq 1.0`
- `finite_contexts`: `nonfinite_values eq 0`
- `canonical_repeat`: `canonical_repeat_match eq True`
- `fidelity`: `canonical_median_selected_block_context_cosine ge 0.95`
- `service_untouched`: `serving_process_unchanged eq True`

## Abort conditions

- Any frozen source hash, receipt fingerprint or implementation hash differs.
- The bundle is not exactly the frozen 298,304 bytes/SHA, lacks any of 36 keys,
  has an extra key, wrong dtype/shape, nonfinite value or tensor-hash mismatch.
- Either scorer invocation fails or their canonical payloads differ at any
  field. There is no tolerance between canonical repetitions.
- The runner touches the inference service, observes a PID/argv/restart/health
  change, or cannot bind complete provenance.
- Preserve a negative result and stop. R1/R2 evidence is immutable; thresholds
  and reduction rules cannot change after this preregistration.

## Allowed claims

- `SLX08_FIDELITY_FALSE_NEGATIVE_CANONICAL_RESCORE_R3`
- `SLX08_FIDELITY_CANONICAL_RESCORE_NEGATIVE_R3`

Claims outside these codes are forbidden even if a metric looks favorable.
The strongest legal state is `VERIFIED` because this remains an offline
`proxy_realization`. No result authorizes promotion, TTFT, integrated runtime,
production, language-model quality, new inference or causal attribution solely
to computed indices.
