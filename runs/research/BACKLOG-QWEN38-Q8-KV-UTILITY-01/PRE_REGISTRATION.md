# BACKLOG-QWEN38-Q8-KV-UTILITY-01 preregistration

Task: Test Qwen3.8 Q8_0 KV utility noninferiority on a broad fresh math panel
Evidence class: `serving_runtime`

## Hypothesis

Although Q8_0 changed some extracted answers relative to F16 in R2, its actual
task correctness will be noninferior on a fresh 128-task panel: the paired
bootstrap lower 95% bound for Q8-minus-F16 accuracy will exceed -5 percentage
points, with point regression at most 3 points, at least 500 MiB VRAM saving
and at least 95% F16 throughput.

## Frozen inputs

- `runs/research/BACKLOG-QWEN38-KV-PRECISION-02/raw/receipt.json`
- `runs/research/BACKLOG-QWEN38-KV-PRECISION-02/raw/samples.jsonl`
- `runs/research/BACKLOG-QWEN38-KV-PRECISION-02/raw/actual_scores.json`
- `tools/research/run_qwen38_kv_precision.py`
- `tools/research/run_qwen38_kv_precision_r2.py`
- `config/qualified_model_fleet.json`
- `workloads/gsm8k.jsonl`

- Admission: `897c4bfc3820b6376e9124681d1f843b2eee4c13a807f331e65111500da3f139`.
- Q8 R2 receipt: `aba1cc2685f5b74ab01a16937b13d7c063898e9cf2df6ac97bb30564a4074bd7`.
- Q8 R2 samples: `24151e0077e34172f25eafdba5ea24377cb4293ae225bca1b90b85edd10be10a`.
- Q8 R2 scores: `17369a2faaa67899d249772dc96c03f6ef2fb94b0c37b9eea990bcbc29b50b37`.
- Physical KV base runner: `84da3a32cccf309a6cc7106e25a7afb282f9c57acc4dbd3ab2c6c7694a22baf9`.
- Q8 R2 runner: `fdf2d469a652381821af23d7d4612898d9ec65d1f4fae5323ee9e50efa7d8158`.
- Fleet registry: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`.
- GSM8K: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`.

## Command

```powershell
python tools/research/run_qwen38_q8_kv_utility.py --outdir runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-01
```

## Factors

- Two fresh-process arms: F16 then Q8_0, both K and V changed together.
- Fresh panel: GSM8K tasks `32..159`, 128 tasks, disjoint from R2 tasks
  `0..31`; frozen ID-list SHA-256
  `78338489c487181cc63b42f0f26c90e3068c1bb6cef789257ab522258249a786`.
- 256 physical requests total after four discarded warmups per process.
- Model, binary, context 32768, FlashAttention, one slot, MTP depth 3,
  prompt, task order, temperature 0, top-k 1, seed 20260826 and 256-token cap
  remain fixed.
- Primary outcome is paired correctness, not literal extracted-answer equality.
  Bootstrap uses 20,000 deterministic task resamples. VRAM and throughput are
  freshly remeasured; load time is excluded.

## Acceptance gates

- `source_integrity`: `q8_r2_sources_and_artifacts_verified eq True`
- `panel_isolation`: `fresh_panel_disjoint_from_q8_r2 eq True`
- `treatment_identity`: `explicit_cache_controls_verified eq True`
- `request_coverage`: `recorded_requests eq 256`
- `request_integrity`: `successful_response_rate eq 1.0`
- `utility_noninferiority`: `paired_bootstrap_95ci_lower_q8_minus_f16_accuracy gt -0.05`
- `quality_regression`: `f16_minus_q8_accuracy le 0.03`
- `physical_memory_saving`: `vram_saving_mib ge 500`
- `throughput_non_regression`: `q8_vs_f16_throughput_ratio ge 0.95`
- `service_recovery`: `service_and_embedding_restored eq True`

## Abort conditions

- Abort on frozen artifact mismatch, overlap with R2 panel, wrong cache argv,
  occupied temporary port, load failure, three consecutive request failures,
  unhealthy embedding boundary, incomplete panel or service restoration
  failure. Wrong answers and output differences are evidence, not aborts.

## Allowed claims

- `QWEN38_Q8_KV_UTILITY_NONINFERIOR_R1`
- `QWEN38_Q8_KV_UTILITY_NOT_NONINFERIOR_R1`

Claims outside these codes are forbidden even if a metric looks favorable.
