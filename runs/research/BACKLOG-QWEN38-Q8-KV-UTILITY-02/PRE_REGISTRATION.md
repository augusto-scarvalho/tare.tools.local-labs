# BACKLOG-QWEN38-Q8-KV-UTILITY-02 preregistration

Task: Repeat Qwen3.8 Q8_0 KV utility without mutable handoff provenance
Evidence class: `serving_runtime`

## Hypothesis

On the same frozen, disjoint 128-task GSM8K panel, Q8_0 KV cache remains
noninferior to F16: the paired-bootstrap lower 95% bound for Q8-minus-F16
accuracy exceeds -5 percentage points, point regression is at most 3 points,
VRAM saving is at least 500 MiB, and throughput is at least 95% of F16.

## Frozen inputs

- `runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-01/raw/receipt.json`
- `runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-01/raw/samples.jsonl`
- `runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-01/raw/actual_scores.json`
- `tools/research/run_qwen38_q8_kv_utility.py`
- `tools/research/run_qwen38_kv_precision.py`
- `config/qualified_model_fleet.json`
- `workloads/gsm8k.jsonl`

- Admission: `9f833f08530d4060caab73e706c3fb06cc80a74c0cdffcc6b325ba1f5a23ecba`.
- R1 receipt: `3a7e0eaabc678e6514e13875906005d89789adccca1edd44b6b880e7d72dbb24`.
- R1 samples: `f072f9e297a5ac1f1681a4c849c163ee99956536c26a057fa1feaa40dc5036ef`.
- R1 scores: `9aed2ce35d30c5e9b35b4f86848eb2d3a7d6e47bfebc3a9063e12563a25e7a4f`.
- R1 utility runner: `e1e0c30a201dc87b0d924067d59b5fe17fa6b81049c696cd1ad8e723e2de58d6`.
- Physical KV runner: `84da3a32cccf309a6cc7106e25a7afb282f9c57acc4dbd3ab2c6c7694a22baf9`.
- Fleet registry: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`.
- GSM8K: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`.
- Mutable handoff documents are explicitly excluded from scientific identity.

## Command

```powershell
python tools/research/run_qwen38_q8_kv_utility_r2.py --outdir runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02
```

## Factors

- Two fresh processes in fixed order: F16 then Q8_0, changing K and V together.
- GSM8K tasks 32..159, 128 paired tasks and 256 physical requests.
- Four discarded warmups per arm; temperature 0, top-k 1, seed 20260826,
  prompt cache off, one slot and 256-token cap.
- Primary outcome is paired task correctness. Literal output parity remains
  descriptive only.

## Acceptance gates

- `source_integrity`: `q8_sources_and_artifacts_verified eq True`
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

Abort on immutable input mismatch, panel overlap, wrong cache argv, occupied
temporary endpoint, load failure, three consecutive request failures, unhealthy
embedding boundary, incomplete panel, or service restoration failure. Wrong
answers and output differences are retained as evidence.

## Allowed claims

- `QWEN38_Q8_KV_UTILITY_NONINFERIOR_R2`
- `QWEN38_Q8_KV_UTILITY_NOT_NONINFERIOR_R2`

Claims outside these codes are forbidden even if a metric looks favorable.
