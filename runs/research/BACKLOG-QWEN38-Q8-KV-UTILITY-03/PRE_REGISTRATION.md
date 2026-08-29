# BACKLOG-QWEN38-Q8-KV-UTILITY-03 preregistration

Task: Semantic rescore of frozen Qwen3.8 F16 versus Q8 KV utility outputs
Evidence class: `serving_runtime`

## Hypothesis

Under a conservative semantic rescore of all retained responses, Q8 KV remains within the preregistered paired quality margin while preserving the already-measured physical memory and throughput gates.

## Frozen inputs

- `runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02/raw/receipt.json`
- `runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02/raw/samples.jsonl`
- `runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02/raw/actual_scores.json`
- `runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02/REVIEW.json`
- `workloads/gsm8k.jsonl`
- `tools/analysis/final_numeric_answer_v2.py`
- `tests/fixtures/final_numeric_answer_v2_cases.json`
- `tests/test_final_numeric_answer_v2.py`

Every listed input is frozen by SHA-256 in `tools/research/run_qwen38_q8_kv_utility_r3.py`; any mismatch aborts before writing evidence.

## Command

```powershell
python tools/research/run_qwen38_q8_kv_utility_r3.py
```

## Factors

Offline, gold-blind rescore only: 128 paired tasks per F16/Q8 arm (256 retained outputs), no inference or service mutation. Physical VRAM, throughput, treatment identity, and recovery metrics are imported only from the hash-bound R2 packet. Paired bootstrap: 20,000 replicates, seed 2026082812.

## Acceptance gates

- `source_integrity`: `q8_r2_sources_verified eq True`
- `fixture_validation`: `external_fixture_pass_rate eq 1.0`
- `retained_regressions`: `retained_regression_pass_rate eq 1.0`
- `request_coverage`: `rescored_requests eq 256`
- `utility_noninferiority`: `paired_bootstrap_95ci_lower_q8_minus_f16_accuracy gt -0.05`
- `quality_regression`: `f16_minus_q8_accuracy le 0.03`
- `physical_memory_saving`: `vram_saving_mib ge 500`
- `throughput_non_regression`: `q8_vs_f16_throughput_ratio ge 0.95`
- `service_recovery`: `service_and_embedding_restored eq True`
- `scorer_blinding`: `scorer_does_not_receive_gold eq True`

## Abort conditions

Abort on input hash mismatch, non-empty raw directory, duplicate/missing pair, fixture/regression failure, non-gold-blind scorer signature, or incomplete provenance. A failed scientific gate is recorded without rerunning inference.

## Allowed claims

- `QWEN38_Q8_KV_UTILITY_NONINFERIOR_R3`
- `QWEN38_Q8_KV_UTILITY_NOT_NONINFERIOR_R3`

Claims outside these codes are forbidden even if a metric looks favorable.
