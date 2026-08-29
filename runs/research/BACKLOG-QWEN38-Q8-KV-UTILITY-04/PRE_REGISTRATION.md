# BACKLOG-QWEN38-Q8-KV-UTILITY-04 preregistration

Task: Final Qwen3.8 Q8 KV utility aggregation from promoted blind labels
Evidence class: `serving_runtime`

## Hypothesis

With the independently promoted blind labelset, Q8 remains within the -5 percentage-point paired noninferiority margin while retaining the hash-bound physical memory and throughput gates.

## Frozen inputs

- `runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-03/raw/receipt.json`
- `runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-03/raw/sealed_scored_labels.jsonl`
- `runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-03/REVIEW.json`
- `runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-03/raw/receipt.json`
- `runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-03/raw/actual_scores.json`
- `runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-03/REVIEW.json`

The promoted label receipt/review/scored rows and Q8 R3 receipt/review/physical metrics are frozen by SHA-256 in `tools/research/run_qwen38_q8_kv_utility_r4.py`.

## Command

```powershell
python tools/research/run_qwen38_q8_kv_utility_r4.py
```

## Factors

Offline aggregation only: 128 paired F16/Q8 rows, 20,000 paired-bootstrap replicates, seed 2026082815. Physical metrics are imported without mutation from the audited R3 execution. No inference, GPU or service mutation.

## Acceptance gates

- `source_integrity`: `promoted_blind_labels_and_physical_sources_verified eq True`
- `request_coverage`: `q8_labeled_rows eq 256`
- `utility_noninferiority`: `paired_bootstrap_95ci_lower_q8_minus_f16_accuracy gt -0.05`
- `quality_regression`: `f16_minus_q8_accuracy le 0.03`
- `physical_memory_saving`: `vram_saving_mib ge 500`
- `throughput_non_regression`: `q8_vs_f16_throughput_ratio ge 0.95`
- `service_recovery`: `service_and_embedding_restored eq True`

## Abort conditions

Abort on source mismatch, non-promoted label receipt, missing/duplicate pair, any row outside Q8 source, physical-metric mismatch, or incomplete provenance. A scientific gate failure is recorded and ends this rescore family.

## Allowed claims

- `QWEN38_Q8_KV_UTILITY_NONINFERIOR_R4`
- `QWEN38_Q8_KV_UTILITY_NOT_NONINFERIOR_R4`

Claims outside these codes are forbidden even if a metric looks favorable.
