# BACKLOG-QWEN38-KV-PRECISION-02 preregistration

Task: Qualify Qwen3.8 Q8_0 KV cache as the bounded compromise after Q4_0 divergence
Evidence class: `serving_runtime`

## Hypothesis

On the frozen Qwen3.8 runtime, physical Q8_0 K/V cache will recover at least
95% extracted-answer parity with F16 while saving at least 500 MiB of resident
GPU memory, losing at most one of 32 GSM8K answers and retaining at least 95%
of F16 decode throughput. Both repeated arms must be exactly repeatable. Any
gate failure rejects this bounded configuration; this family stops after R2.

## Frozen inputs

- `runs/research/BACKLOG-QWEN38-KV-PRECISION-01/raw/receipt.json`
- `runs/research/BACKLOG-QWEN38-KV-PRECISION-01/raw/samples.jsonl`
- `tools/research/run_qwen38_kv_precision.py`
- `config/qualified_model_fleet.json`
- `workloads/gsm8k.jsonl`

- Admission: `d93073fa3d6cdebacb02c651bd4c73a4447bdfdcde2c413b7dafa3aa5f754e90`.
- R1 receipt: `9e068c702f431a8fc6cc34a393f0daee6ac5a02d3329d10233747f451f7a2652`.
- R1 samples: `bec190f0b731b5931cd14bfcaf4808828ad50449fc3f209827e769b0102c4a5d`.
- R1 runner: `84da3a32cccf309a6cc7106e25a7afb282f9c57acc4dbd3ab2c6c7694a22baf9`.
- Fleet registry: `042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82`.
- GSM8K source: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`.
- Runtime binary and model identities are reverified against the immutable
  registry/infra ledger before service maintenance.

## Command

```powershell
python tools/research/run_qwen38_kv_precision_r2.py --outdir runs/research/BACKLOG-QWEN38-KV-PRECISION-02
```

## Factors

- Four fresh-process blocks in `F16, Q8_0, Q8_0, F16` order.
- Each block runs four discarded warmups plus the same first 32 frozen GSM8K
  tasks: 128 recorded responses and 64 cross-precision task pairs.
- Binary, model, prompt order, context 32768, MTP depth 3, temperature 0,
  top-k 1, seed 20260826 and 256-token cap remain fixed.
- F16/F16 and Q8/Q8 repeat parity are measured separately from Q8/F16 parity.
- VRAM is sampled after health and warmup; throughput uses server timing
  counters. Model-load time is excluded.

## Acceptance gates

- `treatment_identity`: `explicit_cache_controls_verified eq True`
- `balanced_crossover`: `valid_abba_blocks eq 4`
- `request_integrity`: `successful_response_rate eq 1.0`
- `f16_repeatability`: `f16_repeat_parity_rate eq 1.0`
- `q8_repeatability`: `q8_repeat_parity_rate eq 1.0`
- `semantic_parity`: `q8_f16_extracted_answer_parity_rate ge 0.95`
- `quality_non_regression`: `q8_accuracy_regression le 0.03125`
- `physical_memory_saving`: `vram_saving_mib ge 500`
- `throughput_non_regression`: `q8_vs_f16_throughput_ratio ge 0.95`
- `service_recovery`: `service_and_embedding_restored eq True`

## Abort conditions

- Abort on any frozen hash or physical artifact mismatch, wrong cache argv,
  occupied temporary port, load failure, three consecutive request failures,
  embedding failure at a block boundary, or inability to restore the original
  gateway route and persistent service.
- Wrong answers and cross-arm differences are evidence and do not abort.

## Allowed claims

- `QWEN38_Q8_KV_PHYSICALLY_QUALIFIED_R2`
- `QWEN38_Q8_KV_PHYSICALLY_REJECTED_R2`

Claims outside these codes are forbidden even if a metric looks favorable.
