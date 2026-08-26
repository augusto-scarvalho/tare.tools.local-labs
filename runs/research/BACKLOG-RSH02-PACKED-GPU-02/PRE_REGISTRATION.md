# BACKLOG-RSH02-PACKED-GPU-02 preregistration

Task: Requalify RSH-02 physical block-Huffman after Triton namespace fix
Evidence class: `mechanism_research`

## Hypothesis

With only the Triton namespace binding corrected, a physical restart-block Huffman decoder over real Qwen INT4 weight symbols will retain the historical RSH-02 rejection unless exact physical compression, at least 100 GB/s input throughput and at most 2x paired INT4 latency all pass. If every gate passes, the historical rejection is a false negative on this frozen panel.

## Frozen inputs

- `runs/research/BACKLOG-RSH02-PACKED-GPU-01/ABORTED.md`
- `runs/research/RSH-02-HYPERQUANT-2026-08-25/PRE_REGISTRATION.md`
- `runs/research/RSH-02-HYPERQUANT-2026-08-25/RESULT.md`
- `runs/research/RSH-02-HYPERQUANT-2026-08-25/raw/receipt.json`
- `tools/probes/rsh02_hyperquant_entropy_coding.py`
- `runs/research/BACKLOG-NEGATIVE-KV-REAL-SCREEN-02/raw/model_hash.json`

- Compiler-abort record SHA-256: `f6f59c8e18b219ecd6a28d14a0849de026627a509896ec486be0ac60545cacca`.
- Historical preregistration/result/receipt/probe SHA-256: `171a5de2ac6964a963152d8b2f682e37de8a70a5a638fe86dab10f239985cc8b`, `6aa4a6b9484409fbae285a0b6c9c4effd5f95d25d99c68f98992f89fd8b871d7`, `5a7294dacae8de2133e63bb9b486e769a6883987fb841d28efab9d53956b29fd`, `60ab8218491378d1f731b3987395e00683fb3dd695a343f9fea4cc19f84f15f2`.
- Model hash ledger binds the sole safetensors shard to `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`.

## Command

```powershell
python tools/research/run_rsh02_packed_gpu_r2.py --outdir runs/research/BACKLOG-RSH02-PACKED-GPU-02
```

## Factors

- Identical frozen factors to `BACKLOG-RSH02-PACKED-GPU-01`: four layer-0/1 Qwen gate/up matrices, blockwise signed INT4 quantization, empirical canonical Huffman, 128-symbol physical restart blocks, uint32 offsets and 12-bit lookup table.
- Paired physical signed-INT4 Triton unpack control over the exact same symbol ledger.
- Exact CPU and GPU round-trip mandatory; storage counts bitstream, offsets and codebook.
- At least 25 warmups; five batches of 100 CUDA-event-timed iterations per arm; medians decide throughput and penalty.
- Sole implementation change from the blocked predecessor: expose `triton.language` in the JIT function's module-global namespace.

## Acceptance gates

- `real_source`: `actual_model_weight_elements ge 14000000`
- `physical_packing`: `physical_packed_bitstream eq True`
- `exact_decode`: `exact_roundtrip_rate eq 1.0`
- `compression`: `physical_bits_per_element le 3.0`
- `throughput`: `decoder_input_throughput_gbs ge 100.0`
- `penalty`: `latency_penalty_vs_int4 le 2.0`

## Abort conditions

- Any source/model mismatch, fewer than 14 million real elements, service health loss, decode mismatch, kernel/compiler error, OOM or nonfinite timing.
- Any post-hoc factor, threshold, block-size, warmup or timing-count change.

## Allowed claims

- `RSH02_PACKED_GPU_QUALIFIED_R2`
- `RSH02_NEGATIVE_RETAINED_R2`
- `RSH02_FALSE_NEGATIVE_CONFIRMED_R2`

Claims outside these codes are forbidden even if a metric looks favorable.

This is a physical codec-kernel screen, not serving or model-quality qualification.
