# BACKLOG-RSH02-PACKED-GPU-01 preregistration

Task: Requalify RSH-02 with a physical block-Huffman GPU decoder on real Qwen weights
Evidence class: `mechanism_research`

## Hypothesis

A physical restart-block Huffman decoder over real Qwen INT4 weight symbols will reproduce the historical RSH-02 rejection: even if exact decoding and physical compression at or below 3.0 bits/element pass, decoder input throughput will remain below 100 GB/s or latency will exceed 2x a paired physical INT4 unpack kernel. If every mandatory gate passes, the historical rejection is a false negative on this frozen hardware/tensor panel.

## Frozen inputs

- `runs/research/RSH-02-HYPERQUANT-2026-08-25/PRE_REGISTRATION.md`
- `runs/research/RSH-02-HYPERQUANT-2026-08-25/RESULT.md`
- `runs/research/RSH-02-HYPERQUANT-2026-08-25/raw/receipt.json`
- `tools/probes/rsh02_hyperquant_entropy_coding.py`
- `runs/research/BACKLOG-NEGATIVE-KV-REAL-SCREEN-02/raw/model_hash.json`

- Historical preregistration SHA-256: `171a5de2ac6964a963152d8b2f682e37de8a70a5a638fe86dab10f239985cc8b`.
- Historical result SHA-256: `6aa4a6b9484409fbae285a0b6c9c4effd5f95d25d99c68f98992f89fd8b871d7`.
- Historical receipt SHA-256: `5a7294dacae8de2133e63bb9b486e769a6883987fb841d28efab9d53956b29fd`.
- Historical emulation probe SHA-256: `60ab8218491378d1f731b3987395e00683fb3dd695a343f9fea4cc19f84f15f2`.
- Frozen model-hash ledger binds `model.safetensors-00001-of-00001.safetensors` to SHA-256 `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`.
- Model: `/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe`.

## Command

```powershell
python tools/research/run_rsh02_packed_gpu.py --outdir runs/research/BACKLOG-RSH02-PACKED-GPU-01
```

## Factors

- Frozen real tensors: `gate_proj` and `up_proj` weights from Qwen layers 0 and 1, totaling at least 14 million physical parameters.
- Quantize each consecutive 64-value block symmetrically to signed INT4 symbols `[-7,7]`; this symbol ledger is shared by both arms.
- Treatment: empirical canonical Huffman code, bit-packed bytes, 128-symbol restart blocks with physical uint32 offsets, and a Triton GPU decoder using a prefix lookup table. Count bitstream, offsets and codebook bytes in physical bits/element.
- Control: two signed INT4 nibbles per physical byte and a paired Triton GPU unpack kernel.
- Exact equality against all original signed symbols is mandatory; an inaccurate decoder aborts rather than reporting speed.
- Warm up each compiled kernel at least 25 iterations, then time 100 iterations with CUDA events and synchronization. Report median of five independently timed batches for each arm.
- Throughput uses physical compressed input bytes divided by kernel time; penalty uses Huffman latency divided by INT4 latency.
- Record tensor keys/shapes/hashes, symbol histogram/hash, code lengths, packed buffer/hash, offsets/hash, decoded output/hash, GPU identity, compiler/runtime versions and paired timings.

## Acceptance gates

- `real_source`: `actual_model_weight_elements ge 14000000`
- `physical_packing`: `physical_packed_bitstream eq True`
- `exact_decode`: `exact_roundtrip_rate eq 1.0`
- `compression`: `physical_bits_per_element le 3.0`
- `throughput`: `decoder_input_throughput_gbs ge 100.0`
- `penalty`: `latency_penalty_vs_int4 le 2.0`

## Abort conditions

- Frozen source or model ledger mismatch, missing Triton/CUDA/RTX 3090, fewer than 14 million real elements, or service health loss.
- Packed bytes or offsets are synthesized without encoding the frozen symbol stream.
- Any decode mismatch, out-of-range bit read, unstable kernel, OOM or nonfinite timing.
- Thresholds, block size, warmup or timing counts change after results are observed.

## Allowed claims

- `RSH02_PACKED_GPU_QUALIFIED_R1`
- `RSH02_NEGATIVE_RETAINED_R1`
- `RSH02_FALSE_NEGATIVE_CONFIRMED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.

This is a physical codec-kernel screen, not serving or model-quality qualification.
