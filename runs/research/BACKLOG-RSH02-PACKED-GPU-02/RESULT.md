# BACKLOG-RSH02-PACKED-GPU-02 result

## Verdict

`RSH02_NEGATIVE_RETAINED_R2` pending independent AGY review.

The successor encoded `14680064` real Qwen weight symbols into a physical Huffman bitstream and decoded them exactly on the RTX 3090. Physical storage was `3.7847` bits/element including restart offsets and lookup table. Huffman latency was `1.3184` ms and input throughput `5.268` GB/s, versus `0.0434` ms for physical INT4 (`30.408x`).

Failed gates: `compression, throughput, penalty`. Claim scope is limited to this physical codec-kernel screen.
