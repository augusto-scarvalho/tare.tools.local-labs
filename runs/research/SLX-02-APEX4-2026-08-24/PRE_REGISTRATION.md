# SLX-02 APEX4 RTX 3090 qualification - pre-registration

## Frozen identities

- Source: `APEX4-W4A4/APEX4-W4A4`
- Source commit: `6ffc9ea07b7da8fc0b0d8937ccd0012878ae337d`
- Candidate: `APEX4-W4A4/Qwen2.5-7b-g128`
- Candidate revision: `3be1cefb76f45c734ad8f4102394cadd5cf6a691`
- Hardware: NVIDIA RTX 3090, compute capability 8.6, driver 591.86.
- Build target: `sm_86` only.

## Question

Is the public APEX4 accuracy package reproducibly buildable and executable on
this RTX 3090, and does its released Qwen2.5-7B g128 checkpoint reproduce a
finite WikiText-2 perplexity close to the repository's stated 7.87 result?

This run cannot answer the end-to-end throughput question unless the separate
vLLM performance package cited by the authors is public and runnable. Kernel
timings from the accuracy package are not an end-to-end substitute.

## Sequential phases

1. Build the released extension with Python 3.11, PyTorch 2.5.1+cu124,
   CUDA 12.4, and `TORCH_CUDA_ARCH_LIST=8.6`.
2. Run the repository's documented `test_groups` correctness test.
3. Download only the public Qwen2.5-7B uniform-g128 checkpoint at the frozen
   revision.
4. Stop only `llm-inference.service`, leaving the embedding service untouched.
5. Run WikiText-2 perplexity with the released evaluation command and seed 42.
6. Restore and verify the Fable inference baseline immediately after the run.

## Gates

- Build must succeed for `sm_86` with the frozen source and toolchain.
- The documented group-kernel test must pass its numerical assertions.
- Evaluation must exercise the W4A4 model path and return finite perplexity.
- Accuracy reproduction passes when absolute PPL difference from 7.87 is at
  most 0.25. Larger difference is preserved, not tuned away.
- No `slop.cpp` port study opens without public, reproducible evidence of at
  least 20% relevant end-to-end gain, or at least 10% plus a material capacity
  gain. Kernel-only evidence cannot satisfy this gate.

## Fail-fast rules

- Stop on source/build incompatibility, non-finite output, OOM, or checkpoint
  identity mismatch.
- Do not download the other three checkpoints unless the first candidate
  passes and a remaining causal question requires another arm.
- Do not alter the production service configuration or default.
