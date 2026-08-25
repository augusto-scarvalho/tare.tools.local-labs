# SLX-05D CUDA Graph replay result

Verdict: `QUALIFIED_CUDA_GRAPH_REPLAY`

CUDA Graph capture succeeded in all five preregistered cells with exact logit
parity. Batch-1 wall-time speedups were 2.68x at context 128, 2.56x at 512 and
2.28x at 2048; the median was 2.56x. Batch 2 and 4 at context 512 measured
2.42x and 2.11x respectively. Every observation restored the same hybrid cache
snapshot outside its timed region.

Provenance is complete. Receipt fingerprint:
`11ed8746b7d0cc05273456538305d25f6cebcfe7980a11a55b0852eaefaa1538`.
This result does not isolate driver-launch overhead and does not establish a
persistent-megakernel ceiling. It supersedes the invalid SLX-05 interpretation;
SLX-05B and SLX-05C remain preserved implementation failures.

