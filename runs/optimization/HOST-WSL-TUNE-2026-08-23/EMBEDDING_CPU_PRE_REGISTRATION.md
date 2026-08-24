# Embedding GPU-to-CPU offload A/B

Frozen after restoring the documented embedding endpoint and before changing its runtime.

- Baseline: deployed binary, context 32,768, parallel 8, default GPU placement.
- Candidate: same binary/model/API, context 16,384, parallel 8, explicit `--gpu-layers 0`.
- Seven sequential requests after one warmup using fixed input.
- Gates: HTTP success, 768 dimensions, cosine similarity at least 0.999, candidate median latency
  below 250 ms, and at least 256 MiB recovered VRAM while text port 8080 remains healthy.
- Roll back to the exact baseline unit if any gate fails.
