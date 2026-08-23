# LAB-IMG-001 - Qwen-Image on RTX 3090

## Objective

Qualify an open-weight image-generation path that fits a single 24 GiB RTX
3090 and is useful for local product/UI work. This is a separate Diffusers
stack and does not change the llama.cpp deployment.

## Frozen artifact

- Model: `Qwen/Qwen-Image`
- Revision: `75e0b4be04f60ec59a75f475837eced720f823b6`
- Upstream-declared license: Apache-2.0
- Runtime: isolated Python 3.12 environment under `/home/augus/image-venv`
- Precision target: NF4 4-bit transformer and text encoder, BF16 compute
- Hardware: one RTX 3090; the embedding service on port 8081 remains resident

The exact LFS weight identities are written to `OFFICIAL_SOURCE_MANIFEST.json`
and verified after download. Package versions and CUDA identity are recorded in
the run directory.

## Frozen prompt panel

1. Typography: a clean laboratory poster containing exactly `TARE LAB`,
   `BUILD 10161`, and `STATUS READY`.
2. UI: a dark operations dashboard containing cards labelled `QUEUE 7`,
   `GPU 68%`, and `CACHE OK`.
3. Composition: a red cube on the left, a blue sphere on the right, and a green
   triangle above them on a white background.
4. Determinism replay: repeat prompt 3 with the exact same seed and settings.

## Measurements

- Cold model-load latency and peak GPU memory.
- Per-image wall latency and peak GPU memory.
- Output SHA-256, dimensions, and same-seed byte determinism.
- Frozen clause-based semantic inspection using the resident VLM when feasible;
  otherwise retain the images for blind human review.

## Gates

- `FIT_PASS`: all required components load and every image completes without
  OOM while the embedding service remains healthy.
- `MECHANISM_PASS`: 4/4 outputs are non-empty valid images and the same-seed
  replay is byte-identical.
- `SEMANTIC_PASS_BOUNDED`: at least 85% of frozen clauses are recognized and at
  least 2/3 unique prompts pass every clause. This is automated VLM evidence,
  not a substitute for blind human preference.
- Promotion still requires a blind human comparison and a matched baseline;
  this run can qualify the mechanism but cannot make a final product choice.

## Frozen gate-triggered recovery

If the 30-step automated semantic gate fails, rerun only the failing unique
prompts at the upstream-recommended 50 steps with identical seeds and all other
settings unchanged. This tests the highest-ROI lever (step count) before any
new quantization or caching work. A recovered prompt must pass all of its
original clauses; the original result remains retained.
