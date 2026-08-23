# LAB-IMG-002 - SDXL matched local baseline

## Objective

Run the image-generation backlog's comfortable 24 GiB baseline against the
same frozen 768x768 prompt panel used by LAB-IMG-001. This provides a
matched-model comparison before any promotion claim.

## Frozen artifact and runtime

- Model: `stabilityai/stable-diffusion-xl-base-1.0`
- Revision: `462165984030d82259a11f4367a4eed129e94a7b`
- Upstream-declared license: OpenRAIL++
- Exact FP16 UNet, VAE and two text-encoder hashes:
  `OFFICIAL_SOURCE_MANIFEST.json`
- Runtime: the same isolated Diffusers environment and RTX 3090 used by
  LAB-IMG-001; embedding port 8081 remains resident.
- 768x768, 30 steps, FP16, identical prompts and seeds.

## Gates

- `FIT_PASS`: model and all four outputs complete without OOM.
- `MECHANISM_PASS`: 4/4 valid images and same-seed replay byte-identical.
- `SEMANTIC_PASS_BOUNDED`: the unchanged 13-clause VLM gate reaches at least
  85% and at least 2/3 unique cases pass all clauses.
- Compare latency, peak VRAM and clause results to Qwen-Image. This bounded
  panel is not a blind human preference study and cannot independently promote
  a product default.

