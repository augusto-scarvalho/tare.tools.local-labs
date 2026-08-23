# LAB-IMG-001 result

## Decision: FIT PASS / MECHANISM PASS / QUALITY HOLD

The exact official Qwen-Image revision was verified across all 14 safetensors
files (57,699,249,390 bytes). An isolated Diffusers 0.40.0 / PyTorch
2.13.0+cu132 stack loaded NF4 transformer and text encoder with BF16 compute on
the RTX 3090 while the embedding endpoint on port 8081 remained healthy.

At 768x768 and 30 steps, cold load took 29.88 s and peaked at 18,280 MiB total
GPU memory used. Generation peaked at 13,076 MiB and took 49.63-53.35 s per
image. All 4/4 PNGs were valid. The identical-seed composition replay had the
same SHA-256, so the bounded deterministic mechanism gate passed.

## Semantic result

The resident Gemma-4-12B Vision evaluator scored 10/13 frozen clauses (76.92%)
and only 1/3 unique cases passed every clause, below the 85% / 2-of-3 gate:

| Prompt | Clauses | Observation |
|---|---:|---|
| typography | 3/3 | rendered `TARE LAB`, `BUILD 10161`, `STATUS READY` exactly |
| dashboard | 2/3 | queue and GPU were correct; `CACHE` became `CACSE` |
| composition | 5/7 | layout/colors were correct; cube/sphere became square/circle |

The gate-triggered 50-step recovery did not help. Latency rose to 79.63-81.90 s
for the two failing prompts, VRAM was unchanged, and their panel score fell from
7/10 to 6/10 (`CACHE` remained misspelled; 3D objects remained flat shapes).
Therefore step reduction is retained: 30 steps dominates 50 on this panel.

Qwen-Image is a viable open-weight 3090 mechanism with notably strong simple
text rendering, but it is not promoted as a general image-product default from
this small quality panel. Blind human preference and a matched baseline remain
required; LAB-IMG-002 provides the latter.
