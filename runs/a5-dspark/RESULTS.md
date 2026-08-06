# A5 — DSpark / EAGLE-family draft probe (CLOSED NEGATIVE, 2026-08-06)

Built + measured. Cherry-picked llama.cpp PR #25173 (DSpark) onto probe branch `dspark-probe`
(commit `4c9d7675a`, clean; binary lists `draft-dspark`). Question: can a trained EAGLE-family
draft (DFlash/DSpark) beat our native MTP head anywhere on this fleet?

## Verdict: NO — DSpark is dead for our fleet. Native MTP wins everywhere it exists.

| target | arch | DSpark result | native MTP (incumbent) |
|--------|------|---------------|------------------------|
| qwen3-vl-8b | qwen3vl | **CRASH** on decode: `llama-graph.cpp:1247 GGML_ASSERT(t_layer_inp[il] != nullptr)` — the vision graph doesn't expose the target-hidden-state tensors the dflash draft injects (n_extract=5). Draft loads (vocab/dims OK) but can't hook. | (undrafted today) |
| gemma-4-12b-vision | gemma4v | public GGUF (ankk98) is **format-incompatible** (ft-dspark fork arch `dspark` ≠ upstream arch `dflash`); converter is Qwen-only | MTP QAT assistant: nospec 83 → **228 t/s (+175%), accept 0.939** (reasoning text) |
| **fable-tc l1.0 (dense 27B)** | qwen35 | **loads + runs but −81.6%**: nospec 43.2 → dspark **7.9 t/s**, accept **0.212** (satgeze/Qwen3.6-27B-DSpark, arch dflash, upstream fmt) | native nextn: **55.0 t/s (+27%)**, accept **0.444** |
| MoE 35B-A3B | qwen35moe | no community draft exists | native nextn: +53–73%, accept 0.99 |

## Why DSpark loses on the dense (where it even loads)
1. **Accept collapse (0.212 vs MTP 0.444):** the satgeze draft is trained on BASE Qwen3.6-27B hidden
   states; fable-tc is a MERGE (`fable + λ(TC−base)`) → shifted hidden states → the base-trained draft
   mispredicts. (Same mismatch the native MTP tolerates better because it was carried through the merge.)
2. **Draft cost dominates:** the DSpark drafter is 3.73GB **bf16** with `n_extract=5` (reads 5 target
   layers/step) at block_size 7. On this **bandwidth-bound** 3090 (dense decode = 83% of weight-BW), a
   heavy bf16 draft per step destroys throughput. The cheap in-model nextn head (1 extra layer) has
   near-zero draft cost — which is exactly why MTP wins.

## Structural boundary established
EAGLE-family drafts (DFlash/DSpark/Eagle3) are **target-feature-conditioned** — they inject the target's
internal hidden-states. → (a) incompatible with VLM target graphs (crash); (b) even on text targets, a
heavy external draft loses to the free in-model MTP head on a batch-1 bandwidth-bound box. Standalone
drafts (native MTP, ngram) remain the right tools here. The cherry-pick is a banked capability (would
serve a future *vanilla* Qwen3 text model), inert on the default path; deploy stays on `lifecycle` MTP.

Evidence: A/B drivers `ops/a5_dspark_{ab,qwen,dense}.py`; MTP baselines `runs/a1-mtp-depth/a5_fable.csv`.
