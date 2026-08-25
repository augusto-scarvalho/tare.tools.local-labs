# ADAPT-00B seven-geometry matrix - result

## Verdict

`SCREEN_COMPLETE`; six arms passed the preregistered mechanics/retention gate,
and DoRA failed with non-finite loss at optimizer step 0. No production geometry
is selected by this loss-only screen.

## Common evidence

All arms used the frozen ADAPT-00A base, revision, hashes, split, seed, 24-step
budget, BF16 precision, and target/protected loss panels. LoRA is the exact
ADAPT-00A result; the other six arms ran sequentially in one isolated GPU
window. Machine-readable receipts are under `raw/` and the combined summary is
`raw/matrix.json`.

## Results

| Method | Trainable params | Target loss improvement | Protected loss change | Peak VRAM | Timed section | Verdict |
|---|---:|---:|---:|---:|---:|---|
| LoRA | 5,411,328 | 38.62% | +0.53% | 4.88 GiB | 16.39 s | `PASS` |
| DoRA | 5,811,264 | not available | not available | not available | stopped at step 0 | `FAIL_NONFINITE_LOSS_STEP_0` |
| LoHa | 10,822,656 | 31.96% | -0.04% | 6.76 GiB | 19.44 s | `PASS` |
| LoKr | 359,040 | **39.74%** | +0.05% | 6.66 GiB | 17.85 s | `PASS` |
| BOFT | 1,505,856 | 36.74% | +0.03% | 7.99 GiB | 84.22 s | `PASS_ON_TORCH_FALLBACK` |
| IA3 | **239,616** | 16.95% | -0.01% | **4.68 GiB** | **13.45 s** | `PASS` |
| Trainable tokens | 2,026,496 | 25.95% | +1.93% | 4.97 GiB | 15.12 s | `PASS` |

Every passing arm had finite loss, a nonzero gradient, target improvement above
1%, protected regression below 15%, exact clean-reload target loss, and peak
allocation below 23 GiB.

## Interpretation

- LoKr is the strongest target-learning and parameter-efficiency candidate in
  this budget: it slightly exceeded LoRA's target improvement with about 6.6%
  as many trainable parameters.
- IA3 is the footprint candidate: smallest parameter count and peak VRAM, but
  also the weakest passing target improvement.
- LoHa preserved the protected panel slightly better than baseline but used the
  most trainable parameters.
- BOFT learned, but its optional PEFT FBD CUDA extension did not compile against
  the frozen PyTorch/CUDA headers. The preregistered torch fallback was about
  five times slower than the other passing arms in this small timed section.
- The trainable-token arm learned the target, but had the largest protected loss
  regression among passing arms.
- DoRA's initial evaluation was finite; the first training-mode example was
  non-finite before backward. The frozen BF16/all-linear arm is rejected without
  a post-result precision or targeting rescue.

Under the preregistered four-objective definition, every passing arm remains on
the Pareto frontier because each retains at least one trade-off advantage. That
is not a tie in behavior; it means the mechanics screen alone cannot choose a
deployment geometry.

## Next gate

Any behavioral follow-on must preregister how finalists are chosen and evaluate
generated exact answer, termination, output length/concision, and an unrelated
protected task. ADAPT-01 trace distillation remains gated behind that selection.
Do not infer 27B/Fable transfer from this 0.8B screen.

## Service restoration

Only `llm-inference.service` was stopped. Embeddings stayed active. After the
matrix, both services were active and the no-thinking Fable canary returned
exactly `adapt00-baseline-restored-ok`. Fan Control and MSI Afterburner were not
touched.
