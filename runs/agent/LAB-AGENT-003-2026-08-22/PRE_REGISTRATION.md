# LAB-AGENT-003 bounded stress and scale

**Status:** `PREREGISTERED`  
**Date:** 2026-08-22  
**Substrate:** restored canonical Qwen3.8-27B historical Q4_K_XL service, slop.cpp
`b9863-5e7f6271c`, MTP n3, seed 0, temperature 0.

## Matrix

Run one fixed-seed cell at four monotonically increasing levels on each axis:

- distractor tool count: 2, 8, 16, 32; target stock tool fixed near the middle;
- exact parallel weather fan-out: 2, 4, 8, 12 cities;
- completed sequential tool depth: 0, 2, 4, 8; dispatch exactly the next stage/token;
- irrelevant dialogue history: 0, 4, 8, 16 user/assistant turns before the final stock request.

Every response must contain the exact dispatchable tool-call count, names, and arguments. Full raw OpenAI
responses are retained. This is a bounded local capacity curve, not a BFCL score.

## Gate and stopping

Full robust-scale qualification requires 16/16. If a level fails, preserve it and report the largest
contiguous passing level on that axis; still execute the remaining preregistered cells to detect
non-monotonic behavior. Any lowest-level failure is a harness/substrate stop requiring diagnosis before
BigCodeBench. Higher-level failures close the capacity curve without blocking the other axes.

## Fixture-correction amendment

Frozen after the first matrix produced a non-monotonic sequential result (depth 0 failed; depths 2/4/8
passed). Inspection showed the depth-0 prompt never supplied the `root` token required by the schema, while
the validator required it; the model explicitly identified the missing input and called with an empty
token. Mark `results.json` **INVALID / SUPERSEDED AS A GATE**. Add the intended phrase "starting stage 1
with token root" to the shared sequential prompt and rerun the complete 16-cell matrix as
`results.corrected.json`. Preserve both receipts. No scoring rule or other axis changes.
