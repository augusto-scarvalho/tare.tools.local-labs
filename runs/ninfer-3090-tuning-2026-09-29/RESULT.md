# NInfer-3090 0.11 flags for Qwen3.8 GSQ-RCO IQ3_S, and LitJEV on it (2026-09-29)

Node aaaaa, RTX 3090 at 420 W, `ninfer-serve` 0.11.0 (f118551). The production server was yielded
through the shared GPU lease (`image` kind, `/internal/gpu/yield`) for the run. Workload per config:
three ~15K-token prefills (README + docs/performance.md, one-sentence summary) and three greedy code
generations (900 tokens, thinking off). Script: `ninfer_ab.py` (phases 2 and 3 changed only the config
list and the LitJEV request shape). Raw rows: `phase1.json`, `phase2.json`, `phase3-litjev.json`.

| Config | Prefill tok/s | Decode tok/s | Draft acceptance | Greedy text = A | VRAM |
|---|---:|---:|---|---|---:|
| A: production flags, built-in 3090 route profile | 1,578 | 153 | 0.78-0.84 | 3/3 | 18.6 GB |
| B: + `--prefill-cublas --prefill-chunk 2048` | 1,517 | 153 | same | 3/3 | 19.5 GB |
| C: + `--embedding-q4 --lm-head-q6` | refused: the GSQ artifact stores the head and embedding outside row-split Q8_G32 | | | | |
| D2: `--device-profile calibrate` | 1,496 | 153 | same | 3/3 | 18.6 GB |
| E2: `--spec dflash2 --draft-tokens 7 --lm-head-draft` | 1,467 | **180** | 0.75-0.77 | **1/3** | 20.3 GB |

- The fork's recommended 3090 flags were measured on its own artifact; on this one, cuBLAS prefill and
  a fresh calibration are slower than the built-in profile (tuned at 420 W on 2026-09-27), and the
  head/embedding trades do not load. The calibration was written to `~/.cache/ninfer/device-profiles.json`
  and made later `auto` starts slower (A2: 1,479); it was moved out of the cache
  (`~/ninfer-ab/device-profiles.calibrated-2026-09-29.json`), so production keeps the built-in profile.
- DFlash2 decodes 18% faster but changed 2 of 3 greedy generations, so it is not a drop-in: it needs the
  agent cohort before it replaces MTP3.
- Production keeps its flags and adds `--first-token-logprobs` (no effect on generation).

## LitJEV on the resident model

NInfer rejects `logprobs=true`; with `--first-token-logprobs`, a non-streaming request that sets only
`top_logprobs` (max 20) gets the first token's alternatives. Frozen LitJEV prompt, thinking off,
`max_tokens` 1, restricted softmax over the option codes. Same three frozen screens as Laya (2026-09-21)
and hosted Jev (2026-09-28), graded by `tare.tools.os/scripts/qualify_laya_routes.grade`:

| Screen | Qwen3.8 LitJEV | Jev 1.13 | Agreement | Latency (median / max) |
|---|---|---|---:|---|
| Qualification (28) | **SCREEN_PASSED** (order-invariant on both real cases) | NOT_QUALIFIED | 23/28 | 0.51 / 0.60 s |
| Few-shot (32) | SCREEN_PASSED | SCREEN_PASSED | 30/32 | 0.46 / 0.59 s |
| Strategy (6) | SCREEN_PASSED, 6/6 | SCREEN_PASSED | 6/6 | 0.52 / 0.52 s |

All 66 answers valid; no option code was missing from the top 20. This is a screen on an idle,
resident model, not a routing qualification: a probe waits behind a running generation
(`--max-concurrency 1`), and its effect on the agent's cached prefix is not measured yet.
