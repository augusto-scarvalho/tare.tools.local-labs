# STATUS — evidence (auto-generated)

> Generated 2026-07-31 by `model_lifecycle.reports.status` from **lifecycle.db** (546 runs). **Numbers only** — the interpretation lives in the authored `STATUS.md`. Do not hand-edit this file; regenerate it.

Every delta is `a − b`, paired by `(round, ncmoe)`, median (not mean). `sign_p` is the exact two-sided sign test (floor 0.031 at n=6); the CI is the seeded percentile bootstrap of the paired delta; `δ` is Cliff's delta.

**Noise floor** (median |%| of the null A/B's prefill, true Δ = 0 by construction): **2.33%**. A prefill delta at or below this is flagged ⚠ and is not evidence, however tidy its median.

## NOISE FLOOR — same binary, same env; true delta is zero by construction
`ingest:ab-null-qwen36-35b` — 12 records

| comparison | metric | n | Δ median | sign_p | boot CI95 | δ |
|---|---|---|---|---|---|---|
| `same − base` | prompt_tps | 6 | +0.61 (+0.29%) | 1.0000 | [-3.1, +6.4] | +0.17 ⚠ |
| `same − base` | gen_tps | 6 | -0.29 (-0.67%) | 0.6875 | [-0.5, +0.5] | -0.06 ⚠ |

## PINNING — qwen36-35b (256 experts)
`ingest:ab-pinning-qwen36-35b` — 18 records

| comparison | metric | n | Δ median | sign_p | boot CI95 | δ |
|---|---|---|---|---|---|---|
| `pin − base` | prompt_tps | 6 | +218.78 (+104.89%) | 0.0312 | [+215.5, +223.4] | +1.00 |
| `pinpf − pin` | prompt_tps | 6 | -94.64 (-22.14%) | 0.0312 | [-98.6, -92.1] | -1.00 |
| `pinpf − base` | prompt_tps | 6 | +125.06 (+60.24%) | 0.0312 | [+120.2, +127.0] | +1.00 |
| `pin − base` | gen_tps | 6 | -0.07 (-0.15%) | 0.6875 | [-0.2, +0.1] | +0.06 ⚠ |

## PINNING — qwen3-30b (128 experts, independent geometry)
`ingest:ab-pinning-qwen3-30b` — 18 records

| comparison | metric | n | Δ median | sign_p | boot CI95 | δ |
|---|---|---|---|---|---|---|
| `pin − base` | prompt_tps | 6 | +247.61 (+123.32%) | 0.0312 | [+242.7, +252.0] | +1.00 |
| `pinpf − pin` | prompt_tps | 6 | -50.61 (-11.30%) | 0.0312 | [-51.4, -47.9] | -1.00 |
| `pin − base` | gen_tps | 6 | +0.07 (+0.24%) | 0.6875 | [-0.1, +0.3] | +0.17 ⚠ |

## PINNING — gpt-oss-20b (32 experts, independent geometry)
`ingest:ab-pinning-gpt-oss-20b` — 18 records

| comparison | metric | n | Δ median | sign_p | boot CI95 | δ |
|---|---|---|---|---|---|---|
| `pin − base` | prompt_tps | 6 | +215.84 (+114.59%) | 0.0312 | [+209.6, +224.3] | +1.00 |
| `pinpf − pin` | prompt_tps | 6 | -34.88 (-8.62%) | 0.0312 | [-40.4, -33.2] | -1.00 |
| `pin − base` | gen_tps | 6 | +0.01 (+0.04%) | 1.0000 | [-0.5, +0.1] | +0.22 ⚠ |

## GENPIN — does pinning move GENERATION? (35B, near-resident)
`ingest:ab-genpin-qwen36-35b` — 12 records

| comparison | metric | n | Δ median | sign_p | boot CI95 | δ |
|---|---|---|---|---|---|---|
| `pin − base` | gen_tps | 6 | -0.08 (-0.18%) | 1.0000 | [-1.1, +1.0] | +0.06 ⚠ |
| `pin − base` | prompt_tps | 6 | +246.93 (+115.66%) | 0.0312 | [+241.1, +250.9] | +1.00 |

## GENPIN — the transfer-bound regime (Nemotron-120B) — §B1
`ingest:ab-genpin-nemotron-120b` — 12 records

**12/12 REJECTED** — no comparable metric produced. Dominant reason: `ram 5492MB < 16384MB for 3 samples`.

## TURBO-MMA DECODE — 35B
`ingest:ab-decode-qwen36-35b` — 12 records

| comparison | metric | n | Δ median | sign_p | boot CI95 | δ |
|---|---|---|---|---|---|---|
| `turbo − base` | gen_tps | 6 | -0.02 (-0.04%) | 1.0000 | [-0.7, +0.4] | -0.11 ⚠ |
| `turbo − base` | prompt_tps | 6 | +1.60 (+0.37%) | 0.6875 | [-9.8, +3.4] | +0.11 ⚠ |

## TURBO-MMA DECODE — 30B
`ingest:ab-decode-qwen3-30b` — 12 records

| comparison | metric | n | Δ median | sign_p | boot CI95 | δ |
|---|---|---|---|---|---|---|
| `turbo − base` | gen_tps | 6 | +0.04 (+0.15%) | 1.0000 | [-0.8, +0.3] | +0.00 ⚠ |
| `turbo − base` | prompt_tps | 6 | -2.01 (-0.45%) | 1.0000 | [-35.9, +1.4] | -0.44 ⚠ |

## FORK vs REBASED vs BASE — is the fork still worth carrying? (n=18)
`ingest:ab-rebased` — 54 records

| comparison | metric | n | Δ median | sign_p | boot CI95 | δ |
|---|---|---|---|---|---|---|
| `fork − base` | prompt_tps | 18 | +124.36 (+59.29%) | 0.0000 | [+118.8, +187.6] | +0.49 |
| `rebased − base` | prompt_tps | 18 | +124.22 (+60.36%) | 0.0000 | [+120.8, +190.6] | +0.51 |
| `rebased − fork` | prompt_tps | 18 | +2.21 (+0.48%) | 0.0963 | [-0.1, +5.0] | +0.17 ⚠ |
| `rebased − fork` | gen_tps | 18 | +0.22 (+0.61%) | 0.4807 | [+0.0, +0.6] | +0.09 ⚠ |

## STACK 2×2 — is the L18/A-B disagreement about the BUILD or the prefetch?
`ingest:ab-stack` — 24 records

| comparison | metric | n | Δ median | sign_p | boot CI95 | δ |
|---|---|---|---|---|---|---|
| `prefetch − base` | prompt_tps | 6 | -93.42 (-22.06%) | 0.0312 | [-95.4, -90.5] | -1.00 |
| `stackpf − stack` | prompt_tps | 6 | -98.89 (-23.27%) | 0.0312 | [-102.6, -98.3] | -1.00 |
| `stack − base` | prompt_tps | 6 | +2.75 (+0.65%) | 0.6875 | [-2.3, +5.0] | +0.28 ⚠ |
| `stackpf − prefetch` | prompt_tps | 6 | -5.22 (-1.57%) | 0.0312 | [-7.5, -3.1] | -0.89 ⚠ |

