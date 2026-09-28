# Swift 1.5 Qwen3.8-Flash-Next on Strata — cohort (2026-09-28)

Engine: Strata `b742ff9` (Niko1221/Strata), engine compiled on the node for sm_86. Model:
`ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, IQ3_XXS (shard 1 SHA-256
`3bddaa667c750f63baca766df9433e1afd35a139f401c5ca5e7ff560ceff3d23`, shard 2
`b0b15f782af71eb471909d2f3313b2927c324962f86c084325c247cd411a0160`); Swift Open License 1.0.
128K context, int8 KV fully in VRAM, MTP draft, 8,000 expert slots (base-model profile).

Six real SpecGraph regressions, tare Kernel `23521c8`, 24 calls / 900 s, estimate counting:

| Variant | Fixed | Calls | Agent time | Output tokens |
| --- | ---: | ---: | ---: | ---: |
| Swift Flash-Next IQ3_XXS | 6/6 | 59 | 873 s | 13.0 k |
| Swift Flash-Next IQ2_XS | 6/6 | 58 | 890 s | 15.3 k |
| ISTA Qwen3.8-Flash-Next IQ2_XS | 6/6 | 69 | 1,273 s | 18.6 k |

Server-side decode ~45 tok/s, prefill of an ~8K prompt ~490 tok/s. Loaded: ~2.9 GB VRAM free,
~42 GB RAM used in WSL (all 24,576 experts pinned; the VRAM slots are copies).

## Limits

- Holds ~43 GB of RAM while loaded: not for use while the Windows desktop is in use, nor with the
  image pipeline. The gateway unloads it after 15 idle minutes.
- No fleet counting (/tokenize); clients estimate tokens. Six single runs on one repository.

Evidence: `.reconciliation/evidence-store/reports/flash-next-20260927` and
`local-route-cohort-20260927` in the tare.tools workspace.
