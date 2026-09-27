# Qwen3.8-27B on NInfer-3090 — paired cohort (2026-09-27)

Engine: NInfer-3090 v0.6.1 Linux release (Don-Chad/ninfer-3090, archive SHA-256
`eb6a6e5b…65b5`), unqualified upstream on Linux. Artifact: `neroued/Qwen3.8-27B-NInfer`
revision `18dfc887423f` (`qwen3_8_27b.ninfer`, 18,210,531,328 bytes, SHA-256
`eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e`), the revision validated
for the RTX 3090. Server: 32K context, INT8 KV, MTP with 3 draft tokens, one request.

## Result

Six real SpecGraph regressions, same instruction, verifier, limits (24 calls, 900 s,
32K) and tare Kernel `23521c8` for every arm; full suite checked after each delivery.

| Arm | Fixed | Agent time | GPU busy | Marginal energy |
| --- | ---: | ---: | ---: | ---: |
| NInfer + Qwen3.8 (prefix reuse on) | 4/6 | 711 s | 7.2 min | ~51 Wh |
| slop.cpp + Swift-1.5 Q4_K_L | 4/6 | 1,014 s | 11.8 min | ~78 Wh |
| slop.cpp + Qwen3.8 UD-Q4_K_XL | 3/6 | 1,064 s | 12.7 min | ~87 Wh |

Every unfinished case stopped on tare context management at 32K or on a truncated reply,
not on the engine. Speed on identical requests: decode ~66 vs ~54 tok/s, prefill of an
~8K prompt ~1,020 vs ~1,235 tok/s (NInfer vs slop.cpp); NInfer reuses the prior turn with
MTP active (`restore_turn_checkpoint`), slop.cpp runs with its prompt cache off.

## Limits

- No `/props`, `/apply-template` or `/tokenize`: fleet counting and bindings are refused;
  clients count by estimate.
- No reasoning-token budget in 0.6.1; `chat_template_kwargs.enable_thinking` is rejected
  (the gateway moves thinking controls to the top level).
- n = 6 single runs on one Python repository; not a general quality ranking.

Full report and scripts: tare.tools.os `docs/operations/LOCAL_ROUTE_COHORT_2026-09-27.md`,
`.reconciliation/evidence-store/reports/local-route-cohort-20260927`.
