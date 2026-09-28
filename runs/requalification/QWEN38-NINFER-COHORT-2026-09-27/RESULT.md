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
| slop.cpp + Qwen3.8, production config (prompt cache, 2048-token reasoning budget, exact fleet counting) | 6/6 | 919 s | 8.6 min | ~59 Wh |
| NInfer + Qwen3.8 (prefix reuse, estimate counting, no reasoning budget) | 4/6 | 711 s | 7.2 min | ~51 Wh |
| slop.cpp + Swift-1.5 Q4_K_L (cache off, estimate counting) | 4/6 | 1,014 s | 11.8 min | ~78 Wh |
| slop.cpp + Qwen3.8 (cache off, estimate counting) | 3/6 | 1,064 s | 12.7 min | ~87 Wh |

The last two rows ran without the production prompt cache by mistake and are not a fair
engine comparison. Against the production configuration, NInfer used ~23% less agent time,
~17% less GPU time and ~14% less energy, and fixed 4/6 against 6/6: both misses were
COMPACTION_TARGET_NOT_REACHED, which exact token counting and a reasoning budget address.

Every unfinished case stopped on tare context management at 32K or on a truncated reply,
not on the engine itself. Server-side decode medians with MTP 3: NInfer ~78 tok/s, slop.cpp
~62 tok/s (Qwen3.8) and ~64 tok/s (Swift-1.5); prefill of an ~8K prompt ~1,020 vs ~1,235 tok/s.
Both production engines reuse prefixes: NInfer restores the prior turn checkpoint with MTP
(`restore_turn_checkpoint`), slop.cpp uses its qualified prompt cache.

## Limits

- No `/props`, `/apply-template` or `/tokenize`: fleet counting and bindings are refused;
  clients count by estimate.
- No reasoning-token budget in 0.6.1; `chat_template_kwargs.enable_thinking` is rejected
  (the gateway moves thinking controls to the top level).
- n = 6 single runs on one Python repository; not a general quality ranking.

Full report and scripts: tare.tools.os `docs/operations/LOCAL_ROUTE_COHORT_2026-09-27.md`,
`.reconciliation/evidence-store/reports/local-route-cohort-20260927`.
