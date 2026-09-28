# NInfer-3090 0.11 — Qwen3.8 and Swift 1.5 27B GSQ-RCO cohort (2026-09-28)

Engine: NInfer-3090 0.11.0-rtx3090, `iamwavecut/ninfer-all` commit
`f118551fb401de073555807a48c50238e180e3b8`, built on the node for sm_86 (CUDA 13.3, GCC 13) and
installed read-only at `~/opt/ninfer/v0.11.0-rtx3090-f118551` (`ninfer-serve` SHA-256
`1a3a1a1cd533f2469aa2eedf36d1b4fe76e1317f6998a5adeaea2597a772ee3e`). This line multiplies GGUF
blocks in place (`gguf_blocks_v1`), so GSQ-RCO GGUFs convert without requantization.

Artifacts:

- Qwen3.8-27B GSQ-RCO IQ3_S, `WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3`
  (15,017,456,896 bytes, SHA-256 `29e54467ca0ddae7e9aac04439f72ad20dbf6dc8fb5e84833b41ec7de9c46750`;
  from ISTA-DASLab's GSQ-RCO GGUF; Apache-2.0).
- Swift 1.5 Qwen3.8-27B GSQ-RCO IQ3_S-mtp, `ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF`
  (GGUF SHA-256 verified against the release), converted on the node with recipe
  `qwen3_8_27b_gguf`, components text+mtp, NInfer's `qwen3_8.jinja`
  (`swift15_27b_gsq_rco_iq3_s.ninfer`, 12,494,820,864 bytes; Swift Open License 1.0).
  Swift's own template is byte-identical to Qwen3.8's; NInfer's adds mid-conversation system
  turns and tool-result histories after compaction.

Server: 128K context (`--kv-capacity auto`), rk8v4 KV, FP16 GDN state, MTP 3 drafts with the
proposal head, prefix reuse on. VRAM with 128K reserved: ~18.1 GB used, ~6.2 GB free; 8.6 GB of
host RAM pinned for the context cache (`--host-cache-mib` lowers it). Decode ~107 tok/s on a
1,500-token answer; 130–134 tok/s with MTP in agent turns; 96–97% prompt cache hits.

## Result

Six real SpecGraph regressions, same instruction, verifier and limits (24 calls, 900 s) as the
2026-09-27 cohort; full suite checked after each delivery. GPU energy is the board's total over
each run window, the same method for every row.

| Arm | Kernel | Fixed | Agent time | Calls | GPU energy |
| --- | --- | ---: | ---: | ---: | ---: |
| slop.cpp + Qwen3.8 UD-Q4_K_XL, production (32K, exact counting) | 23521c8 | 6/6 | 919 s | 66 | 72 Wh |
| **NInfer 0.11 + Qwen3.8 GSQ-RCO IQ3_S** (128K, low effort, 2,048 thinking budget) | 23521c8 | **6/6** | **478 s** | 64 | **33 Wh** |
| same, confirmation | b2816e8 | **6/6** | 515 s | 68 | 40 Wh |
| NInfer 0.11 + Swift 27B (32K, low, 2,048 budget) | 23521c8 | 5/6 | 731 s | 55 | 43 Wh |
| NInfer 0.11 + Swift 27B (128K, low, 2,048 budget) | 23521c8 | 4/6 | 542 s | 80 | 41 Wh |
| NInfer 0.11 + Swift 27B (128K, xhigh, no budget, 16K output) | 23521c8 | 5/6 | 388 s | 61 | 32 Wh |
| same, second run | b2816e8 | 6/6 | 604 s | 70 | 53 Wh |

Swift misses: at 32K, tare's token estimate read a ~24K prompt as 33.5K and compaction could not
reach its target (COMPACTION_TARGET_NOT_REACHED); at 128K/low, two cases ran out of calls, one
without any edit; at xhigh, four `read` calls without `repository` and with `offset` ended in
NO_PROGRESS. Kernel b2816e8 makes those refusals name the fields; no argument refusal occurred in
either second run, so Swift's 6/6 there is run-to-run variation, not that fix.

UkisAI benchmarks Swift at `xhigh` (its template default) without a thinking budget; `low` adds a
"keep your thinking brief" instruction. At `xhigh` Swift used ~1.7x the output tokens of Qwen3.8
at `low`.

## Limits

- No `/tokenize`: fleet counting and bindings are refused; clients estimate tokens. Use the 128K
  window: at 32K the estimate (~38% high) triggers compaction early.
- No vision in the Swift artifact (no projector published for this release).
- n = 6 cases per run on one Python repository; two runs per candidate.

Scripts and per-case receipts: `.reconciliation/evidence-store/reports/local-route-cohort-20260927`
in the tare.tools workspace (`chain4.sh`–`chain8.sh`, `*/ninfer011-*/result.json`).
