# Current Unsloth Qwen3.8 revision screen

**Status:** `PREREGISTERED / EXECUTION_STARTED`  
**Date:** 2026-08-21  
**Question:** do the current revision's smaller `UD-IQ4_XS` or `UD-Q2_K_XL` artifacts preserve or improve the
historical local frontier strongly enough to supersede it?

## Frozen arms

| Arm | Bytes | SHA-256 / binding |
|---|---:|---|
| historical IQ4_XS | 15,705,861,088 | `9fd40d7036f5e0918e20aaeebf11468fafd06bb53d4d980eef6bb7e4e4ace666` |
| current IQ4_XS | 14,252,845,984 | `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199` |
| historical Q2_K_XL | 10,676,423,744 | `46151b52a5cad673d90a00222103254864326c251130b8fc4381d6f34386b3c8` |
| current Q2_K_XL | 9,828,981,664 | `fd4730dd8aad070517978752b63d530aeb1740d2283cab9fa24f1e404032ddb0` |
| current separate MTP Q4_0 | 1,369,590,656 | `50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e` |

Current artifacts are bound to `unsloth/Qwen3.8-27B-GGUF` revision
`4ca720788d1e01f1bff70c033e0d0028fd02e502`. They must use a revision-qualified directory and must not
overwrite historical files.

## Controlled compact packet

Use the same deployed `b9863-5e7f6271c` engine, external `qwen-sharp.jinja`, q4 KV, one slot, and 65,536
context tokens for all quality arms. Keep the embedding server on port 8081 alive.

For every arm run, in order:

1. load/residency and deterministic smoke;
2. eight-case agent suite;
3. cache/cancel/reuse lifecycle;
4. paired retrieval, multikey, multihop, and aggregation at 8k/32k/64k, one replicate per cell;
5. `Mbpp/260` at 2,048 tokens;
6. strict replay of the five historical GSM8K failures.

The historical IQ4_XS may reuse its already-valid same-harness agent/GSM receipts only where runtime,
template, and artifact identity match; new current-revision arms require fresh receipts. The Q2 pair must be
run fresh for the discriminating compact packet.

After base-role screening, inventory MTP layout. Run a fixed-seed throughput/acceptance sentinel with the
historical embedded head and current separate head only for candidates that survive correctness gates.

## Decision rules

- Any critical tool failure, blind irreversible retry, cache oracle failure, new context cliff, or higher
  non-termination rate prevents supersession.
- `Mbpp/260` is diagnostic: terminating correctly is a positive discriminator; repeating the incumbent's
  known non-termination is non-improvement, not an automatic whole-model rejection.
- A current artifact must be non-inferior on strict agent/context/GSM correctness and offer a material
  Pareto gain in memory, termination, or task-correct performance before it can supersede its historical peer.
- MTP acceptance alone cannot promote. Task-correct decode latency must improve and deterministic output must
  remain equivalent.
- Stop the expensive full MBPP+/wide-context packet for any arm that fails a compact eligibility gate.
- Preserve invalid attempts separately and restore the exact service baseline before ending the tranche.

## Broad-packet amendment frozen after compact screening

This amendment was written after the compact results were known but before any broad-packet generation.
Only the current `UD-IQ4_XS` survived compact screening. The current `UD-Q2_K_XL` is ineligible because it
added an 8k aggregation miss and changed `Mbpp/260` from a terminating correct answer in its historical peer
to a 2,048-token truncation.

Run the surviving current `UD-IQ4_XS`, without MTP, through:

1. all 378 MBPP+ tasks at the historical 768-token cap, scored with EvalPlus 0.2.0;
2. the frozen 100-item GSM8K manifest and strict scorer used by the 2026-08-20 requalification;
3. three replicates of every retrieval, multikey, multihop, and aggregation cell at 8k/32k/64k.

The broad base artifact is non-inferior only if agent and cache remain perfect, MBPP Base loses no more than
2 tasks from 326/378, MBPP Plus loses no more than 3 tasks from 284/378, GSM8K scores at least 94/100, and
the replicated context run shows no repeatable new failure mode beyond the already-observed aggregation
position sensitivity. These small tolerances acknowledge quantization variation while requiring the 1.45 GB
artifact reduction to be the material Pareto gain.

Only after all of those conditions pass may the current separate MTP head enter a deterministic no-spec/MTP
sentinel. MTP promotion additionally requires byte-equivalent answers within each arm, unchanged task
correctness across arms, and a material median task-correct latency or throughput improvement. Any failed
condition leaves the deployed historical artifact and embedded MTP head unchanged.
