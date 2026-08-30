# SLX08 relevance-selected physical prefill

Date: 2026-08-30

Final state: `PROMOTED`

Claim: `SLX08_RELEVANCE_SELECTED_PREFILL_QUALIFIED_R11`

## Executive result

The physical serving experiment showed that Qwen3.8 can evaluate half of a
4096-token prompt without losing exact-key retrieval quality when the client
selects the correct eight of sixteen 256-token blocks. On 126 new balanced
cases, dense and relevance-selected prefill both answered 126/126 correctly;
the fixed position-only selector answered 54/126.

The relevance arm reduced evaluated prompt tokens from 4096 to 2048, improved
median time to first non-empty streamed content by `1.79838x`, improved marginal
p95 by `1.83200x`, and was faster than dense in every paired case. A one-sided
exact 95% bound placed the worst supported relevance failure rate at `2.3495%`,
inside the preregistered 3% noninferiority margin.

This qualifies the combination of an exact-key client selector and physical
server token compaction on the frozen Qwen3.8 panel. It does not qualify a
production selector, generic RAG, server-side semantic routing or sparse
attention.

## What was implemented

The experimental server path remains off by default and requires
`SLOP_EXPERIMENTAL_SLX08=1`. A request may enable
`slx08_selected_block_prefill` and optionally send
`slx08_selected_block_indices`.

The server rejects explicit selections unless they:

- retain exactly half of the prompt blocks;
- use strictly increasing, unique, in-range indices;
- retain the first and final blocks;
- operate on an even exact multiple of 256 text tokens;
- contain no media.

The final response reports route, selection mode, original and retained token
counts, original and retained block counts, retained fraction and the exact
selected indices. The implementation compacts tokens before the existing dense
prefill path; it adds no CUDA kernel and makes no attention sparsity claim.

## Experiment lineage

| Packet | Outcome | Meaning |
|---|---|---|
| Physical R4 | Operational abort | Physical route existed, but volatile service identity and one-token empty semantics invalidated the attempt. |
| Physical R5 | Rejected | Alternating half-context was fast but reduced accuracy from 38/64 to 28/64, proving that indiscriminate deletion is unsafe. |
| Relevance R6-R9 | Operational aborts | Tokenizer ordering, direct-file import, digest transcription and delegated evidence writing failed closed with zero measured rows. |
| Relevance R10 | Audit hold | Dense/relevance reached 64/64 and p50 speedup 1.7916x, but a degenerate normal CI falsely reported zero uncertainty and the wrapper chain was incompletely bound. |
| Relevance R11 | Promoted | New balanced panel, exact one-sided bound, short bound code path and independent approval. |

The failed predecessors remain immutable because they explain both harness
repairs and why R11 was necessary. They are not counted as scientific samples.

## R11 protocol

- 126 new case IDs, 126 through 251.
- 126 dense, 126 naive and 126 relevance requests: 378 total.
- Exactly nine cases at each of 14 middle evidence positions.
- Exactly 42 cases in each three-arm execution-order period.
- Identical 4096-token prompt bytes across each triple.
- Dense retains 16/16 blocks.
- Naive retains `[0,2,4,6,8,10,12,15]`.
- Relevance retains 8/16 blocks selected by exact query-key overlap.
- Greedy streamed decoding with `n_predict=16`, `temperature=0`, `top_k=1`,
  `seed=0` and `cache_prompt=false`.
- TTFT measured from HTTP POST to first non-empty streamed content.
- Noninferiority uses the exact one-sided binomial upper bound on relevance
  failures among dense-success opportunities.

With zero failures in 126 opportunities, the exact bound is:

```text
1 - 0.05^(1/126) = 0.0234952388666701
```

The corresponding conservative accuracy-delta lower bound is `-0.0234952389`,
which passes the frozen `-0.03` gate. The 64-case R10 bound was `-0.0457297023`
and correctly failed independent audit.

## Independently reconstructed results

| Metric | Dense | Naive | Relevance |
|---|---:|---:|---:|
| Requests | 126 | 126 | 126 |
| Correct | 126 | 54 | 126 |
| Accuracy | 100.00% | 42.86% | 100.00% |
| Evaluated prompt tokens | 4096 | 2048 | 2048 |
| p50 TTFT | 3195.56 ms | 1785.20 ms | 1784.49 ms |
| p95 TTFT | 3360.17 ms | 1847.74 ms | 1834.16 ms |

Additional checks:

- relevance retained evidence in 126/126 cases;
- naive retained evidence in 54/126 and was correct in exactly those 54;
- response telemetry matched requests and evaluated token counts in 378/378;
- prompt cache reuse was zero in all measured rows;
- relevance was faster than dense in 126/126 pairs;
- the strict whole-response four-digit scorer matched the stored scorer;
- no R11 prompt hash overlapped R10;
- Qwen3.8 and embedding services were restored successfully.

## Audit result

Independent audit performed the canonical transitions:

```text
EXECUTED -> VERIFIED -> PROMOTED
```

It recomputed every row, balance cell, route, selected index, answer, timing,
bound, artifact identity, source binding, watcher terminal and service recovery.
No result-reversing false positive or false negative remained.

- Receipt SHA-256:
  `2d32c1fce9149d9cee46eeea725a0065c009e1287ec51f45c098bd846af8071f`
- Samples SHA-256:
  `7eae2f7168a4383b964a5780b1b79db72dcc47bc1170d03161131ea2ffc12442`
- Review SHA-256:
  `5088bbfc14919b09d0cf3d035cefc0fb6a77810aa3d9c4015d38dadba5d7f1d5`
- Tests: 109 focused and 494 repository-wide, all passing.

## Limits and next step

The client already knows the exact query key and scans original block text. The
experiment excludes selector latency, so the measured speedup covers server
prefill only. Four-digit codes are synthetic, although R11 uses prompt hashes
disjoint from R10 and the naive arm demonstrates that missing evidence is not
recovered by the code pattern.

The next high-value step is a separate preregistered packet that includes
selector time and uses natural long documents with a semantic or embedding
selector. It should compare total end-to-end latency and answer quality against
dense prefill and a position-only control. Until then, keep this route
experimental and off by default.

## Primary evidence

- `runs/research/BACKLOG-SLX08-RELEVANCE-PREFILL-11/PRE_REGISTRATION.md`
- `runs/research/BACKLOG-SLX08-RELEVANCE-PREFILL-11/RESULT.md`
- `runs/research/BACKLOG-SLX08-RELEVANCE-PREFILL-11/REVIEW.json`
- `runs/research/BACKLOG-SLX08-RELEVANCE-PREFILL-11/raw/receipt.json`
- `runs/research/BACKLOG-SLX08-RELEVANCE-PREFILL-11/raw/samples.jsonl`
- `runs/autonomous/SLX08-RELEVANCE-R11-2026-08-30/FINAL.json`
- `tools/research/run_slx08_relevance_prefill_r11.py`
- `tests/test_slx08_relevance_prefill_r11.py`
