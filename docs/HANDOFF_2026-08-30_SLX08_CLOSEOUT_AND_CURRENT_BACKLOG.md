# Current research handoff: SLX08 closeout and backlog — 2026-08-30

This is the current restart document for `tare.tools.local-labs`. Older
handoffs remain valuable chronological evidence, but their embedded counts,
queue heads and service observations must not be treated as current state.

## Executive state

- The physical SLX08 selected-block prefill route is implemented in
  `slop.cpp`, opt-in and off by default.
- `BACKLOG-SLX08-RELEVANCE-PREFILL-11` is independently `PROMOTED`.
- The canonical backlog gate passes.
- Historical predecessors remain immutable but are hidden from the operational
  ranking by default.
- The next dependency-ready item is
  `BACKLOG-SLX08-SEMANTIC-PREFILL-12` (`P0`, score `94`).

## Published repository baseline

| Repository | Published commit | Check |
| --- | --- | --- |
| `tare.tools.local-labs` research closeout | [`8ed792c`](https://github.com/augusto-scarvalho/tare.tools.local-labs/commit/8ed792cd682b3c4db77a03c242cd44ba0c07a979) | [local-labs CI passed](https://github.com/augusto-scarvalho/tare.tools.local-labs/actions/runs/33318669712) |
| `slop.cpp` physical implementation | [`d2c6ce6d`](https://github.com/augusto-scarvalho/slop.cpp/commit/d2c6ce6d0e96f1d9951c6cdd40d6cae353cd371d) | [fork changelog policy passed](https://github.com/augusto-scarvalho/slop.cpp/actions/runs/33319602148) |

The commit containing this handoff is the authority for the newer 137-record
backlog snapshot and lineage-aware ranking behavior described below.

## What SLX08 established

The promoted R11 experiment ran 126 fresh balanced dense/position/relevance
triples, 378 physical streamed requests in total. Dense and exact-key relevance
selection both answered 126/126 correctly; the fixed position control answered
54/126. Relevance evaluated 2048 rather than 4096 prompt tokens, improved p50
TTFT by `1.79838x`, improved marginal p95 by `1.83200x`, and was faster than
dense in 126/126 pairs.

The one-sided exact 95% lower accuracy-delta bound was `-0.0234952389`, inside
the preregistered `-0.03` margin. Independent audit reconstructed all rows,
balances, routes, indices, answers, hashes, service recovery and watcher state
before performing `EXECUTED -> VERIFIED -> PROMOTED`.

This qualifies only client-selected token deletion before ordinary dense
prefill on the frozen Qwen3.8 exact-key panel. It does not qualify production
deployment, generic RAG, learned routing, generic sparse attention or latency
that includes selection work.

Primary result: [`research/SLX08_RELEVANCE_PREFILL_2026-08-30.md`](research/SLX08_RELEVANCE_PREFILL_2026-08-30.md).

## Canonical backlog snapshot

Manifest update: `2026-08-30T15:40:53Z`.

The immutable ledger contains 137 records:

| State | Count |
| --- | ---: |
| `PROPOSED` | 1 |
| `IMPLEMENTED` | 7 |
| `EXECUTED` | 13 |
| `VERIFIED` | 1 |
| `PROMOTED` | 34 |
| `REJECTED` | 28 |
| `BLOCKED` | 53 |

Those raw counts intentionally retain every failed, aborted and displaced
attempt. They are not the execution queue. After removing 75 records that have
an explicit successor, 62 current lineage tips remain:

| Current-tip state | Count |
| --- | ---: |
| `PROPOSED` | 1 |
| `VERIFIED` | 1 |
| `PROMOTED` | 27 |
| `REJECTED` | 14 |
| `BLOCKED` | 19 |

There are no current-tip `IMPLEMENTED` or `EXECUTED` packets awaiting an
executor or auditor. Their raw-state occurrences are historical predecessors
with successors. `BACKLOG-SLX08-REAL-FIDELITY-03` remains a bounded retained-only
`VERIFIED` result; it is not authority for a runtime or production claim.

## Current queue head

`BACKLOG-SLX08-SEMANTIC-PREFILL-12` is a deliberately narrow successor to R11.
It freezes 128 natural-document questions and compares three paired arms:

1. dense full-context prefill;
2. a fixed-position half-context control;
3. embedding-selected half-context prefill.

The semantic arm must pay for document embedding, block ranking and serving.
Its gates require physical route telemetry, at least 98% evidence retention,
an exact one-sided quality bound, a ten-point advantage over the position
control, `1.10x` p50 end-to-end speedup, no p95 regression and service recovery.
The experiment cannot reuse the synthetic exact-key panel to claim success.

Admission source:
[`../config/research_backlog_admissions/BACKLOG-SLX08-SEMANTIC-PREFILL-12.json`](../config/research_backlog_admissions/BACKLOG-SLX08-SEMANTIC-PREFILL-12.json).

## Portfolio navigation

The operational ranking now excludes any item named by a successor's
`supersedes` field. This changes presentation only; no state, gate, receipt or
history is rewritten.

```powershell
python tools/analysis/backlog_pipeline.py gate
python tools/analysis/backlog_pipeline.py rank --explain
python tools/analysis/backlog_pipeline.py next
```

To inspect the complete historical ledger, including displaced predecessors:

```powershell
python tools/analysis/backlog_pipeline.py rank --include-superseded --include-terminal
python tools/analysis/backlog_pipeline.py status
```

Expected current `next` result:

```text
BACKLOG-SLX08-SEMANTIC-PREFILL-12 P0
```

## Resume and stop rules

- Scaffold and preregister R12 before implementation; do not treat `PROPOSED`
  as execution authority.
- The watcher may advance only a complete `IMPLEMENTED -> EXECUTED` packet.
- Independent review owns verification, rejection and promotion.
- New packets use the
  [`frugal independent-audit contract`](research/FRUGAL_INDEPENDENT_AUDIT.md):
  an auditor may return executed work only after proving a result-reversing
  defect with decision impact, and must prefer retained-evidence repair over a
  full rerun.
- Preserve every R4-R11 predecessor and receipt; corrections require a new
  successor.
- Stop if the semantic corpus is synthetic, selector timing is omitted, the
  physical route is not observed, service identity drifts or recovery fails.
- Do not relaunch old nonterminal predecessors merely because `status` retains
  their historical state.

## Last verified service closeout

At the end of R11, Qwen3.8 was restored on port 8080 and embeddings remained
healthy on port 8081. This is a closeout receipt, not a permanent health claim;
recheck live services before the next experiment.
