# Documentation map

This is the shortest reliable entry point into `tare.tools.local-labs`.
The repository preserves many dated reports because failed experiments,
superseded conclusions and independent reviews are evidence. They are not all
current operating instructions.

## Authority order

When documents disagree, use this order:

1. Machine-readable state in
   [`config/research_backlog.json`](../config/research_backlog.json),
   [`config/qualified_model_fleet.json`](../config/qualified_model_fleet.json)
   and registered run packets.
2. The [current research handoff](HANDOFF_2026-08-30_SLX08_CLOSEOUT_AND_CURRENT_BACKLOG.md)
   for restart context, current queue interpretation and bounded live-state
   observations.
3. Method contracts for how work may advance.
4. Scientific catalogs and result reports for measured findings.
5. Dated handoffs and closeouts as historical evidence only.

Prose never overrides a receipt, frozen gate or canonical backlog state. A
service observation in a handoff is a timestamped closeout, not permanent
health evidence.

## Start here

| Need | Canonical document or command |
| --- | --- |
| Resume current work | [Current handoff](HANDOFF_2026-08-30_SLX08_CLOSEOUT_AND_CURRENT_BACKLOG.md) |
| See the executable queue | `python tools/analysis/backlog_pipeline.py next` and `rank --explain` |
| Inspect the complete ledger | `python tools/analysis/backlog_pipeline.py status` |
| Understand backlog state and receipts | [Backlog implementation pipeline](research/BACKLOG_IMPLEMENTATION_PIPELINE.md) |
| Understand prioritization | [Backlog priority policy](research/BACKLOG_PRIORITY_POLICY.md) |
| Run long experiment chains | [Experiment watcher](research/EXPERIMENT_WATCHER.md) |
| Perform independent review | [Frugal independent audit](research/FRUGAL_INDEPENDENT_AUDIT.md) |
| Discover qualified models | [Qualified model fleet](QUALIFIED_MODEL_FLEET.md) |
| Browse scientific findings | [Research knowledge base](research/README.md) |
| Browse the broad taxonomy | [Research catalog](RESEARCH_CATALOG.md) |
| Inspect engine-specific conclusions | [slop.cpp fork report](research/FORK.md) |

## Current control surfaces

- `config/research_backlog.json` is the immutable-lineage research ledger.
- `config/backlog_priority_policy.json` scores current portfolio value without
  changing scientific outcomes.
- `tools/analysis/backlog_pipeline.py` validates and advances backlog packets.
- `tools/analysis/watch_experiment_processes.py` observes authorized runs and
  may close execution, but cannot audit or promote them.
- `config/qualified_model_fleet.json` defines the models agents may route
  through the OpenAI-compatible gateway.
- `CHANGELOG.md` records material repository changes.

## Document lifecycle

- Current documents describe where to resume; dated documents explain how the
  repository got there.
- Superseded or rejected evidence is retained and linked through successor
  lineage instead of rewritten.
- New summaries should link to primary results rather than copy large metric
  tables.
- Only one handoff should be labeled current. Older `HANDOFF_*.md` files and
  [`HANDOFF.md`](HANDOFF.md) are historical ledgers.
- Update this map, the current handoff and the changelog when a change alters
  the restart boundary or documentation authority.
