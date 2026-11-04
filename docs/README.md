# Documentation map

This is the entry point into the `tare.tools.local-labs` documentation. The repository keeps dated reports
because failed experiments, superseded conclusions and independent reviews are evidence. They are not all
current operating instructions.

## What is published

The public repository holds the write-ups, the tooling and the tests. The raw run records under `runs/`, the
machine-readable research backlog (`config/research_backlog.json` and its packets), the agent handoffs and audit
ledgers, the raw evidence bundles under `docs/research/evidence/`, third-party datasets and model weights stay
private. Documents still name those paths so a result can be traced to its record, but the records themselves
are not in this repository.

## Authority order

When documents disagree, use this order:

1. Receipts and machine-readable state (the run records and the research backlog, kept private) and
   [`config/qualified_model_fleet.json`](../config/qualified_model_fleet.json).
2. Method contracts for how work may advance.
3. Scientific catalogs and result reports for measured findings.
4. Dated reports as historical evidence only.

Prose never overrides a receipt, frozen gate or canonical backlog state.

## Start here

| Need | Document or command |
| --- | --- |
| Understand backlog states and receipts | [Backlog implementation pipeline](research/BACKLOG_IMPLEMENTATION_PIPELINE.md) |
| Understand prioritization | [Backlog priority policy](research/BACKLOG_PRIORITY_POLICY.md) |
| Run long experiment chains | [Experiment watcher](research/EXPERIMENT_WATCHER.md) |
| Run WSL jobs from Windows | [`ops/wsl/README.md`](../ops/wsl/README.md) |
| Perform an independent review | [Frugal independent audit](research/FRUGAL_INDEPENDENT_AUDIT.md) |
| Measure local/cloud routes and loading cost | [EXP-3 bounded locality result](research/LOCALITY_COST_QUALIFICATION.md) and `python tools/analysis/locality_study.py STUDY.json` |
| Discover qualified models | [Qualified model fleet](QUALIFIED_MODEL_FLEET.md) (moved to the node service in tare.tools.os) |
| Browse scientific findings | [Research knowledge base](research/README.md) |
| Browse the broad taxonomy | [Research catalog](RESEARCH_CATALOG.md) |
| Inspect engine-specific conclusions | [slop.cpp fork report](research/FORK.md) |
| Follow the changelog rules | [Changelog policy](CHANGELOG_POLICY.md) |

## Control surfaces

- `tools/analysis/backlog_pipeline.py` validates and advances backlog packets. It needs the private backlog
  (`config/research_backlog.json`); its tests build their own temporary backlog.
- `config/backlog_priority_policy.json` scores current portfolio value without changing scientific outcomes.
- `tools/analysis/watch_experiment_processes.py` observes authorized runs and may close execution, but cannot
  audit or promote them.
- `config/qualified_model_fleet.json` is the registry of qualified models that the research cites; the live
  catalog ships with the node service (`node/` in tare.tools.os).
- `CHANGELOG.md` records material repository changes.

## Document lifecycle

- Current documents describe where to resume; dated documents explain how the repository got there.
- Superseded or rejected evidence is retained and linked through successor lineage instead of rewritten.
- New summaries should link to primary results rather than copy large metric tables.
- Update this map and the changelog when a change alters documentation authority.
