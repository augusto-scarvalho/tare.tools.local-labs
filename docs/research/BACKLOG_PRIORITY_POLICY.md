# Frugal backlog priority policy

## Purpose

Backlog state answers whether an experiment is scientifically and operationally
ready. Priority answers which eligible question currently offers the most value.
They are separate concerns: ranking never verifies, rejects, promotes, unblocks
or launches an item.

The authority remains:

- `config/research_backlog.json` for items, dependencies, states and applied
  priority metadata;
- `config/backlog_priority_policy.json` for weights and explicit assessments;
- `tools/analysis/backlog_pipeline.py` for validation, ranking and atomic apply.

The experiment harness contains no portfolio policy. In experiment mode the
watcher is only a completion trigger: it requests an apply, records a compact
summary, refreshes `status` and `next`, and stops fail-closed if any step fails.

## Commands

Explain the current portfolio view without changing a file:

```powershell
python tools/analysis/backlog_pipeline.py rank --explain
```

The default ranking contains only current lineage tips. A predecessor named by
another item's `supersedes` field remains immutable in the manifest but is not
shown as current work. Restore the historical view explicitly:

```powershell
python tools/analysis/backlog_pipeline.py rank --include-superseded --include-terminal
```

Preview only explicit assessment updates:

```powershell
python tools/analysis/backlog_pipeline.py rebalance --dry-run
```

Apply them atomically through the canonical pipeline:

```powershell
python tools/analysis/backlog_pipeline.py rebalance --apply `
  --actor "Codex portfolio review"
```

`--dry-run` is the default. `--apply` requires an actor, validates the manifest
and policy before mutation, changes only priority metadata, appends a bounded
history event and validates the candidate manifest before its atomic replace.
It never edits scientific state, dependencies, gates, claims or raw evidence.
Each applied history row binds its own assessment digest. Adding an unrelated
assessment therefore does not manufacture priority churn for unchanged items.

## Scoring contract

Each explicitly reviewed item receives scores from zero to five for:

| Dimension | Weight |
| --- | ---: |
| Ecosystem leverage | 35% |
| Community innovation | 25% |
| Information per cost | 20% |
| Evidence readiness | 10% |
| Downstream unlock | 10% |

Scores map to P0/P1/P2/P3 at 80/60/40/0. After 30 waiting days, a bounded
anti-starvation bonus adds two points per 30-day period, up to ten. The report
shows base score, waiting days and bonus separately.

Assessments are deliberately sparse. An item without a new assessment keeps
its existing priority and receives no invented score. This prevents a periodic
review from silently reinterpreting all historical work. Explicitly assessed
items are shown first; the remaining portfolio stays ordered by the current
P-band and ID.

Within one P-band, `next` uses the persisted explicit score as a tie-breaker,
then the item ID. It still selects only dependency-ready `PROPOSED` items.
Blocked, executed and terminal states cannot become executable through score.

## When to reassess

Reassess after an independently reviewed wave changes the expected value of
other work, after a material dependency appears or disappears, weekly while a
campaign is active, or monthly while idle. Do not rewrite an assessment merely
because time passed; the bounded aging rule already handles starvation.

Every assessment must state the actor, review time, reason and concrete change
trigger. New evidence changes the policy entry; `rebalance --apply` records the
policy digest in the item's `priority_history`.

## Watcher integration

At the end of a valid `--experiment-mode` run, the watcher performs:

```text
rebalance --apply -> status --json -> next --json
```

The final watcher record retains only the policy digest, counts and changed
IDs, not the full ranking. This keeps controller handoff small. A rebalance
failure changes the final result to `complete_with_alert`; the watcher does not
dispatch the next candidate.

This trigger applies only already-authored assessments. It never reads an
experimental result and invents strategic scores, and it remains independent
of scientific audit authority.
