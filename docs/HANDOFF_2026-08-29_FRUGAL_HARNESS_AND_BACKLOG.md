# Frugal experiment harness and canonical backlog handoff — 2026-08-29

Read this document first after a context reset. It supersedes the operational
instructions in `HANDOFF_2026-08-28_CONTINUOUS_RECOVERY_CLOSEOUT.md` while
preserving that file as the scientific closeout of the preceding campaign.

## Outcome

The specialized experiment harness, launcher, watcher and compact continuation
path are implemented and independently approved. The result is intentionally a
small seed rather than a workflow platform:

- one runner-side lifecycle context;
- one foreground launcher;
- one low-noise watcher;
- one terminal receipt and append-only journal;
- one compact handoff to the existing backlog pipeline and independent auditor.

No daemon, scheduler, database, plugin layer or second state machine was added.
The canonical authority remains `config/research_backlog.json` plus
`tools/analysis/backlog_pipeline.py`.

## Durable implementation

| Artifact | Role |
| --- | --- |
| `src/model_lifecycle/experiment_harness.py` | Fresh-attempt lock, journal, samples, receipt and atomic `SEALED`/`ABORTED` terminal. |
| `tools/analysis/launch_watched_experiment.py` | Pre-spawn authority check, worker launch, deadline, exit receipt, tree cleanup and foreground completion delivery. |
| `tools/analysis/watch_experiment_processes.py` | Evidence validation, health observation, legal `IMPLEMENTED -> EXECUTED` transition and compact next-item signal. |
| `tools/analysis/smoke_experiment_mode.py` | Temporary non-mutating end-to-end canary. |
| `tools/analysis/mutation_test_experiment_harness.py` | Seeded semantic mutation gate with a persisted hash-bound report. |
| `docs/research/EXPERIMENT_WATCHER.md` | Complete operator and failure-semantics reference. |
| `config/backlog_priority_policy.json` | Sparse, explicit portfolio assessments and frugal scoring weights. |
| `docs/research/BACKLOG_PRIORITY_POLICY.md` | Ranking, rebalance, anti-starvation and watcher-trigger contract. |

The launcher rejects managed work unless the requested ID and packet path are
canonical and both manifest state and `PIPELINE.json` stage are exactly
`IMPLEMENTED`. It performs that check before creating logs or calling `Popen`.
Existing `runner.stdout.log` or `runner.stderr.log` files are never truncated;
a new attempt is required. Timeout, early watcher exit and watcher-spawn failure
terminate the worker tree on Windows or its isolated process group on POSIX.

`--unmanaged-canary` is the only explicit authority bypass. It cannot mutate the
backlog or enter `audit_ready_ids`.

## Independent audit closeout

The final independent auditor verdict was `APPROVE`. The auditor reproduced:

- process-parent and real child termination;
- byte-for-byte preservation of existing runner logs with zero spawn attempts;
- rejection of `PROPOSED/PROPOSED`, `IMPLEMENTED/PROPOSED` and
  `PROPOSED/IMPLEMENTED` state combinations before logs or spawn;
- positive launch controls for `IMPLEMENTED/IMPLEMENTED` and the explicit
  unmanaged canary;
- the full fixture, smoke, backlog and mutation gates.

Final evidence at closeout:

| Check | Result |
| --- | --- |
| Focused harness/watcher fixtures | 131 passed |
| Priority pipeline/watcher fixtures | 90 passed |
| Full repository suite | 366 passed |
| Live non-mutating smoke | PASS |
| Canonical backlog gate | PASS |
| Seeded semantic mutants | 76 killed, 0 survived, 0 invalid |
| `git diff --check` | PASS |

The persisted mutation report is
`runs/benchmarks/HARNESS-MUTATION-2026-08-28/report.json`, generated at
`2026-08-29T12:21:19Z`, with SHA-256
`91cff8f9722d7094426bd9226ec6c94d073944c79c23b8d603c5b71088394434`.
Its recorded source and test hashes matched the final audited worktree.

## Canonical backlog snapshot

The backlog required no scientific state transition for this infrastructure
change. Direct state edits remain forbidden. Live validation on 2026-08-29 found
108 records:

| State | Count |
| --- | ---: |
| `PROMOTED` | 24 |
| `REJECTED` | 21 |
| `EXECUTED` | 19 |
| `BLOCKED` | 39 |
| `IMPLEMENTED` | 2 |
| `PROPOSED` | 3 |

The 19 `EXECUTED` records include preserved HOLD/superseded evidence from prior
audit waves. Their presence is not authority to rerun or promote them. The two
historical `IMPLEMENTED` SLX-03 build packets are displaced by reviewed
successors and must not be relaunched merely because they are nonterminal.
Blocked items retain their recorded unblock conditions; rejected evidence
remains immutable and may be revisited only through a successor.

## Exact continuation queue

The portfolio policy now ranks explicitly reviewed work by ecosystem leverage,
community innovation, information per cost, evidence readiness and downstream
unlock. It cannot change scientific state or make a blocked item executable.
Items without a fresh assessment preserve their current priority. The watcher
requests an atomic rebalance only when an `--experiment-mode` wave finishes and
stores a compact result.

`backlog_pipeline.py next` selects:

1. `BACKLOG-FLEET-CONTEXT-ENVELOPE-04` (P0): reconstruct and recompute all 72
   retained rows from an immutable final source set, with no new inference.
2. `BACKLOG-FLEET-CONTEXT-INTERFERENCE-02` (P0): reconstruct the 72 retained
   hard-decoy rows, including the frozen 31-decoy construct.
3. `BACKLOG-GATEWAY-ROUTE-STRESS-02` (P0 after portfolio review): recompute the retained 30-switch,
   120-request campaign while separating transport, route identity and semantic
   content eligibility.

The first item is the only default next launch. The second and third remain
dependency-ready successors, not permission to run out of order. These three
are retained-evidence recomputations and should consume no GPU.

## Restart procedure

From the repository root:

```powershell
python tools/analysis/backlog_pipeline.py gate
python tools/analysis/backlog_pipeline.py rank --explain
python tools/analysis/backlog_pipeline.py status
python tools/analysis/backlog_pipeline.py next
```

Expected next ID:

```text
BACKLOG-FLEET-CONTEXT-ENVELOPE-04
```

Scaffold and preregister only the selected item, advance legally to
`IMPLEMENTED`, then launch it through the foreground launcher. The executor and
watcher stop at `EXECUTED`. A fresh independent auditor must recompute evidence,
look for false positives and false negatives, and alone decide promotion,
rejection or a bounded hold.

## Operational baseline at handoff

- Gateway health on port 8080: HTTP 200.
- Embedding health on port 8081: HTTP 200.
- RTX 3090 memory at the observation point: 19,526 MiB of 24,576 MiB used;
  this is a point observation, not a model-identity receipt.
- No Windows Python process matching the launcher or watcher was active.
- This work changed infrastructure and documentation only; it produced no new
  scientific result and authorizes no model or runtime promotion.

## Non-negotiable boundaries

- Passing tests prove software consistency, not experimental truth.
- A `SEALED` terminal proves local self-consistency, not executor honesty.
- Raw receipts and failed predecessors remain immutable.
- Existing runner logs require a fresh attempt; never delete them to force a
  rerun.
- The launcher may start only a canonical `IMPLEMENTED` packet.
- The watcher may advance only `IMPLEMENTED -> EXECUTED` after complete
  evidence; it never self-reviews.
- `dispatch_next_candidate` is a compact instruction to the controlling agent,
  not an autonomous queue mutation or permission to bypass dependencies.
- Commit and push remain separate maintainer actions.
