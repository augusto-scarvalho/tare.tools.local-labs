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

The 2026-08-29 execution wave consumed the dependency-ready queue. Current
outcomes, still pending independent review, are:

1. `BACKLOG-FLEET-CONTEXT-ENVELOPE-04` is `EXECUTED` with a preserved negative
   result caused by the executor using raw UTF-8 SHA-256 instead of the
   predecessor's canonical JSON prompt digest. Do not rewrite it; a correction
   requires a successor.
2. `BACKLOG-FLEET-CONTEXT-INTERFERENCE-02` is `BLOCKED` after its versioned
   wrapper aborted before receipt due recursive function substitution. Its
   aborted terminal is preserved and a retry requires a successor.
3. `BACKLOG-GATEWAY-ROUTE-STRESS-02` is `EXECUTED` and audit-ready after all 12
   gates passed over the retained 30-switch/120-request campaign.
4. `BACKLOG-FLEET-REGRESSION-SCREEN-02` is `EXECUTED` and audit-ready: 448/448
   requests, 100% HTTP success, exact repeat rate 1.0, all request payloads
   retained and terminal runner state bound before receipt.
5. `BACKLOG-FLEET-SEEDED-STABILITY-02` is `EXECUTED` and audit-ready: 288/288
   requests, 100% HTTP success, exact seeded repeat rate 1.0, all request
   payloads retained and terminal runner state bound before receipt.

`backlog_pipeline.py next --json` currently returns `null`. This means no
dependency-ready `PROPOSED` item, not that blocked, executed or audit-pending
research disappeared.

Two corrected successors completed the first real watched-wave dispatch and
are now `EXECUTED`, audit-ready:

- `BACKLOG-FLEET-CONTEXT-ENVELOPE-05` fixes only the canonical prompt digest;
- `BACKLOG-FLEET-CONTEXT-INTERFERENCE-03` fixes the recursive wrapper while
  preserving the 31-decoy construct.

Their ordered manifest is
`config/research_trails/continuous_context_repair_2026-08-29.json`. The
supervisor launched both without conversational dispatch and wrote a terminal
wave receipt under `runs/autonomous/CONTINUOUS-CONTEXT-REPAIR-2026-08-29/`.
Independent review order is frozen in
`config/research_audit_queue_2026-08-29_continuous_wave.json`.

## Restart procedure

From the repository root:

```powershell
python tools/analysis/backlog_pipeline.py gate
python tools/analysis/backlog_pipeline.py rank --explain
python tools/analysis/backlog_pipeline.py status
python tools/analysis/backlog_pipeline.py next
```

Expected next value:

```text
null
```

Do not manufacture a next command from `EXECUTED` or `BLOCKED` items. First send
the audit-ready packets to a fresh independent auditor. Any corrected context
retry must be admitted as a successor. A future authorized multi-item queue
must freeze every packet at `IMPLEMENTED` and run through the watched-wave
supervisor. The executor and watcher stop at `EXECUTED`; only the independent
auditor may promote, reject or place a bounded hold.

## Operational baseline at handoff

- Gateway health on port 8080: HTTP 200.
- Embedding health on port 8081: HTTP 200.
- The physical wave completed and restored the initially resident qualified
  route without service restart; exact hardware identity remains in each raw
  receipt rather than this prose handoff.
- No Windows Python process matching the launcher or watcher was active.
- The new physical results remain `EXECUTED` and authorize no model or runtime
  promotion before independent review.

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
  not an autonomous queue mutation or permission to bypass dependencies. Long
  AFK chains must use a frozen `local-labs-watched-wave-v1` manifest through
  `tools/analysis/run_watched_experiment_wave.py`; it dispatches only canonical
  `IMPLEMENTED` packets and stops on any alert.
- Legacy progress is typed: `files` counts marker files and `jsonl_lines` counts
  non-empty sample records. Never equate one JSONL file with its row count.
- Commit and push remain separate maintainer actions.
