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
115 records:

| State | Count |
| --- | ---: |
| `PROMOTED` | 29 |
| `REJECTED` | 21 |
| `EXECUTED` | 22 |
| `BLOCKED` | 40 |
| `IMPLEMENTED` | 3 |

The 21 `EXECUTED` records include preserved HOLD/superseded evidence from prior
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

The 2026-08-29 execution wave consumed the dependency-ready queue. Independent
review of the five-packet closeout queue is complete:

1. `BACKLOG-FLEET-CONTEXT-ENVELOPE-04` is `EXECUTED` with a preserved negative
   result caused by the executor using raw UTF-8 SHA-256 instead of the
   predecessor's canonical JSON prompt digest. Do not rewrite it; a correction
   requires a successor.
2. `BACKLOG-FLEET-CONTEXT-INTERFERENCE-02` is `BLOCKED` after its versioned
   wrapper aborted before receipt due recursive function substitution. Its
   aborted terminal is preserved and a retry requires a successor.
3. `BACKLOG-GATEWAY-ROUTE-STRESS-02` is `PROMOTED` for bounded transport and
   route stress. Empty `muse-vision` outputs remain excluded from semantic or
   visual claims.
4. `BACKLOG-FLEET-REGRESSION-SCREEN-02` is `PROMOTED` for request integrity and
   exact greedy repeatability. Its last-number math scores are descriptive and
   authorize no quality claim.
5. `BACKLOG-FLEET-SEEDED-STABILITY-02` remains `EXECUTED` under
   `HOLD_FAIL_CLOSED`: 192/192 seeded comparisons were exact, but the run did
   not bind the physical GGUF and runtime artifact used by each route.

`backlog_pipeline.py next --json` currently returns `null`. This means no
dependency-ready `PROPOSED` item, not that blocked, executed or audit-pending
research disappeared.

Two corrected successors completed the first real watched-wave dispatch and
were independently promoted:

- `BACKLOG-FLEET-CONTEXT-ENVELOPE-05` established that R4 was a digest-contract
  false negative and recovered all 72 unchanged physical responses;
- `BACKLOG-FLEET-CONTEXT-INTERFERENCE-03` fixed the recursive wrapper and
  reconstructed all 72 prompts with exactly 31 decoys, while limiting the claim
  to that synthetic construct.

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

Do not manufacture a next command from `EXECUTED` or `BLOCKED` items. The next
scientific repair candidate, `BACKLOG-FLEET-SEEDED-STABILITY-03`, is now
`EXECUTED` and audit-ready. It captured 288 exact HTTP arguments, 192/192 exact
seeded comparisons, and bound each route to the live backend PID, `/proc`
command, executable SHA-256 and loaded GGUF SHA-256 before requests. All 12
executor gates passed, the watcher terminal is complete, and `qwen38` plus the
embedding service were restored. Independent review placed R3 in
`HOLD_FAIL_CLOSED`: all 288 row hashes were shifted by four route canaries, and
the binary gate tested only digest presence because the registry froze no
expected executable digest. Preserve R3. A successor must atomically bind only
experimental request/response calls, require zero pending calls and freeze
expected binary SHA-256 plus bytes. A future authorized multi-item queue
must freeze every packet at `IMPLEMENTED` and run through the watched-wave
supervisor. The executor and watcher stop at `EXECUTED`; only the independent
auditor may promote, reject or place a bounded hold.

`BACKLOG-FLEET-SEEDED-STABILITY-04` is the independently `PROMOTED` successor for
those two blockers. Its frozen binary ledger is
`config/fleet_runtime_binary_identities_2026-08-29.json`; its runner excludes
route canaries, pairs the HTTP argument to the returned response ID, preserves
that request without reconstruction and requires a drained capture queue. The
watched run and independent review both found 288/288 request-hash and
response-ID matches, 192/192 exact seeded comparisons, four exact binary/GGUF
bindings, zero restarts and restored `qwen38`. The claim is limited to seeded
stability on this frozen panel and artifact set; 126/192 math responses were
truncated, so no quality or unseeded-determinism claim follows.

`BACKLOG-SLX11-OFFICIAL-HYBRID-02` is the next `IMPLEMENTED` packet. It repairs
the official-hybrid R1 audit hold by retaining all 24 next-token logits vectors
and scoring them from a separate, frozen reader. The only eligible conclusion
is bounded artifact/topology plus finite-forward qualification; historical
4.49x speed, recall, quality and production claims remain forbidden.

## Operational baseline at handoff

- Gateway health on port 8080: HTTP 200.
- Embedding health on port 8081: HTTP 200.
- The physical wave completed and restored the initially resident qualified
  route without service restart; exact hardware identity remains in each raw
  receipt rather than this prose handoff.
- No Windows Python process matching the launcher or watcher was active.
- Four closeout packets are independently promoted with bounded claims; seeded
  stability remains `EXECUTED` under a physical-identity hold.

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
