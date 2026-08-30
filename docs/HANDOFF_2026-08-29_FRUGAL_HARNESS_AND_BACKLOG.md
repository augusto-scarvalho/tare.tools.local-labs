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

The post-audit portfolio review did not rewrite any scientific result. It
admitted three bounded successors through the canonical pipeline and changed
only priority metadata. The subsequent value-ordered watched wave added two
minimal recovery successors. Direct state edits remain forbidden. Live
validation on 2026-08-29 found 122 records:

| State | Count |
| --- | ---: |
| `PROMOTED` | 30 |
| `REJECTED` | 22 |
| `EXECUTED` | 24 |
| `BLOCKED` | 43 |
| `IMPLEMENTED` | 2 |
| `VERIFIED` | 1 |

The 22 `EXECUTED` records include preserved HOLD/superseded evidence from prior
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

Independent review created only the smallest set of new experiments that can
materially change an ecosystem decision:

1. `BACKLOG-ADAPT06-SLOP-LIVE-06` (`P0`, score 89) is the current dependency-ready
   head. It repeats the positive live LoRA result with bound schedules, cache
   and logs plus route-correct isolated baselines. Its maximum claim is client
   affinity under the frozen route mix; it cannot establish server scheduling,
   fused-GEMM or production benefit.
2. `BACKLOG-SLX08-PHYSICAL-PREFILL-04` (`P0`, score 80) converts the independently
   verified offline fidelity result into an actual runtime OFF/ON comparison.
   It freezes 64 paired requests per arm, runtime telemetry, semantic
   noninferiority and a 1.10x TTFT gate. It does not inherit the historical
   1.40x proxy target.
3. `BACKLOG-GDN02-LEARNED-STATE-02` (`P1`, score 76) retains the three decisive
   vectors and all 147 collateral cosines so the negative learned-state result
   can be recomputed independently instead of resting on summaries.

`BACKLOG-HUMAN-JUDGE-CALIBRATION-01` scores 79 but remains `BLOCKED`: it has high
ecosystem value and no executable path until a blind human-label packet exists.
Lower-yield codec, guard, Hyper and negative-KV investigations remain behind
the three ready successors. Superseded attempts and families with an explicit
stop were moved to `P3`; no evidence or lifecycle state was deleted.

The value-ordered watcher/harness wave consumed that dependency-ready queue:

- ADAPT06 R6 aborted before service mutation because its inherited R5 helper
  assumed systemd port 8080 was a direct llama-server. R7 resolved the physical
  binary from the frozen fleet registry, bound the real gateway before/after,
  sealed 72 routed comparisons and advanced to `EXECUTED`. It achieved 100%
  route-correct parity and 93.10% requested-switch reduction, but its grouped
  schedule was slightly slower (`0.988x`); only the narrow client-ordering claim
  is audit-eligible.
- GDN02 R2 aborted before model load because a Windows `pathlib.Path` converted
  the WSL model path to backslashes. R3 used the literal POSIX path, retained
  three decisive cells and 147 collateral cosines, and independently reproduced
  every metric. It remains scientifically negative: leakage `12.24%` exceeded
  `5%` and fidelity `86.62%` missed `95%`; collateral retention was `99.97%`.
- SLX08 R4 is `BLOCKED`, not executed. Clean `slop.cpp` commit `34b3dac7c`
  contains DFlash and dense prefill but no callable selected-block prefill route
  or physical per-request telemetry. Labeling dense traffic as ON would be a
  false treatment. The exact unblock contract is in its packet.

The audit queue is
`config/research_audit_queue_2026-08-29_post_audit_value_wave.json`.

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

That closeout queue is historical context. After the post-audit watched wave,
`backlog_pipeline.py next --json` returns `null`: no dependency-ready
`PROPOSED` work remains.

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

Do not manufacture a next command from `EXECUTED` or `BLOCKED` items. ADAPT06
R7 and GDN02 R3 are audit-ready; SLX08 needs a separately reviewed physical
runtime implementation before it can re-enter the queue. The prior
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

`BACKLOG-SLX11-OFFICIAL-HYBRID-02` is independently `PROMOTED`. It repairs the
official-hybrid R1 audit hold by retaining all 24 next-token logits vectors in
an 11.9 MiB safetensors bundle and scoring them from a separate, frozen reader.
The auditor reopened that bundle directly and reproduced 24 exact BF16 tensors
of shape `[1, 248320]`, zero nonfinite values, every min/max/argmax/SHA
projection, 24 distinct tensor hashes, the official five-file checkpoint and
the physical 18 recurrent plus six full-attention topology. The constant
argmax token 271 in all 24 forwards reinforces the narrow boundary: claim
`SLX11_OFFICIAL_HYBRID_ARTIFACT_QUALIFIED_WITH_LOGITS_R2` establishes artifact
identity, topology and finite next-token smoke only. Historical 4.49x speed,
recall, generation quality, dense superiority and production claims remain
forbidden. Receipt SHA is `97a761d5...7196f`; review SHA is
`ad4a3979...44de1`; the full repository suite is 400 passing tests.

`BACKLOG-SLX08-REAL-FIDELITY-02` is contractually `REJECTED` with a confirmed
numerical false negative. Its physical run repeated the frozen two contexts by
six layers, retained 36 dense/corrected/legacy context tensors in a 298,304-byte
safetensors bundle and restored the serving baseline. Independent reopening
matched every tensor byte/hash/shape and recomputed corrected median fidelity
`0.9954493344`, but the frozen cross-device projection gate used absolute
tolerance `1e-9`. CUDA-versus-CPU cosine deltas were only zero to four FP32
ULPs (maximum `2.3841858e-7`), so the gate reported 1/12 despite identical
inputs. The gate was not relaxed after the result: R2 preserves that failure
and claim `SLX08_FIDELITY_FALSE_NEGATIVE_WITH_RETAINED_CONTEXTS_R2`. A frugal
successor may rescore the same retained bytes with a preregistered canonical
reduction; it needs no GPU inference. The claim remains offline 12-cell
fidelity only, never TTFT, runtime integration, production or quality.

`BACKLOG-SLX08-REAL-FIDELITY-03` is the independently `VERIFIED`, retained-only
successor. It binds the exact R2 receipt/review/bundle/samples, defines a
row-major float64 `math.fsum` cosine and runs the scorer twice in separate WSL
processes. Both canonical payloads hash to `b6f3124c...1ff0`; an auditor using
an independent binary safetensors parser reproduced exactly 36 float32 tensors,
zero nonfinite values, all tensor hashes, all 24 cosines, corrected median
`0.9954494256753561` and legacy median `0.9918341527542236`. The service was
observed but untouched and the full suite is 416 passing tests. Claim
`SLX08_FIDELITY_FALSE_NEGATIVE_CANONICAL_RESCORE_R3` closes the numerical
diagnosis only. It is not new inference or a fresh sample, so it cannot expand
generalization and, as `proxy_realization`, can never become `PROMOTED`.

## Operational baseline at handoff

- Gateway health on port 8080: HTTP 200.
- Embedding health on port 8081: HTTP 200.
- The physical wave completed and restored the initially resident qualified
  route without service restart; exact hardware identity remains in each raw
  receipt rather than this prose handoff.
- No Windows Python process matching the launcher or watcher was active.
- Four closeout packets, seeded-stability R4 and SLX11 R2 are independently
  promoted with bounded claims; SLX08 R3 is independently verified as an
  offline proxy; the superseded seeded-stability R2/R3 packets remain
  `EXECUTED` as immutable hold evidence.

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

## 2026-08-30 SLX08 physical relevance-prefill closeout

`BACKLOG-SLX08-RELEVANCE-PREFILL-11` is independently `PROMOTED` with claim
`SLX08_RELEVANCE_SELECTED_PREFILL_QUALIFIED_R11`. It supersedes the blocked R10
packet and the earlier physical attempts without rewriting them.

The physical `slop.cpp` experiment route accepts an optional ordered list of
block indices only when `slx08_selected_block_prefill=true` and the environment
gate is active. It requires exactly half of the prompt blocks, strict increasing
order, valid indices and retention of the first and final blocks. Responses bind
the exact selected indices and selection mode. This is token deletion before
ordinary dense prefill, not a sparse-attention kernel.

R11 executed 126 new prompt IDs, balanced as nine cases at every one of 14
evidence positions and 42 cases in each arm-order period. Its 378 physical rows
recompute to dense/naive/relevance accuracy `126/54/126`; relevance retained the
target in 126/126 and naive was correct exactly when its fixed positional policy
retained the target. Relevance evaluated 2048 rather than 4096 prompt tokens,
improved p50 TTFT `1.79838x`, improved marginal p95 `1.83200x`, and was faster in
126/126 dense pairs. The one-sided exact 95% lower delta bound is
`-0.0234952389`, passing the frozen `-0.03` margin.

Independent review SHA-256 is
`5088bbfc14919b09d0cf3d035cefc0fb6a77810aa3d9c4015d38dadba5d7f1d5`.
The receipt SHA-256 is
`2d32c1fce9149d9cee46eeea725a0065c009e1287ec51f45c098bd846af8071f`.
The watcher sealed 378/378 rows, advanced only to `EXECUTED`, and the independent
auditor performed `EXECUTED -> VERIFIED -> PROMOTED`. The full suite passed
494 tests and the pipeline gate passed. Qwen3.8 was restored on port 8080 and
embeddings remained healthy on 8081.

Do not generalize the promotion to production, learned retrieval, server-side
semantic selection, selector-inclusive latency, other models/context sizes,
RAG quality or generic sparse attention. The next useful successor should
measure selector cost and test a genuinely semantic selector on natural long
documents; another exact-key panel is not warranted.
