# Local inference lab handoff — 2026-08-23

This handoff is superseded for current live state and next work by
[`HANDOFF_2026-08-24_FABLE_HAUHAUCS_SERVING.md`](HANDOFF_2026-08-24_FABLE_HAUHAUCS_SERVING.md).
It remains the authoritative campaign/cleanup receipt for the state captured
below. Historical measurements remain authoritative only for the exact
artifact/build/runtime tuple they record.

## 1. Executive state

- The authorized dependency-free **non-soak** backlog was executed through its frozen gates.
- There is no unconditional GPU experiment ready to launch. Every residual research item has a named
  external dependency, a missing human input, or a requirement for a new falsifiable hypothesis.
- Reliability soaks are explicitly excluded. The cancelled 24-hour observations are incomplete and are
  not a PASS; the 48/72-hour successors were not run.
- The incumbent Qwen3.8 deployment decision is unchanged. The generation service is currently stopped;
  the embedding service on port 8081 is healthy and must not be stopped as collateral damage.
- `slop.cpp` is the canonical home for engine source, runtime flags, builds, and qualification tooling.
  `tare.tools.local-labs` owns experiment design, raw receipts, statistics, and RTX 3090 promotion decisions.
- 238.17 GB of rejected, superseded, redundant, or rebuildable local model artifacts were removed after
  exact-path and process checks. The retained research/deployment fleet is intact.

Primary reading order:

1. This handoff.
2. [`AUTONOMOUS_CAMPAIGN_REPORT_2026-08-23.md`](research/AUTONOMOUS_CAMPAIGN_REPORT_2026-08-23.md).
3. [`REMAINING_EXPERIMENTS_2026-08-22.md`](research/REMAINING_EXPERIMENTS_2026-08-22.md).
4. [`BACKLOG_V2_STATUS.md`](research/BACKLOG_V2_STATUS.md).
5. Exact `RESULT.md`, `DECISION_PACKET.md`, and raw receipts for the track being discussed.
6. The historical [`2026-08-21 handoff`](HANDOFF_2026-08-21_QWEN38_REQUAL_AND_NEXT_WAVE.md) only for
   earlier context and commands that have not been superseded here.

## 2. User intent and binding operating posture

The user authorized autonomous execution of the experiment backlog from start to finish and allowed the
idle generation service to be stopped for controlled LAB work. The later clarification is equally binding:

- run the non-soak backlog sequentially through dependency gates;
- do **not** run, resume, extend, or relabel 24/48/72-hour soaks without a new explicit instruction;
- use cheap discriminating gates before expensive suites;
- preserve raw receipts, invalid attempts, provenance, and displaced conclusions;
- label superseded evidence instead of silently replacing it;
- protect the independent embedding endpoint on port 8081;
- prefer open-weight candidates that genuinely fit an RTX 3090, but do not promote breadth for its own sake;
- engine implementation belongs in `slop.cpp`; local-labs should not become a second engine source tree;
- no sarcastic/anti-policy manifesto change was retained. The attempted direction was abandoned and the
  repositories contain only the ownership, runtime, evidence, and CI work described below.

Autonomy does not waive prerequisites. When the residual queue says `BLOCKED`, wait for the named trigger
or prepare a new falsifiable packet; do not fill the time with an unregistered broad benchmark.

## 3. Live machine and service state

Captured on 2026-08-23 in Windows PowerShell with WSL distribution `Ubuntu-24.04`.

| Component | Current state |
|---|---|
| GPU host | RTX 3090 24 GB; Windows 11 + WSL2 Ubuntu 24.04 |
| Generation unit | `llm-inference.service`: `inactive`, still `enabled` |
| Canonical generation model | `/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf` |
| Canonical context policy | 131,072 for exclusive SERVE; 81,920 when a task requires the measured 4 GiB reserve |
| Embedding endpoint | PID 413264, port 8081, `{"status":"ok"}` |
| Embedding artifact | `/home/augus/models/embedding/nomic-embed-text-v1.5.Q8_0.gguf` |
| WSL model filesystem | 1007 GiB total, 594 GiB used, 363 GiB available, 63% used |
| `/home/augus/models` | 503 GiB after cleanup |
| Host `C:` free | 254,950,301,696 bytes (237.44 GiB) after cleanup |

The only running `llama-server` at capture was the embedding process:

```text
/home/augus/src/slop.cpp/build/bin/llama-server
  -m /home/augus/models/embedding/nomic-embed-text-v1.5.Q8_0.gguf
  --host 0.0.0.0 --port 8081 --ctx-size 32768 --parallel 8
  --embedding --pooling mean
```

Do not kill a child process to stop canonical generation. Use `systemctl` on `llm-inference.service`,
because the unit is configured to restart its child. Before and after any LAB tranche, capture unit state,
process argv, ports, GPU state, and health.

## 4. Repository, branch, remote, and CI state

### 4.1 `tare.tools.local-labs`

- Path: `C:\projects\tare.tools.local-labs`
- Branch at handoff preparation: `master`
- Remote: `https://github.com/augusto-scarvalho/tare.tools.local-labs.git`
- Remote `origin/master`: `4db671b` (`docs(engine): establish slop.cpp ownership boundary`)
- Local cleanup commit: `2b75d00` (`docs(ops): record model disk cleanup`)
- Before this handoff commit, local `master` was one commit ahead of `origin/master` and otherwise clean.
- PR #3, `feat(lab): publish autonomous RTX 3090 experiment campaign`, merged on 2026-08-23.
- Latest remote `local-labs-ci` on `4db671b`: success, run `32619399203`.
- Equivalent local checks after cleanup: `compileall` PASS and `pytest` 17/17 PASS.
- This handoff and its pointer update are intentionally local until the user explicitly requests another push.

### 4.2 `slop.cpp`

- Path: `C:\projects\slop.cpp`
- Branch: `main`
- HEAD and `origin/main`: `71676e46c` (`ci: report disabled CANN check cleanly`)
- Worktree: clean and synchronized at capture.
- Remote: `https://github.com/augusto-scarvalho/slop.cpp.git`
- Open or historical PRs in this fork: none returned by `gh pr list --state all`.
- Fork Python Type-Check: success on `b1683da58`, run `32619518431`.
- CANN workflow: success on current `71676e46c`, run `32619551770`.
- A `Publish Docker image` run for older commit `87a416bd7` was still queued (`32618786054`). It is not
  evidence of a regression in current HEAD and should be inspected only if image publication matters.
- Older inherited upstream workflows include failures/skips on prior commits. The fork-specific checks that
  were changed in this continuation are green; do not claim every inherited llama.cpp platform matrix was
  requalified by these small documentation/CI commits.

## 5. What was completed

### 5.1 Autonomous RTX 3090 experiment campaign

The campaign report contains the complete prose account. The operational decisions are summarized here:

| Track | Qualified result | Consequence |
|---|---|---|
| Qwen3.8 current Unsloth revision | Current IQ4_XS and Q2_K_XL rejected for supersession | Keep the historical qualified artifacts; freshness alone earned no upgrade |
| Muse Glimmer | VQA specialist 107/150; safety 5/5 bounded; overall HOLD | Retain text+vision specialist, reject DFlash and general-role promotion |
| Cold Fusion | Base rejected; nine-cell embedded MTP arm rejected | No base or speculative deployment role |
| Resident breadth | Mistral, two Gemma variants, GPT-OSS, and Ornith all HOLD | Useful 3090 evidence, no general replacement |
| RWKV7 | Mechanism qualified, deployment blocked by weight-license posture | Keep research-local until publisher assertion and serving packet exist |
| Falcon-H1R | HOLD_ROLE | Fit/tools/GSM useful; coding non-termination blocks broader role |
| RetNet | BLOCKED_UPSTREAM | No official pretrained checkpoint to qualify |
| Qwen-Image | 10/13, deterministic replay, QUALITY HOLD | Retain as image research candidate |
| SDXL | About 9.5x faster, 3/13, QUALITY REJECT | Retain only as a fast comparison baseline |
| Gemma-4-12B Vision | 4/4 fixtures, 20/20 clauses | Bounded screenshot-analysis role qualified |
| Agent robustness | Corrected stress 16/16; policy 16/16; order weakness isolated | Promote the idempotent status-check policy, not model invariance |
| Harness product | 83.85% token reduction; 5/5 recall; 6/7 mutations; critic 8/8 | Bounded fail-closed evidence-pack primitives qualified |
| BigCodeBench-Hard | 48/147 adjusted | Harder coding baseline established |
| Mini SWE-bench pilot | 5/10; all five submitted patches resolved | Empty-patch/call-exhaustion remains the limiting failure mode |
| Loop remediation | Prompt-only and duplicate-command middleware negative | Do not spend more on repeat-command blocking without a new mechanism |
| Context | RULER bounded 128k pass; RepoBench-P 39.56 fail | Retrieval capacity does not imply useful repository completion |
| Cache | No-spec persistence pass; intermittent MTP restoration failure | MTP remains ephemeral-only for promotion purposes |
| Engine parity | SGLang/vLLM about 18% ahead on fresh prefill; decode unresolved | llama.cpp retains lower VRAM/startup advantages; no universal winner |
| Serving isolation | Reproducible CUDA illegal-memory-access path | Closed-loop TPOT conclusion blocked |
| Power | No reduced cap met the frozen 95% throughput rule | Retain 420 W |
| Colocation | Same-GPU contender materially increased energy | Keep canonical serving isolated from same-GPU experiment/judge work |
| Context operations | 131,072 retained; 81,920 named reserve profile | No live default mutation required |
| Provenance | 33 GGUFs inventoried and full-hashed; one TC MTP mismatch isolated | Identity claims are traceable and UNKNOWN remains explicit |
| Authorial requant | Construction provenance closed; parity rejected | Reproducibility is not behavioral parity |
| A2 Stage 2 | 0/44 directions passed; G0 KILL | No edit, conversion, or merge was built |
| Fable termination | Instruct bounded-safe; thinking agentic role disqualified | Never mask empty final content with `finish_reason=stop` |
| no-mmap close-out | Residual decode effect confounded | No new operational promotion |

### 5.2 Reliability-soak correction

- The mistakenly launched fresh 24-hour run was stopped immediately after the user clarified the exclusion.
- It ended after 8/8 operations with zero observed health failures, but that is partial evidence only.
- It must remain `CANCELLED_BY_USER`, incomplete, and not a PASS.
- The queued 48/72-hour sequencer was stopped before either successor began.

### 5.3 `slop.cpp` ownership and qualification

Work completed in `slop.cpp`:

- `e4233c31c`: established the consumer-hardware fork identity in `README.md`.
- `87a416bd7`: added `docs/LEVERS.md` and `docs/AGENT_MAINTENANCE_GUIDE.md`.
- `50681c664`: centralized fork qualification in `tools/scripts_sh/bless_fork.sh` and
  `tools/scripts_sh/verify_mtp.py`.
- The relocated qualification harness rechecked the historical build-10159 tuple at 3/3 PASS. This is
  tuple-specific evidence, not blanket qualification of newer engine commits or models.
- `648ce0e7f`, `b1683da58`, and `71676e46c`: repaired/scoped the fork-specific Python and disabled-CANN
  workflows until both affected checks reported success.

Work completed in local-labs:

- `4db671b`: made the ownership boundary explicit in README/catalog/fork docs, converted local fork patches
  and harnesses into archival/evidence shims, and retained the exact 3/3 qualification receipt.
- Engine code and maintenance instructions now point to `slop.cpp`; local-labs retains experiment evidence.
- No anti-AI contribution manifesto was added. The inherited upstream issue-template wording remains
  upstream material; it is not the policy or identity of this fork.

### 5.4 Documentation, integration, and CI

- `6a5ec02`: consolidated the campaign, decisions, receipts, canonical remaining queue, and report.
- `8073773` + `5339b64`: added deterministic repository CI and fixed its dependency installation path.
- PR #3 merged the campaign into `master`; subsequent ownership reconciliation also reached `master`.
- Local-labs deterministic CI is CPU-only by design. GPU findings are represented by immutable receipts and
  must not be silently re-executed or generalized by CI.

### 5.5 Model disk cleanup

The exact receipt is [`disk_cleanup_2026-08-23.md`](campaigns/serving/disk_cleanup_2026-08-23.md).

Removed 238,166,467,040 bytes from ten exact targets:

- rejected current-revision Qwen3.8 candidates and their unused separate draft;
- rejected Cold Fusion candidate;
- closed official-source/BF16/authorial-requant build chain;
- redundant Bartowski Q4_K_M;
- completed one-shot Qwen3-4B engine-parity artifact;
- Q3_K_XL, IQ3_XXS, and failed-cliff IQ2_M frontier intermediates;
- only the rejected Muse DFlash draft, while retaining Muse text and `mmproj`.

Deletion was irreversible locally. Upstream artifacts are redownloadable and the authorial chain is
rebuildable from retained revision, source-verification, hash, imatrix, quantizer, and result receipts.

Deliberately retained:

- Qwen3.8 `UD-Q4_K_XL`, `IQ4_XS`, and low-footprint `UD-Q2_K_XL`;
- embedding model and endpoint;
- Muse text+vision, Qwen-Image, SDXL, Gemma Vision, RWKV7, Falcon, and the HOLD breadth fleet;
- promoted `fable-tc-l1.0`, Fable/ThinkingCap research artifacts, and FP16 authorial parents;
- all Git-tracked experiment and provenance receipts.

## 6. Remaining backlog and exact unblock triggers

There is no dependency-free non-soak experiment ready for unconditional execution.

| Priority when unblocked | Item | Current blocker | Required trigger before execution |
|---:|---|---|---|
| 1 | MTP persistence root cause | Intermittent restored-state correctness failure has no sufficiently specific mechanism packet | A new falsifiable cache-lifecycle hypothesis with invariant controls and invalidation rules |
| 2 | ThinkingCap Qwen3.8 | No official Qwen3.8 ThinkingCap weights | Official release plus a 3090-fit artifact |
| 3 | RWKV7 deployment | Publisher does not assert a weight license; no serving-quality packet | Publisher-asserted license posture plus frozen serving/quality qualification |
| 4 | ThinkingCap MTP identity | Local 17,221,641,152-byte digest differs from revision `f015d8b` metadata | Publisher/download receipt identifying the local content; never overwrite or infer identity |
| 5 | Third-party quantizer provenance | 31 third-party model cards do not disclose exact quantizer/llama.cpp builds | Publisher build receipts; hash only newly admitted or promotion-relevant downloads |
| 6 | Human-judge calibration | Model labels are not human calibration | 50–100 frozen blind human preference labels |
| 7 | RetNet | No official pretrained checkpoint | Official upstream checkpoint release |
| Excluded | LAB-REL-001/002 soaks | Explicit user exclusion | A new explicit direction that specifically reopens soaks |

Parked, not automatically executable:

- the serving closed-loop TPOT matrix while the CUDA illegal-memory-access path remains unresolved;
- open-ended training/distillation and custom CUDA kernels without a measured bottleneck;
- distributed serving, Kubernetes, cloud/product integration, and sub-4-bit KV work;
- additional task-oriented Q3/mixed quantization, superseded by the completed seven-quant frontier;
- more candidate breadth without a concrete role discriminator stronger than the completed compact gates.

## 7. Observations and recommendations for the next agent

1. **The next high-value work is mechanism-led, not benchmark-led.** A concrete MTP persistence hypothesis
   would unlock the most operationally relevant local blocker. Repeating fresh-session MTP speed tests will
   not answer the persistent-state failure.
2. **Do not replace the incumbent for novelty.** Several fresh/open models fit the 3090, and some win a narrow
   gate, but none cleared the general-role evidence packet. Maintain role-specific HOLD decisions.
3. **Muse is genuinely useful but narrow.** Its local VQA result is the strongest measured comparison, while
   its agent/code/cache/DFlash/full-stack failures still block general deployment.
4. **Keep image conclusions asymmetric.** Qwen-Image is the quality research candidate; SDXL is a fast
   baseline. Neither is a production promotion.
5. **Treat engine results as tuples.** A build/model/quant/context/offload change invalidates blanket reuse of
   the 3/3 blessing or a throughput number. Re-run the engine-owned harness on the exact candidate tuple.
6. **Do not confuse evidence retention with disk retention.** Git receipts are canonical. Re-downloadable
   weights may be removed after a closed decision, but active, HOLD-specialist, blocked-dependency, and
   authorial-source artifacts should receive an explicit keep/delete rationale.
7. **The old master handoff contains stale production language.** Its historical tables are useful context,
   but this file, the campaign report, and per-run decisions control current status.
8. **Repository CI cannot validate GPU claims.** A green CPU CI means the harness/docs/tests are coherent;
   it does not requalify a model, CUDA path, or serving profile.
9. **The fork should stay opinionated without rewriting upstream history.** Add local identity and maintenance
   guidance in fork-owned files; avoid turning inherited contribution templates into the fork manifesto.
10. **No local VHDX compaction was performed.** Deletion freed space inside WSL and current host free space is
    recorded. Any future compaction must account for stopping WSL, the embedding service, and Docker safely.

## 8. Safe resume checklist

Run from PowerShell unless a command is explicitly WSL-only:

```powershell
Set-Location C:\projects\tare.tools.local-labs
git status -sb
git log -3 --oneline

Set-Location C:\projects\slop.cpp
git status -sb
git log -3 --oneline

wsl -d Ubuntu-24.04 -- systemctl is-active llm-inference.service
wsl -d Ubuntu-24.04 -- curl -fsS http://127.0.0.1:8081/health
wsl -d Ubuntu-24.04 -- pgrep -af llama-server
wsl -d Ubuntu-24.04 -- nvidia-smi
```

Before launching a newly unblocked experiment:

1. identify the exact backlog item and unblock evidence;
2. freeze question, factors, controls, gates, early-stop rules, artifact SHA, engine SHA, template SHA, and
   runtime vector in a decision packet;
3. capture baseline service/process/GPU state;
4. stop `llm-inference.service` through `systemctl` if GPU isolation is required;
5. leave port 8081 alone;
6. run cheap admission gates first and stop on a preregistered failure;
7. retain raw outputs incrementally and distinguish wrong, truncated, empty, crashed, and non-terminating;
8. stop all experimental processes and verify ports/GPU afterward;
9. restore canonical generation only when the task calls for it, then verify health and exact argv;
10. update the canonical queue and mark displaced claims `SUPERSEDED`.

For WSL scripts containing shell variables, stage a Linux script, strip CRLF, and inspect it with `cat -A`
before execution. Prefer `$HOME/.local/bin/hf` inside WSL. Long jobs need a persistent background harness,
PID/log paths, and launch verification; reconstructing a command is not evidence that it ran.

## 9. Commit map

### Local-labs campaign and integration

- `6a5ec02` — close and document the autonomous RTX 3090 campaign.
- `8073773` — add deterministic repository CI.
- `5339b64` — repair CI dependency installation.
- `6de459b` — merge PR #3 into `master`.
- `4db671b` — establish the `slop.cpp` ownership boundary and preserve the qualification receipt.
- `2b75d00` — record the 2026-08-23 model cleanup.
- This handoff commit — update the canonical operator entrypoint and freeze final state.

### `slop.cpp`

- `e4233c31c` — establish fork identity.
- `87a416bd7` — add authorial levers and autonomous maintenance guidance.
- `50681c664` — centralize runtime qualification in the engine repository.
- `648ce0e7f` — keep the disabled CANN workflow structurally valid.
- `b1683da58` — scope Python checks to fork tooling.
- `71676e46c` — make the disabled CANN check report cleanly.

## 10. Clean stopping condition

At the end of the next continuation, leave:

- no experimental server or downloader without a PID, command, and log receipt;
- embedding port 8081 healthy unless an explicitly scoped task requires otherwise;
- generation service state intentional and reported;
- no soak running without a new explicit user instruction;
- no stale claim that partial reliability evidence is a PASS;
- raw receipts and invalid attempts preserved;
- exactly one current backlog disposition for every touched item;
- repo worktrees reported accurately, including any local commits not pushed;
- engine implementation changes in `slop.cpp` and RTX 3090 conclusions in local-labs.
