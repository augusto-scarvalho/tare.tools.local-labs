# Handoff to AGY — execution plans for the guarded backlog

Date: 2026-08-25  
Expected executor: AGY / Gemini 3.7 Flash High  
Repository: `C:\projects\tare.tools.local-labs`  
Canonical queue: `config/research_backlog.json`

## Purpose

This is the operating handoff for invoking AGY manually, one backlog item at a
time. It translates every current item into an execution plan. AGY is the
executor, not the reviewer: it may take an eligible item through
`PREREGISTERED`, `IMPLEMENTED` and `EXECUTED`, then must stop for Codex review.

The repository currently contains uncommitted pipeline work. AGY must preserve
all pre-existing worktree changes and must not commit, push, discard, amend or
rewrite them.

## Mandatory reading and bootstrap for every invocation

Read, in order:

1. [`GEMINI.md`](../GEMINI.md)
2. [`HANDOFF_2026-08-25_CODEX_REMEDIATION.md`](HANDOFF_2026-08-25_CODEX_REMEDIATION.md)
3. [`BACKLOG_IMPLEMENTATION_PIPELINE.md`](research/BACKLOG_IMPLEMENTATION_PIPELINE.md)
4. [`research_backlog.json`](../config/research_backlog.json)
5. The item-specific source artifacts recorded in the manifest

Then run:

```powershell
Set-Location C:\projects\tare.tools.local-labs
python tools/analysis/backlog_pipeline.py gate
python tools/analysis/backlog_pipeline.py status
python tools/analysis/backlog_pipeline.py next
git status --short
```

Stop if `gate` fails. Unless the user explicitly selects a different eligible
item, execute only the ID returned by `next`. Never infer active work from old
receipts or run directories.

## Shared executor protocol

For an eligible `PROPOSED` item:

```powershell
python tools/analysis/backlog_pipeline.py scaffold <ITEM_ID> --actor "AGY / Gemini 3.7 Flash High"
```

Fill the generated `PRE_REGISTRATION.md` completely before writing the runner
or executing the experiment. Freeze exact inputs/hashes, hypothesis, controls,
factors, sample size, command, abort conditions, thresholds and claim boundary.

```powershell
python tools/analysis/backlog_pipeline.py advance <ITEM_ID> --to PREREGISTERED --actor "AGY / Gemini 3.7 Flash High"
```

Implement the smallest runner and deterministic tests that answer the frozen
question. Run focused tests and the repository gate. Bind every implementation
and test file:

```powershell
python tools/analysis/backlog_pipeline.py advance <ITEM_ID> --to IMPLEMENTED --actor "AGY / Gemini 3.7 Flash High" `
  --implementation <RUNNER_PATH> `
  --implementation <TEST_PATH>
```

Execute only the preregistered command. Start from `RECEIPT.template.json` and
write the final receipt to `raw/receipt.json`. Use
`tools/analysis/experiment_provenance.py`, hash all decisive inputs and retain
raw samples. The pipeline, not AGY, recomputes each frozen gate.

```powershell
python tools/analysis/backlog_pipeline.py advance <ITEM_ID> --to EXECUTED --actor "AGY / Gemini 3.7 Flash High"
python tools/analysis/backlog_pipeline.py gate
```

Write `RESULT.md`, clearly recording passed and failed gates, scope limits and
all non-actions. Stop at `EXECUTED`. Do not create `REVIEW.json`, run a
`VERIFIED`/`PROMOTED` transition, start the next item, commit or push.

For a `BLOCKED` item, do not scaffold it. First prove its named unblock
condition. If the condition is absent, return a bounded reconnaissance report
without changing backlog state. Only after the condition is objectively true:

```powershell
python tools/analysis/backlog_pipeline.py advance <ITEM_ID> --to PROPOSED --actor "AGY / Gemini 3.7 Flash High" --reason "<exact evidence that removed the blocker>"
```

Then rerun `gate`, `status` and `next`. Do not bypass dependency ordering.

---

## Plan 1 — `BACKLOG-ADAPT-REQUAL-01`

Priority/state: P0, `PROPOSED`  
Goal: requalify saved `ADAPT-01A` through `ADAPT-05` artifacts without claiming
that their training is reproducible.

### Plan

1. Inventory every saved adapter and config. Produce a deterministic artifact
   ledger with path, bytes and SHA-256; the frozen gate expects 13 identified
   artifacts. Confirm that no excluded tokenizer copy is mistaken for unique
   model evidence.
2. Identify and hash the exact base model, tokenizer/template, math panel, QA
   panel and both scorers. Fail closed if an identity cannot be recovered.
3. Freeze one base-model control plus every loadable adapter arm, a common seed,
   decoding contract and per-sample output schema. Do not use historical scores
   as new observations.
4. Implement an offline-first requalification runner. It must verify clean
   adapter load, capture raw generations and score the same frozen inputs for
   every arm. Keep generation and deterministic scoring separable so Codex can
   reproduce the scores independently.
5. Execute at least 32 scored math samples and 16 scored QA samples per arm.
   Record load failures as failures, not silently dropped arms.
6. Write the artifact/dataset/scorer ledgers, raw samples, score table and
   provenance into the new packet. The only positive claim code is
   `ARTIFACT_REQUALIFIED`; otherwise report `ARTIFACT_REJECTED`.

### Frozen completion gates

- exactly 13 hashed adapter/config artifacts;
- at least 32 scored math samples per arm;
- at least 16 scored QA samples per arm;
- base-model control present;
- independently reproducible scorer result.

### Forbidden conclusions

No training reproducibility, production promotion or general-capability claim.
Do not retrain any adapter in this item.

---

## Plan 2 — `BACKLOG-ADAPT-TRAIN-01`

Priority/state: P1, `BLOCKED`  
Dependency: `BACKLOG-ADAPT-REQUAL-01` must be independently reviewed and
`PROMOTED`, with at least one explicitly named finalist.

### Unblock check

1. Read the P0 `RESULT.md`, receipt and independent review.
2. Confirm the promoted claim identifies the exact finalist artifact/hash.
3. Confirm the TRAIN-00B resource envelope is acceptable for a fresh run.
4. If any check fails, leave this item `BLOCKED` and report the missing fact.

### Plan after unblocking

1. Create a brand-new output root and assert it contains zero pre-existing
   output files. Never resume or count a historical checkpoint as a new run.
2. Freeze base model, tokenizer, dataset revision/splits, preprocessing,
   optimizer, adapter geometry, seeds, step budget, evaluation intervals and
   abort thresholds. Use only P0 finalists.
3. Implement training plus a clean-reload evaluator. Capture loss/resource
   traces and hash every checkpoint. At least two successful repeated seeds are
   required.
4. Evaluate base, frozen saved finalist and freshly trained artifacts on the
   same held-out and protected panels. Separate training success from behavioral
   gain.
5. Preserve failed seeds and partial traces. Never replace them with a later
   successful attempt.

### Frozen completion gates

- zero pre-existing files in the fresh output root;
- at least two successful repeated seeds;
- held-out gain over base greater than zero;
- protected-set regression no greater than 5%.

### Claim boundary

Only `TRAINING_REPRODUCED` or `TRAINING_REJECTED`. A single seed, artifact
identity or decreasing train loss cannot establish reproducibility.

---

## Plan 3 — `BACKLOG-DISTILL-REAL-01`

Priority/state: P1, `PROPOSED`  
Goal: rebuild invalid `DISTILL-00` using actual teacher and student generations.

### Plan

1. Treat the old DISTILL-00 result as a superseded hypothesis source only.
   Inventory every field previously hard-coded or randomly generated.
2. Freeze exact teacher/student model or endpoint identities, prompts, dataset
   split/hash, chat templates, decoding parameters, seed, maximum tokens and
   scorer implementation/hash.
3. Freeze a paired panel of at least 32 samples. Each item must retain the exact
   prompt, raw teacher output, raw student output, token accounting and strict
   score. Missing/empty generations remain in the denominator.
4. Implement generation and scoring so all aggregate numbers are derived from
   raw pairs. Reject any decisive constant that cannot be recomputed.
5. Run the teacher and student under the frozen contract, then compute paired
   accuracy delta and median reasoning-token reduction.
6. Preserve endpoint errors, truncations and retries. Do not replace samples
   after seeing their scores unless the preregistered retry policy requires it.

### Frozen completion gates

- every score derived from preserved raw samples;
- at least 32 paired scored samples;
- student accuracy delta at least -0.03 versus teacher;
- median reasoning-token reduction at least 20%.

### Claim boundary

Only `DISTILLATION_QUALIFIED` or `DISTILLATION_REJECTED`. Do not claim teacher
superiority or distillation success from unpaired, random or hard-coded data.

---

## Plan 4 — `BACKLOG-CUDAGRAPH-SERVING-01`

Priority/state: P2, `PROPOSED`  
Goal: validate CUDA Graph replay inside the actual serving runtime, not only in
the SLX-05D microbenchmark.

### Plan

1. Identify the exact serving integration point, effective model/build/argv,
   request path, batch/slot behavior and mutable cache state. Do not begin from
   a theoretical or isolated kernel path.
2. Capture the live baseline before mutation: systemd unit/drop-ins, PID,
   restart count, health, effective route, GPU state and embedding service.
3. Freeze a paired A/B design using identical real requests: baseline runtime
   versus the serving candidate with graph replay. Freeze warmup, request order,
   repetitions, concurrency, token lengths and timing boundaries.
4. Implement integration plus strict semantic/state oracles. Reuse the full
   hybrid-cache restoration lesson from SLX-05D; compare responses and retain
   raw per-request timing/resource samples.
5. Enter LAB mode via `systemctl` only. Never kill a `Restart=always` child.
   Leave embedding port 8081 untouched and use explicit experiment ports.
6. Run at least 30 paired observations unless the preregistered design sets a
   stronger count. Exercise abort, normal completion and post-run canaries.
7. Restore the original unit/drop-ins, route, model and health. Record new PID,
   restart count and both service health checks before closing.

### Frozen completion gates

- zero response mismatches;
- paired wall-time speedup p50 at least 1.15x;
- no p95 latency regression;
- PID/restart identity and service health restored.

### Claim boundary

Only `SERVING_CUDAGRAPH_QUALIFIED` or `SERVING_CUDAGRAPH_REJECTED`. Do not claim
exclusive launch overhead, a persistent-megakernel ceiling or production gain
from the old microbenchmark alone.

---

## Plan 5 — `BACKLOG-PROXY-REALIZATION-01`

Priority/state: P2, `BLOCKED`  
Goal: materialize exactly one formerly proxy-only systems candidate as a real
runtime/kernel artifact with hardware measurements.

### Selection phase — remain `BLOCKED`

1. Inventory candidate proxies from the remediation report. Exclude candidates
   already falsified by their cheapest discriminating mechanics gate.
2. Rank survivors by expected operational value, implementation cost, hardware
   fit, semantic risk and whether a paired baseline is available.
3. Recommend exactly one candidate, exact source module/runtime integration
   point and bounded implementation budget. Stop for the user's selection; do
   not silently select several candidates or start a portfolio wave.
4. After the user selects the candidate, record that selection and exact target
   as the unblock evidence, advance to `PROPOSED`, then scaffold.

### Realization plan

1. Freeze the proxy prediction only as a hypothesis. Define the real artifact,
   semantic parity oracle, paired hardware baseline, workload and abort gates.
2. Implement the real runtime/kernel path. Random tensors may be used for unit
   correctness only, never as decisive performance evidence.
3. Prove the artifact was actually exercised using runtime identity, compiled
   artifact hash, effective route or another candidate-specific execution
   receipt.
4. Run at least 30 paired hardware samples and retain all raw measurements.
5. Stop at `EXECUTED` with `REALIZATION_QUALIFIED` or
   `REALIZATION_REJECTED` as the bounded result code.

### Frozen completion gates

- real implementation artifact present and exercised;
- semantic parity passes;
- at least 30 paired hardware samples.

### Special promotion boundary

The `proxy_realization` evidence class is intentionally non-promotable. Even a
passing packet may only reach independent verification. Promotion requires a
separate, reviewed successor backlog item with the correct real evidence class;
AGY must not edit the class to bypass this rule.

---

## Plan 6 — `BACKLOG-PACKED-HARDWARE-01`

Priority/state: P3, `BLOCKED`  
Goal: validate an actual packed compression or sparsity artifact on hardware.

### Unblock check

1. Identify an actual packed codec, packed model, compiled sparse kernel or
   runtime-native sparse representation. A dequantized tensor, analytical byte
   estimate or zero mask is not an artifact.
2. Record its exact path, format, bytes, hash, builder/toolchain identity and
   compatible runtime. Confirm it can be loaded and exercised on the target
   hardware.
3. If no such artifact exists, leave the item `BLOCKED`. Creating a codec or
   kernel belongs to a separately scoped implementation item, potentially the
   reviewed successor from Plan 5.

### Plan after unblocking

1. Freeze the unpacked/control artifact and packed candidate, exact runtime,
   model/task panel, GPU identity, power/memory sampling method, warmup,
   repetitions and restoration procedure.
2. Verify packed bytes on disk and in the loaded/runtime representation. Record
   actual measured VRAM; do not derive it from bit width.
3. Run paired throughput/latency measurements on hardware and a frozen quality
   panel. Preserve failures, OOMs and unsupported-kernel fallbacks.
4. Prove the packed/sparse path was exercised rather than silently dequantized
   or dispatched to a dense fallback.
5. Restore the baseline and record service/GPU health if the serving runtime was
   touched.

### Frozen completion gates

- realized size reduction greater than zero;
- measured VRAM reduction greater than zero;
- paired throughput regression no greater than 5%;
- quality regression no greater than 3%.

### Claim boundary

Only `PACKED_ARTIFACT_QUALIFIED` or `PACKED_ARTIFACT_REJECTED`. No analytical
VRAM claim, dequantized-memory saving or zero-mask acceleration claim.

---

## Plan 7 — `BACKLOG-ADAPT-TRACE-DISTILL-01`

Priority/state: P2, `BLOCKED`  
Goal: revisit ThinkingCap trace distillation only after Plan 2 produces and
Codex promotes a reproducible behavioral finalist.

### Unblock check

1. Require `BACKLOG-ADAPT-TRAIN-01` in `PROMOTED`, with exact finalist and
   checkpoint hashes.
2. Require a new preregistered scale, trace-budget or curriculum hypothesis;
   repeating ADAPT-00C unchanged is not an unblock event.
3. If either fact is absent, keep the item blocked.

### Plan after unblocking

1. Freeze the promoted finalist, teacher, real teacher traces, student inputs,
   datasets, scorers and paired evaluation panel.
2. Implement actual trace acquisition and student distillation; no copied or
   synthetic decisive traces.
3. Retain at least 32 paired teacher/student traces and evaluate the distilled
   student against the promoted undistilled finalist.
4. Require positive held-out gain and no more than 5% protected regression.
5. Close with `TRACE_DISTILLATION_QUALIFIED` or
   `TRACE_DISTILLATION_REJECTED`; never claim production promotion.

---

## Plan 8 — `BACKLOG-MTP-PERSISTENCE-01`

Priority/state: P1, `BLOCKED`  
Goal: find the mechanism behind the intermittent restored-state `!` failure in
MTP cache persistence.

### Unblock check

AGY must first write a falsifiable cache-lifecycle hypothesis naming the exact
state transition, invariant controls, predicted failure signature and evidence
that would invalidate it. Clean reruns alone do not unblock the item.

### Plan after unblocking

1. Freeze the exact historical failing tuple and preserved LAB-CACHE-001 raw
   sequence; do not substitute a simpler fresh-session speed test.
2. Reproduce the original failure at least once under strict controls. If it
   cannot reproduce inside the frozen budget, reject the mechanism hypothesis.
3. Instrument target/draft KV, recurrent/convolutional state, slot save/restore,
   cancellation and invalidation boundaries without changing semantics.
4. Change one lifecycle factor at a time and keep no-spec plus fresh-session
   controls invariant.
5. For a proposed fix, require 20 successful repeated fixed-path runs, zero
   output mismatch and all invariant controls passing.
6. Preserve failed attempts as successors. Use `MTP_PERSISTENCE_ROOT_CAUSED`
   only when the failure, intervention and invalidation test form a causal chain.

---

## Plan 9 — `BACKLOG-THINKINGCAP-QWEN38-01`

Priority/state: P2, `BLOCKED`  
Goal: qualify an official ThinkingCap model based on Qwen3.8.

### Unblock check

Require official publisher weights, explicit revision/license identity and a
3090-fit artifact. Community conversions without official source binding and
Qwen3.6 ThinkingCap weights do not satisfy the trigger.

### Plan after unblocking

1. Freeze publisher revision, license, every shard/artifact hash, quantizer
   provenance, template and runtime build.
2. Run the cheap fit gate first; require at least 4 GiB free VRAM after load.
3. If fit passes, run the standing role, agent/tool, math, code, cache and
   bounded-termination gates against the current Fable/Qwen3.8 control.
4. Require the frozen role gate and at least 95% natural termination before any
   positive decision.
5. Record the exact role boundary. A narrow specialist result cannot replace
   the broad default.

---

## Plan 10 — `BACKLOG-THINKINGCAP-MTP-IDENTITY-01`

Priority/state: P3, `BLOCKED`  
Goal: resolve the identity of the legacy 17,221,641,152-byte local MTP artifact.

### Unblock check

Require a publisher or original-download receipt that names the exact local
SHA-256. A similarly named integrated-MTP release is not sufficient.

### Plan after unblocking

1. Recompute full local bytes/SHA-256 without modifying the artifact.
2. Archive the publisher/download receipt with retrieval time, source revision,
   stated bytes and digest.
3. Compare the receipt, pinned `f015d8b` metadata and local artifact in a
   machine-readable lineage table.
4. Have an independent process recompute the binding.
5. Close as `LEGACY_MTP_IDENTITY_RESOLVED` only on exact digest identity;
   otherwise retain `LEGACY_MTP_IDENTITY_UNRESOLVED`. Never rewrite metadata to
   make the values agree.

---

## Plan 11 — `BACKLOG-QUANTIZER-PROVENANCE-01`

Priority/state: P3, `BLOCKED`  
Goal: resolve exact third-party quantizer/llama.cpp build provenance where it is
promotion-relevant.

### Unblock check

Require exact publisher build receipts. A model card that only names the quant
type, imatrix or tool family remains `UNKNOWN_BUILD`.

### Plan after unblocking

1. Start from the LAB-PROV-001 inventory and select only newly admitted or
   currently promotion-relevant artifacts.
2. Preserve publisher receipts containing exact tool revision, command/options,
   source-model revision and imatrix identity where applicable.
3. Bind each receipt to the full local artifact hash. Do not bulk-redownload the
   historical fleet merely to create apparent progress.
4. Independently verify all resolved entries and retain `UNKNOWN` for every
   unsupported field.
5. Positive closure requires 100% exact build receipts and zero unresolved
   promotion-relevant artifacts; partial evidence closes only as
   `QUANTIZER_PROVENANCE_PARTIAL`.

---

## Plan 12 — `BACKLOG-HUMAN-JUDGE-CALIBRATION-01`

Priority/state: P2, `BLOCKED`  
Goal: calibrate automated judges against genuine blind human preferences.

### Unblock check

Require 50–100 genuine human labels. Model-generated, author self-labels or
unblinded labels do not satisfy the trigger.

### Plan after unblocking

1. Freeze prompts, anonymized/reordered response pairs, rubric, rater assignment
   and exclusion policy before collecting labels.
2. Retain rater provenance without leaking response/model identity during
   labeling. Separate collection from model-judge scoring.
3. Collect at least 50 blind labels and preserve disagreements/abstentions.
4. Report inter-rater agreement and compare each automated judge to the frozen
   human reference with confidence intervals.
5. Do not tune and evaluate a judge on the same examples without a frozen split.
6. Close `JUDGE_CALIBRATED` only with complete blind/rater provenance;
   otherwise use `JUDGE_CALIBRATION_INSUFFICIENT`.

---

## Plan 13 — `BACKLOG-RETNET-OFFICIAL-01`

Priority/state: P3, `BLOCKED`  
Goal: qualify an official pretrained Microsoft/TorchScale RetNet checkpoint.

### Unblock check

Require an official Microsoft/TorchScale pretrained checkpoint with license and
revision. Construction code or community weights cannot substitute for it.

### Plan after unblocking

1. Freeze source revision, license, shard hashes, tokenizer and runtime identity.
2. Run fit first and require the standing 4 GiB reserve.
3. Verify recurrent-state mechanics and the frozen retention panel, then open
   broader role gates only if mechanics pass.
4. Compare against the retained recurrent and transformer controls on identical
   prompts and budgets.
5. Require both retention and role-quality gates for
   `RETNET_OFFICIAL_QUALIFIED`; a mechanism-only pass remains non-deployment.

---

## Plan 14 — `BACKLOG-BEE-L2-KV-CODEC-01`

Priority/state: P3, `BLOCKED`  
Goal: execute the completed BEE-L2 design against a real immutable physical KV
codec.

### Unblock check

Require a physical codec candidate with immutable format/hash and effective
backend-route receipts. An analytical representation or requested flag is not
a candidate.

### Plan after unblocking

1. Reuse the staged pack and full-distribution scorer from the completed BEE-L2
   design; freeze the exact candidate tuple and uncompressed control.
2. Prove pack/unpack identity, physical bytes and that the live backend route
   exercised the codec rather than a fallback.
3. Run the cheap distribution gate first. Stop on a preregistered failure.
4. Only survivors open retrieval, task-quality and paired hardware stages.
5. Preserve full score distributions, retrieval outputs, task samples and
   hardware metrics.
6. Qualify only if physical format, route, distribution, retrieval and task
   gates all pass; partial codec mechanics are not a promotion.

---

## Plan 15 — `BACKLOG-APEX4-E2E-01`

Priority/state: P3, `BLOCKED`  
Goal: reproduce APEX4 end to end after a usable public checkpoint/package exists.

### Unblock check

Require corrected complete checkpoint shards plus a reproducible end-to-end
package. The already successful released-kernel build is not enough.

### Plan after unblocking

1. Freeze repository/source revisions, dependency/toolchain versions, every
   checkpoint shard hash and published reference command.
2. Verify shard completeness before build or GPU work. Abort on truncation or
   missing package components.
3. Reproduce the released kernel build/correctness tests on the RTX 3090.
4. Build and exercise the actual end-to-end artifact; prove the APEX4 path is
   active and retain raw hardware measurements against a paired baseline.
5. Independently reproduce correctness and artifact identity.
6. Use `APEX4_E2E_REPRODUCED` only when checkpoint, package, kernel and
   end-to-end measurements all exist; never promote a kernel-only preflight.

---

## Parked and excluded work — not executable plans

The following are intentionally outside the 15-item executable/triggered queue:

- `LAB-SERVE-001d` remains parked behind the CUDA illegal-memory-access crash;
- custom CUDA kernels without a measured bottleneck, sub-4-bit KV, learned MoE
  placement, EAGLE/DSpark, distributed/disaggregated serving, Kubernetes,
  cluster/product integration, broad 35B RL/FT and sophisticated scheduling;
- 24/48/72-hour reliability soaks remain explicitly excluded and cancelled;
- RWKV7 is closed `HOLD_QUALITY`, Falcon-H1R is `HOLD_ROLE`, A2 Stage-2 is
  `G0 KILL`, and old ACT/MASTER tables are historical rather than active queues.

Do not turn a parked theme into an experiment without a new user decision,
measured trigger and reviewed manifest item.

---

## Recommended invocation order

1. Run Plan 1 and return the `EXECUTED` packet to Codex.
2. Codex independently reviews and either promotes or rejects P0.
3. Run `next` again. Execute the dependency-ready P1 selected by the pipeline.
   Plan 2 cannot unblock unless P0 was promoted; Plan 3 is independently ready.
4. Complete independent review between items when a promotion affects a
   dependency. Never let AGY review its own packet.
5. Run Plan 4 when it becomes the next ready item.
6. Run Plan 5's selection phase and wait for the user's single-candidate choice.
7. Run Plan 6 only when an actual packed artifact has been independently
   demonstrated.
8. Plans 7–15 stay blocked until their exact trigger becomes objectively true.
   AGY may revalidate a trigger read-only, but must not fabricate progress or
   execute the experiment while the trigger is absent.

## Completion report required after every AGY invocation

AGY must return:

- item ID and final valid pipeline state;
- files created or changed;
- exact commands and tests run;
- implementation digest and raw receipt/evidence paths;
- every acceptance gate with actual value and pass/fail;
- failures, limitations, service changes/restoration and remaining blockers;
- confirmation that it did not train outside scope, modify historical receipts,
  self-review, verify, promote, commit or push.

Codex remains the independent reviewer and promotion authority.
