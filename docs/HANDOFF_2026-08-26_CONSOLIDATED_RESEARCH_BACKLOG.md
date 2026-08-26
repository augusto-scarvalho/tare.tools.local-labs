# Handoff: consolidated research backlog after the full AGY rerun audit

Date: 2026-08-26  
Repository: `tare.tools.local-labs`  
Canonical manifest: `config/research_backlog.json`  
Manifest schema: `local-labs-backlog-v1`  
Manifest snapshot: `2026-08-26T10:35:44Z`  
Repository baseline before this handoff: `f12531e76d0e21dbb937223f5937b38687e689f4`

## Purpose

This document is the restart point for the next research agent. It consolidates
the original 15-item remediation queue, the independent rerun of all 36 ranked
AGY claims, the provenance-preserving successor packets, the current review
queue, objective integration gaps, and the remaining trigger-blocked research.

Do not reconstruct the backlog from chat transcripts or count only the most
recent five packets. The machine-readable manifest is authoritative for state;
this handoff explains how to interpret that state without mistaking an aborted
attempt for an unresolved scientific hypothesis.

## Executive snapshot

| View | Current disposition |
|---|---:|
| Canonical pipeline records | 52 |
| `PROMOTED` | 3 |
| `REJECTED` | 2 |
| `EXECUTED`, independent review pending | 20 |
| `BLOCKED`, total manifest records | 27 |
| Superseded/aborted records inside `BLOCKED` | 17 |
| Genuinely active trigger-blocked items | 10 |
| Dependency-ready `PROPOSED` items | 0 |
| Historical AGY ranks reconciled | 36/36 |
| Historical ranks with decisive physical successor | 31/36 |
| Historical ranks exclusively blocked by absent integration | 5/36 |

The pipeline gate passes and `next` reports `No dependency-ready PROPOSED
item.` The immediate work is independent review and state reconciliation, not
launching another unregistered experiment.

## Sources of truth, in order

1. `config/research_backlog.json`: canonical IDs, states, dependencies, gates,
   required evidence, allowed claim codes, blockers, and state history.
2. `runs/research/<BACKLOG-ID>/PIPELINE.json`: packet state and bound
   implementation digest.
3. `runs/research/<BACKLOG-ID>/PRE_REGISTRATION.md`: frozen hypothesis,
   treatment, controls, thresholds, abort rules, and claim limits.
4. `runs/research/<BACKLOG-ID>/raw/receipt.json`: canonical execution receipt.
5. Raw evidence named by the receipt. Recompute from these files; do not trust
   narrative flags or `RESULT.md` alone.
6. `runs/research/<BACKLOG-ID>/RESULT.md`: bounded executor summary.
7. `runs/research/<BACKLOG-ID>/REVIEW.json`: independent decision, only when it
   binds the exact receipt and implementation digest.

Human-facing crosswalks:

- `docs/AGY_36_INDEPENDENT_RERUN_TRACKER_2026-08-25.md`: rank-by-rank map of
  all 36 historical AGY claims to their decisive successor or blocker.
- `docs/AUDIT_2026-08-26_CODEX_AGY_36_FULL_RERUN.md`: GitHub-facing synthesis
  of false negatives, false positives, retained results, and integration gaps.
- `docs/research/BACKLOG_IMPLEMENTATION_PIPELINE.md`: state-machine and receipt
  contract.

## Authority and state-machine boundary

The legal success path is:

```text
PROPOSED -> PREREGISTERED -> IMPLEMENTED -> EXECUTED -> VERIFIED -> PROMOTED
                                               +-> REJECTED
```

- An executor stops at `EXECUTED` and writes a bounded `RESULT.md`.
- The executor cannot author its own `REVIEW.json` or advance its own packet to
  `VERIFIED`, `PROMOTED`, or `REJECTED`.
- The 20 successor packets below were executed by Codex and require AGY, or a
  different genuinely independent actor, to review them. Another Codex acting
  as the same executor lineage must not self-sign them.
- A failed frozen gate is a valid scientific result and normally closes as
  `REJECTED`; it is not converted to `BLOCKED` to avoid a negative result.
- `BLOCKED` is reserved for an absent prerequisite or a fail-closed aborted
  attempt. Corrections use successor IDs; raw predecessor evidence is immutable.
- Displaced conclusions remain preserved and are labeled `SUPERSEDED`; they are
  never silently edited or deleted.

## Priority 0: independent review queue

These 20 packets are physically executed and are the next admissible work. The
"expected bounded disposition" is an audit target, not authorization to skip
raw-evidence review.

| Packet | Expected bounded disposition if evidence survives | Essential boundary |
|---|---|---|
| `BACKLOG-ADAPT-REQUAL-02` | `ARTIFACT_REQUALIFIED_R2` | All isolation gates passed, but the old `target_mlp_only` finalist ranking was not reproduced and is invalid for downstream selection. |
| `BACKLOG-CUDAGRAPH-SERVING-02` | `SERVING_CUDAGRAPH_CAUSAL_REJECTED_R2` | Explicit OFF/ON gave 1.036998x, below the 1.10 gate; preserve only the small tuple-specific directional benefit. |
| `BACKLOG-ADAPT-TRACE-DISTILL-03` | `TRACE_DISTILLATION_FALSE_NEGATIVE_CONFIRMED_R3` | Full traces beat answer-only by 8.33 pp mean over three seeds; no teacher noninferiority or production claim. |
| `BACKLOG-NEGATIVE-KV-REAL-SCREEN-02` | `NEGATIVE_KV_REAL_SCREEN_VERIFIED_R2` | Real Qwen tensors retained the five bounded negative mechanism results. |
| `BACKLOG-SLX08-REAL-FIDELITY-01` | `SLX08_FIDELITY_FALSE_NEGATIVE_CANDIDATE_R1` | Fidelity passed at median 0.99545; TTFT remains blocked because selected-block prefill is absent. |
| `BACKLOG-HYPER01-REAL-ADAPTER-01` | `HYPER01_NEGATIVE_RETAINED_R1` | Physical LoRA targets preserved the overhead rejection; this is a module screen, not runtime integration. |
| `BACKLOG-DISTILL01-FLEET-REAL-01` | `DISTILL01_FALSE_POSITIVE_CONFIRMED_R1` | Routed fleet gained 15.38%, below the 20% and math gates. |
| `BACKLOG-CTRL01-REAL-TOKEN-06` | `CTRL01_FALSE_POSITIVE_CONFIRMED_R6` | Offline sidecar reduced valid JSON and is absent from the production runtime. |
| `BACKLOG-RSH02-PACKED-GPU-02` | `RSH02_NEGATIVE_RETAINED_R2` | Physical Triton block-Huffman decoded exactly but failed compression/throughput gates. |
| `BACKLOG-GDN02-LEARNED-STATE-01` | `GDN02_NEGATIVE_RETAINED_R1` | Learned recurrent states failed old-fact leakage and update-fidelity gates. |
| `BACKLOG-BEE-L4-LIVE-MTP-01` | `BEE_L4_LIVE_SLOT_ISOLATION_QUALIFIED_R1` | Bounded observable slot isolation only; do not generalize beyond the frozen live route. |
| `BACKLOG-BEE-L5-LIVE-GUARD-04` | `BEE_L5_FALSE_POSITIVE_CONFIRMED_R4` | Detection worked, but 7.8 us/token p95 failed the 2 us/token gate. |
| `BACKLOG-ADAPT06-SLOP-LIVE-05` | `ADAPT06_LIVE_ISOLATION_QUALIFIED_SLOP_CLIENT_AFFINITY_R5` | Two real LoRAs routed cleanly; the SLOP result is client affinity, not a server-native scheduler/fused-GEMM claim. |
| `BACKLOG-BEE-L3-REAL-TELEMETRY-01` | `BEE_L3_FALSE_POSITIVE_CONFIRMED_R1` | Only +3.68% over K4, 83.33% exact parity, and no per-request K switching in runtime. |
| `BACKLOG-AGY-SYSTEM-BLOCKERS-02` | `AGY_SYSTEM_BLOCKERS_REGISTERED_R2` | Registers absent physical integrations; it does not reject the underlying algorithms. |
| `BACKLOG-SLX10-PACKED-RUNTIME-02` | `SLX10_FALSE_POSITIVE_CONFIRMED_R2` | Q2_K improved speed/VRAM but missed the file ceiling, reduced accuracy to zero, and had 0/32 exact outputs. |
| `BACKLOG-SPEC01-LIVE-HYBRID-01` | `SPEC01_FALSE_POSITIVE_CONFIRMED_R1` | Hybrid preserved 30/30 outputs but reached only 0.689x MTP-only throughput and lacks proposer attribution. |
| `BACKLOG-SLX11-OFFICIAL-HYBRID-01` | `SLX11_OFFICIAL_HYBRID_ARTIFACT_QUALIFIED_R1` | Official architecture/artifact qualification only; historical 4.49x and recall claims remain unverified. |
| `BACKLOG-ADAPT-MECHANISMS-RERUN-01` | `ADAPT01_05_MECHANISMS_MIXED_R1` | ADAPT-01 false negative; ADAPT-02 promotion reproduced without unique MLP causality; ADAPT-03/04/05 negative results retained. |
| `BACKLOG-ADAPT01-640-EVAL-01` | `ADAPT01_640_ARM_REJECTED_R1` | The 640-step arm scored well descriptively but failed frozen EOS/length gates. |

For every packet, recompute all frozen gates from raw evidence, verify receipt
SHA-256 and implementation digest, write the independent `REVIEW.json`, then use
`backlog_pipeline.py advance`. Do not batch-sign based on this table.

## Existing terminal states that require reconciliation

The manifest's three promotions and two rejections are valid historical state
records, but stronger successors now constrain how they may be cited.

| Existing state | Current interpretation |
|---|---|
| `BACKLOG-ADAPT-REQUAL-01` — `PROMOTED` | Superseded in evidentiary quality by process-isolated R2. Artifact identity survived, but the old finalist ranking did not. |
| `BACKLOG-ADAPT-TRAIN-01` — `PROMOTED` | The two-seed training run exists, but it trained a finalist selected by the now-invalidated R1 ranking. Do not use it to authorize downstream distillation without a new selection decision. |
| `BACKLOG-CUDAGRAPH-SERVING-01` — `PROMOTED` | The 1.5115x causal interpretation is contradicted by R2 and should be marked `SUPERSEDED` after independent review. |
| `BACKLOG-DISTILL-REAL-01` — `REJECTED` | Retained: the real student was 56.25 pp less accurate and used 102.11% more tokens than the teacher on the frozen boundary. |
| `BACKLOG-ADAPT-TRACE-DISTILL-01` — `REJECTED` | Challenged by the matched three-seed R3 successor; likely a confirmed false negative after review. |

Do not rewrite terminal state history to express supersession. Record the
successor review and update human-facing ledgers with an explicit `SUPERSEDED`
annotation.

## Priority 1: genuinely active trigger-blocked backlog

These are the ten remaining research lines. None is ready merely because an
agent can scaffold code; each must first satisfy its exact entry condition.

| Packet | Exact unblock condition |
|---|---|
| `BACKLOG-MTP-PERSISTENCE-01` | Freeze a sufficiently specific, falsifiable cache-lifecycle mechanism hypothesis and deterministic restore/spec-decode oracle. |
| `BACKLOG-PROXY-REALIZATION-01` | Select exactly one proxy candidate and a concrete physical implementation target. |
| `BACKLOG-PACKED-HARDWARE-01` | Produce an immutable packed artifact and callable runtime route suitable for physical measurement. |
| `BACKLOG-BEE-L2-KV-CODEC-01` | Provide a physical immutable KV codec plus effective-route and artifact receipts. |
| `BACKLOG-THINKINGCAP-QWEN38-01` | Obtain official ThinkingCap Qwen3.8 weights and an RTX-3090-fit artifact. |
| `BACKLOG-THINKINGCAP-MTP-IDENTITY-01` | Obtain publisher evidence identifying the exact 17,221,641,152-byte legacy artifact. |
| `BACKLOG-QUANTIZER-PROVENANCE-01` | Obtain the publisher's exact quantizer and llama.cpp build receipt. |
| `BACKLOG-HUMAN-JUDGE-CALIBRATION-01` | Freeze a genuine blind human-label packet, originally scoped at 50-100 labels. |
| `BACKLOG-RETNET-OFFICIAL-01` | Obtain an official Microsoft/TorchScale pretrained checkpoint; community weights do not satisfy the gate. |
| `BACKLOG-APEX4-E2E-01` | Obtain corrected complete checkpoint shards and a reproducible end-to-end package. |

If a trigger becomes objectively true, first update evidence and use the legal
`BLOCKED -> PROPOSED` transition. Do not implement while the item is still
blocked.

## Archived fail-closed attempts

The following 17 `BLOCKED` records are not active scientific work. They preserve
failed launch/observability attempts and have a decisive executed successor.

| Archived attempts | Executed successor |
|---|---|
| `BACKLOG-ADAPT-TRACE-DISTILL-02` | `BACKLOG-ADAPT-TRACE-DISTILL-03` |
| `BACKLOG-NEGATIVE-KV-REAL-SCREEN-01` | `BACKLOG-NEGATIVE-KV-REAL-SCREEN-02` |
| `BACKLOG-CTRL01-REAL-TOKEN-01` through `-05` | `BACKLOG-CTRL01-REAL-TOKEN-06` |
| `BACKLOG-RSH02-PACKED-GPU-01` | `BACKLOG-RSH02-PACKED-GPU-02` |
| `BACKLOG-BEE-L5-LIVE-GUARD-01` through `-03` | `BACKLOG-BEE-L5-LIVE-GUARD-04` |
| `BACKLOG-ADAPT06-SLOP-LIVE-01` through `-04` | `BACKLOG-ADAPT06-SLOP-LIVE-05` |
| `BACKLOG-AGY-SYSTEM-BLOCKERS-01` | `BACKLOG-AGY-SYSTEM-BLOCKERS-02` |
| `BACKLOG-SLX10-PACKED-RUNTIME-01` | `BACKLOG-SLX10-PACKED-RUNTIME-02` |

Their raw artifacts must remain immutable. Do not count them as 17 additional
open experiments and do not delete them to make the dashboard look cleaner.

## Objective integration gaps from the 36-item audit

Five historical ranks remain exclusively blocked, and SLX-08 has one additional
blocked TTFT component:

| Claim | Missing physical treatment |
|---|---|
| `SLX-03` state-write elision | Compiled recurrent-state write cadence and physical write counters. |
| `SLX-07` H2O | Attention-score accumulator plus a real KV eviction lifecycle. |
| `SLX-08` TTFT | Selected-block speculative-prefill route wired into real serving. |
| `REP-04` KVarN | Callable fused KVarN kernel; unrelated Hadamard code is not a comparator. |
| `REP-05` layerwise KV precision | Per-layer KV allocator and configuration surface; global KV types are insufficient. |
| `RETRO-01` recurrent retrofit | Trained retrofit checkpoint plus a physical inference route. |

These are blocked integration claims, not algorithm rejections. A mechanism,
proxy, or analytical simulation cannot promote them to production evidence.

## Historical corrections that must not be forgotten

### Three confirmed false negatives

1. ADAPT-01: fresh LoKr LR 1e-4 at 384 steps reached 17/32 math, 4/16
   protected QA, and 41/48 natural EOS; the separate 640-step arm remained
   rejected.
2. Trace distillation: full traces beat matched answer-only SFT by 8.33
   percentage points on the preregistered three-seed mean, with one negative
   seed and bounded protected-QA regression.
3. SLX-08 fidelity: corrected gather over real QKV reached median 0.99545; the
   historical probe computed indices but evaluated random QKV.

### Eight false positives or historical overclaims

1. CUDA Graph serving: 1.036998x causal speedup, not 1.5115x.
2. DISTILL-00 concise student: -56.25 pp accuracy and 102.11% token inflation.
3. DISTILL-01 fleet: +15.38%, below the frozen 20% and math gates.
4. CTRL-01: valid JSON fell from 24/24 raw to 18/24 with the sidecar.
5. BEE-L5: behavior passed, but 7.8 us/token p95 failed the 2 us/token gate.
6. BEE-L3: +3.68% over K4, 83.33% exact parity, no live per-request switching.
7. SLX-10: Q2_K improved systems metrics but failed size, accuracy, and exact
   semantic-parity gates.
8. SPEC-01: combined route was 0.689x MTP-only, not 3x, with no proposer-level
   attribution.

DISTILL-00 appears in both correction directions because two different
hypotheses were tested: concise-student superiority was a false positive, while
matched full-trace versus answer-only training exposed a false negative.

## Retained or bounded results worth carrying forward

- ADAPT-02 targeting reproduced positive behavior, but MLP-only was not the
  unique causal winner; QV-gate led the fresh math panel.
- ADAPT-03 soft prompts retained rejection due to protected-QA collapse.
- ADAPT-04 prior preservation and ADAPT-05 composite merging retained negative
  conclusions under fresh training.
- BEE-L4 live MTP slot isolation passed its bounded observable controls.
- ADAPT-06 plus SLOP client affinity achieved route-distinct, contamination-free
  replay with exact same-route behavior; no server-native scheduling claim.
- SLX-11 qualified the official hybrid artifact architecture only.
- The real-Qwen negative KV screen retained RSH-01, REP-03, RSH-03, RSH-04, and
  REP-06 mechanism-level rejections.
- HYPER-01, GDN-02, and physical RSH-02 retained their bounded negative results.

## Current operational baseline

Observed while writing this handoff:

- Public OpenAI-compatible gateway: `http://127.0.0.1:8080/v1`.
- Gateway identity: `qualified-model-gateway`.
- Resident generation model: `qwen38`.
- Private backend: healthy on `127.0.0.1:18080`, PID 46426.
- Maximum resident generation models: one.
- Available routes: `qwen38`, `qwen36-moe`, `fable-tc`, `hauhaucs`,
  `gemma-vision`, and `muse-vision`.
- Embedding service remains independent on port 8081.

This is a time-stamped observation, not a permanent guarantee. Recheck it
before any GPU maintenance. Stop the persistent inference service through
`systemctl`, never by killing its restart-managed child, and leave 8081 alone.

## Exact restart procedure for the next agent

Run from the repository root:

```powershell
git status --short
git rev-parse HEAD
git rev-parse origin/master
python tools/analysis/backlog_pipeline.py gate
python tools/analysis/backlog_pipeline.py status
python tools/analysis/backlog_pipeline.py next
python tools/agents/modelctl.py status --json
python -m pytest -q
```

Expected baseline at handoff creation:

- gate: `BACKLOG PIPELINE: PASS`;
- tests: 148 passing;
- `next`: no dependency-ready `PROPOSED` item;
- Git baseline: local `master` and `origin/master` at `f12531e` before this
  handoff change;
- serving: healthy gateway with `qwen38` resident.

If any value differs, treat that as drift and investigate before reviewing or
executing packets. Do not copy the expected output into a new receipt.

## Independent review procedure

For one `EXECUTED` packet at a time:

1. Read its manifest entry, `PRE_REGISTRATION.md`, `PIPELINE.json`, and
   `REVIEW.template.json` completely.
2. Hash `raw/receipt.json`; verify the receipt fingerprint and exact
   implementation digest.
3. Resolve every required evidence path and recompute sample counts, metrics,
   thresholds, and gate booleans independently.
4. Inspect treatment identity, model/artifact hashes, process isolation,
   service-maintenance evidence, and abort conditions relevant to the packet.
5. Enforce the narrowest allowed claim. Passing a mechanism gate does not imply
   production integration, general capability, or cross-model validity.
6. Write `REVIEW.json` using `local-labs-independent-review-v1`, including
   findings and the exact receipt/implementation bindings.
7. Use `backlog_pipeline.py advance`; never edit manifest or packet states by
   hand.
8. Rerun `gate` and tests after each logical review batch.
9. Update the tracker/audit with explicit `SUPERSEDED` annotations where a
   successor displaces a historical conclusion.

Suggested review order:

1. `BACKLOG-ADAPT-REQUAL-02`, because it constrains downstream adapter claims.
2. `BACKLOG-CUDAGRAPH-SERVING-02`, because it contradicts an active promotion.
3. `BACKLOG-ADAPT-TRACE-DISTILL-03`, because it contradicts an active rejection.
4. `BACKLOG-ADAPT-MECHANISMS-RERUN-01` and
   `BACKLOG-ADAPT01-640-EVAL-01`, reviewed together but decided separately.
5. The remaining false-positive candidates.
6. Bounded qualifications and retained-negative mechanism screens.

## Stop conditions

Stop and preserve evidence if any of the following occurs:

- pipeline gate failure;
- receipt fingerprint, source hash, or implementation digest mismatch;
- missing raw evidence named by the receipt;
- a reviewer is not independent from the executor;
- a proposed claim exceeds `allowed_claim_codes` or the evidence class;
- service identity changes unexpectedly, 8081 is disturbed, or multiple large
  generation models overlap in VRAM;
- a blocked item's trigger has not been objectively met;
- a historical packet would need to be edited to make a result pass.

Create a successor packet for a real correction. Never weaken a frozen gate,
overwrite raw evidence, fabricate a receipt, or infer execution from a scaffold,
running process, green unit test, or narrative report.

## Definition of backlog closeout

This consolidated backlog is not fully closed until:

1. all 20 `EXECUTED` packets have independent, digest-bound decisions;
2. contradicted terminal claims are explicitly marked `SUPERSEDED` in the
   human-facing ledgers;
3. every remaining `BLOCKED` record is either an immutable archived attempt or
   has a concrete externally checkable unblock condition;
4. `backlog_pipeline.py gate`, the portable test suite, and CI pass; and
5. the serving and embedding baselines are restored after any GPU work.

Until then, the correct summary is: execution of the 36-claim rerun is complete,
scientific adjudication of the 20 Codex successor packets is pending, and no new
dependency-ready experiment is currently authorized by the queue.
