# Changelog

Notable changes to `tare.tools.local-labs` are recorded here, newest first. This file starts with a concise retrospective of the current public baseline; Git history remains authoritative for commit-level detail.

## Unreleased

### Added

- Added a fail-closed qualified-model fleet registry, an OpenAI-compatible on-demand
  gateway for the single RTX 3090, and an agent-facing `modelctl` CLI for discovery,
  role recommendation, evidence/limit inspection, live status and requests. Six
  bounded routes are exposed while HOLD/rejected artifacts remain unroutable.
- Added the complete independent rerun and consolidation of all 36 AGY claims,
  with a rank-by-rank tracker, GitHub-facing audit report, provenance-complete
  successor packets, raw scored samples, physical runtime receipts and explicit
  claim limits.
- Added fresh seed-20260827 training for ADAPT-01 through ADAPT-05, including
  12 physical training arms and 768 independently reconciled behavioral
  generations across the main and omitted-arm completion packets.
- Added portable external-source receipts for Git-excluded GGUF/codec
  derivatives. The backlog gate now requires committed path/digest/byte-count
  evidence when a declared large source is absent from a checkout.
- Added a fail-closed Gemini backlog pipeline with a dependency-aware canonical
  queue, preregistration and implementation hash binding, provenance-complete
  immutable receipts, centrally recomputed acceptance gates, evidence-class
  promotion limits and mandatory independent review.
- Added repository-level `GEMINI.md`, operator documentation, ten pipeline
  regression tests and a CI gate covering every registered research packet.
- Added a manual AGY handoff with a dedicated execution plan, frozen gates,
  evidence requirements and stop conditions for each current backlog item.
- Reconciled the six Gemini-remediation items with nine older residual
  trigger-blocked items, yielding a 15-item canonical queue while keeping
  parked, cancelled, `HOLD` and closed work outside executable selection.
- Added the complete Gemini backlog evidence wave: 36 immutable run packets,
  their probes and analysis helpers, the 46-item hypothesis/literature map,
  comprehensive synthesis, and historical closeout handoffs.
- Added `local-labs-experiment-provenance-v1`, binding receipts to the exact
  command, Git state, script, hashed inputs, Python/package environment, GPU,
  runtime parameters, and a canonical receipt fingerprint.
- Added provenance-complete successor packets for `BEE-L1C`, `SLX-01C`,
  `SLX-05D`, `REP-02B`, `SLX-09B`, and `TRAIN-00B`, including preserved failed
  predecessors rather than overwritten receipts.
- Added deterministic tests for effective-route comparison, provenance
  completeness, strict serving recovery, hybrid-cache snapshot restoration,
  and the new analysis helpers. The repository suite now contains 59 passing
  tests plus the 23-case LAB-QA-001 metamorphic harness.
- Added the canonical Codex remediation handoff and a SHA-256 ledger covering
  the original Gemini receipts and documents.
- Added a consolidated 2026-08-24/25 execution closeout covering every material host, service, driver, model, transcript, APEX4, and PEFT action; persistent local artifacts; recovered failures; explicit non-actions; current live state; CI receipts; and the trigger-gated residual backlog.
- Recorded the NVIDIA 591.86 post-reboot qualification and retained rollback/recovery receipts without tracking local driver payloads.
- Added matched Fable, HauhauCS, and vanilla agent/GSM8K evidence, plus safe vanilla canary activation and restoration cleanup.
- Added the official RWKV7 license revalidation and frozen 48-item quality harness and receipt.
- Reconciled the BeeLlama/slop.cpp/PEFT transcript into a dependency-gated research queue with pinned source archaeology, lifecycle-gap, APEX4, and adapter-mechanics packets.
- Added a deterministic LoRA mechanics smoke that binds the existing ThinkingCap teacher receipts to the frozen GSM8K prompt snapshot and measures target learning, protected-text retention, clean reload, and VRAM.
- Added a seven-geometry PEFT screen plus a codec-independent full-distribution KV qualification scorer and staged BeeLlama-inspired qualification contract.

### Changed

- Consolidated the 36-item AGY scientific ledger with the 52-record canonical
  FSM history: 31 claims have a decisive physical successor, five are purely
  implementation-blocked, and SLX-08 has a qualified fidelity result plus a
  separately blocked TTFT claim.
- Reclassified three historical outcomes as false negatives: ADAPT-01 found a
  promoted 384-step/LR=1e-4 LoKr arm, full-trace distillation beat answer-only
  by 8.33 percentage points, and corrected SLX-08 real-QKV fidelity reached
  0.99545.
- Reclassified or bounded eight historical promotions after physical reruns:
  CUDA Graph, DISTILL-00, DISTILL-01, CTRL-01, BEE-L3, BEE-L5, SLX-10 and
  SPEC-01 failed at least one decisive causal, quality or runtime gate.
- Kept large regenerated GGUF, packed-codec and redundant tokenizer artifacts
  out of ordinary Git while retaining immutable SHA-256/byte receipts and the
  PEFT weights required by downstream source bindings.
- Split physical-host and portable-CI validation: the research host passes all
  141 tests, while GitHub runs 139 tests plus the fail-closed backlog gate and
  deselects only two assertions that require locally materialized large inputs.
- Superseded the Gemini claim that all 46 backlog items were executed and
  audited. Of the 36 new run packets, 25 are now `SIMULATION_ONLY`, nine were
  unverified model preliminaries, and two endpoint runs had insufficient gates.
- Reclassified `ADAPT-01A` through `ADAPT-05` as
  `UNVERIFIED_PRELIMINARY`; they require a new artifact/data/evaluation packet
  before expensive reproduction.
- Qualified CUDA Graph replay only for the frozen Qwen3.5-0.8B tuple after
  exact semantic parity and fixed hybrid-cache restoration; removed the former
  exclusive launch-overhead and persistent-megakernel interpretation.
- Kept 12 byte-identical adapter-export `tokenizer.json` derivatives out of Git
  while retaining their configs, templates, metrics, and adapter weights. The
  excluded payload SHA-256 is
  `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`.
- Marked raw research JSON as binary evidence in `.gitattributes`, preserving
  historical bytes and line endings so the published SHA-256 ledger remains
  valid after checkout.
- Kept Fable-TC as the broad serving default after HauhauCS passed the agent core but showed bounded math termination loss.
- Closed FastMTP before installation because the prerequisite broad-default and termination gate failed.
- Revalidated the residual backlog: RWKV7 moved to `HOLD_QUALITY`, six dependency-gated items remain blocked, and the non-soak ready queue is empty.
- Closed the APEX4 preflight without a port: released kernels build and pass correctness on the RTX 3090, but the pinned public checkpoint is internally truncated and no relevant end-to-end package was available.
- Completed the frozen adapter screen: LoKr led target-loss improvement with 359k trainable parameters, IA3 minimized footprint, six arms passed, and DoRA failed non-finite at the first training step.
- Closed the behavioral adapter gate without promotion: LoKr improved exact held-out GSM8K from 4/32 to 15/32 but missed the frozen correctness and natural-termination floors; ADAPT-01 remains blocked.

### Fixed

- Fixed backlog portability so CI can validate source identity without storing
  multi-gigabyte model derivatives; unbound missing inputs still fail closed.
- Fixed the ADAPT-01 false-negative classification by evaluating the fresh
  lower-learning-rate arm and separately closing the 640-step arm omitted by
  the historical driver.
- Replaced causal narratives based on proxy or order-confounded comparisons
  with physical matched controls, including CUDA OFF/ON, MTP K0/K2/K4,
  answer-only/full-trace SFT, and live MTP-plus-ngram serving.
- Made effective-route receipts compare systemd argv, live process cmdline,
  model path/full SHA, build identity, slot realization, and a strict exercised
  canary instead of trusting requested configuration.
- Made the serving-torture gate require explicit idle state, exact abort/normal
  counts, stable PID/restart count, strict canaries, and bounded VRAM drift.
- Fixed the CUDA Graph oracle for Qwen3.5 hybrid caches by restoring the full
  post-prefill KV/convolution/recurrent state outside each timed observation.
- Fixed REP-02's invalid cross-position logit comparator and relabeled its
  dequantized INT4 memory number as an analytical packed-storage estimate.
- Added complete provenance and scope limits to the 2:4 mask and short custom
  GaLore reruns; neither can claim packed-kernel acceleration or general
  algorithm performance.
- Corrected GPU-control ownership documentation and made Fable restoration remove both HauhauCS and vanilla experiment drop-ins.

### Pending

- Independently review the Codex successor packets currently stopped at
  `EXECUTED`; the executor did not self-promote or author approval receipts.
- Materialize the six absent integration claims before retesting them:
  SLX-03 recurrent state-write elision, SLX-07 H2O eviction, SLX-08
  selected-block TTFT, REP-04 KVarN fused kernel, REP-05 per-layer KV precision,
  and RETRO-01 trained retrofit routing.
- Retain the external-prerequisite backlog for genuine human judge labels,
  official ThinkingCap/RetNet/APEX4 artifacts, quantizer provenance, MTP
  identity/persistence and a physical BEE-L2 codec.

## 2026-08-24

### Added

- Published the complete host/WSL tuning receipts, Fable-TC serving baseline, and recovery procedures ([2ccff76](https://github.com/augusto-scarvalho/tare.tools.local-labs/commit/2ccff76170b14057fbd4b347c1821b68e6026f4e)).
- Qualified HauhauCS Aggressive against Fable-TC and vanilla Qwen3.8 for code, ordinary questions, refusal behavior, context, and throughput, with pinned model provenance and raw results ([2ccff76](https://github.com/augusto-scarvalho/tare.tools.local-labs/commit/2ccff76170b14057fbd4b347c1821b68e6026f4e)).
- Added the PT-BR locale-contract proxy, systemd unit, deterministic tests, canary activation, model download, and Fable restoration tooling ([2ccff76](https://github.com/augusto-scarvalho/tare.tools.local-labs/commit/2ccff76170b14057fbd4b347c1821b68e6026f4e)).

### Changed

- Froze the completed experiment campaign and remaining dependency-gated backlog in the operational handoff ([f2ed928](https://github.com/augusto-scarvalho/tare.tools.local-labs/commit/f2ed928)).
- Recorded the exact model-disk cleanup and retained serving artifacts ([7a60aff](https://github.com/augusto-scarvalho/tare.tools.local-labs/commit/7a60aff)).

## 2026-08-23

### Added

- Closed and published the autonomous RTX 3090 experiment campaign ([6a5ec02](https://github.com/augusto-scarvalho/tare.tools.local-labs/commit/6a5ec02)).
- Added deterministic repository checks ([8073773](https://github.com/augusto-scarvalho/tare.tools.local-labs/commit/8073773)).

### Changed

- Established the ownership boundary between laboratory evidence and the `slop.cpp` inference engine ([4db671b](https://github.com/augusto-scarvalho/tare.tools.local-labs/commit/4db671b)).

### Fixed

- Made CI independent of an absent pip cache manifest ([5339b64](https://github.com/augusto-scarvalho/tare.tools.local-labs/commit/5339b64)).

## 2026-08-21

### Added

- Consolidated the historical RTX 3090 MoE sweep and runner setup ([dcc0c55](https://github.com/augusto-scarvalho/tare.tools.local-labs/commit/dcc0c55)).
