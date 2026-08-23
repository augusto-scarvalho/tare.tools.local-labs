# Muse Glimmer 30B qualification result — RTX 3090

**Decision:** `HOLD` base candidate; `DRAFT_REJECTED`; no deployed-role promotion  
**Executed:** 2026-08-21  
**Hardware:** RTX 3090 24 GB, WSL2; embedding service on port 8081 remained live  
**Packet:** `docs/research/MUSE_GLIMMER_3090_EXPERIMENT_PACKET_2026-08-21.md`

## Executive result

Muse Glimmer is a credible local long-context and vision candidate, but it did not clear the fail-closed
promotion gates in this tranche. Text-only and vision-only fit the RTX 3090 reserve, the compact context curve
was 16/16 through 120k tokens, and two local screenshots were understood correctly. However, the model scored
7/8 rather than the Qwen3.8 comparator's 8/8 on the agent suite, failed to terminate on `Mbpp/260` within both
2,048 and 4,096 completion tokens even at reasoning strength `low`, and showed one cold/warm cache transcript
drift. The official DFlash drafter accelerated decode but did not preserve byte-identical greedy output.

The preregistered early-stop rule therefore closed the expensive wider coding, VQA, context-replication,
prompt-injection, and fork-port campaigns. The artifacts and isolated runtime remain available for a focused
upstream-fix requalification; nothing was promoted or copied into the deployed fork.

## Frozen identities

| Component | Identity |
|---|---|
| Text GGUF | `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`; 16,756,683,904 bytes; SHA-256 `4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e` |
| Perception GGUF | `mmproj-Muse-Glimmer-30B-Q4_K_M.gguf`; 1,400,328,928 bytes; SHA-256 `f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6` |
| DFlash GGUF | `dflash-Muse-Glimmer-30B-Q4_K_M.gguf`; 1,631,208,128 bytes; SHA-256 `b2e808bf656086fe86bd0d0bd990f01d33e377537a07c02d45371517c8b264ef` |
| Hub revision | `70bf1b61ac09f91b24d39038091b41c582bc5d7a` |
| Runtime | isolated `/home/augus/src/slop.cpp-muse-glimmer`; llama.cpp `b10573`; commit `d775b8967a46d8beb110d444aa3b8938179e0dd8` |
| Build | static release, CUDA enabled via `/usr/local/cuda/bin/nvcc`; Muse architecture registered |
| Template | embedded ATEM Jinja template, 9,992 characters; `--jinja` enabled |

Expected byte sizes matched exactly. The admission smoke returned `391` for `17 * 23`, kept the answer in
`content`, kept thought in `reasoning_content`, and exposed `general.architecture = muse-glimmer` through the
runtime metadata path. Build and model admission therefore passed.

## Residency matrix

All measurements include the always-on embedding service. The operational floor is 4,096 MiB free VRAM.

| Arm | Context | Used MiB | Free MiB | Envelope |
|---|---:|---:|---:|---|
| A — text | 32,768 | 17,767 | 6,556 | pass |
| B — text + DFlash | 32,768 | 20,202 | 4,121 | pass, narrow |
| C — text + vision | 32,768 | 19,436 | 4,887 | pass |
| D — text + vision + DFlash | 32,768 | 21,848 | 2,475 | **fail reserve** |
| A — text | 131,072 | 19,134 | 5,189 | pass |

The full simultaneous multimodal+DFlash stack is not qualified on this 3090. Vision and DFlash may only be
considered as separate configurations unless memory use changes.

## Correctness and safety gates

- Agent suite: **7/8** at both legacy/default handling and explicit `reasoning_strength=low`; selection,
  nested arguments, abstention, sequential dispatch, multi-turn dispatch, error recovery, and irreversible
  no-blind-retry passed. Parallel dispatch failed by calling weather only for Lisbon and omitting Tokyo.
  Qwen3.8's existing paired receipt is 8/8.
- GSM8K failure replay: **4/5** of the five historical Qwen3.8 failures, 5/5 format adherence, zero
  truncations. Muse still answered `113` rather than gold `98` for ambiguous `gsm8k/1019`.
- MBPP+ focus: `Mbpp/260` produced no final answer at both 2,048 and 4,096 tokens under `low`; generation
  stayed in reasoning until the length limit. The wider coding gate was correctly not opened.
- Vision: the error-dialog fixture was transcribed correctly and the model explicitly rejected blind retry,
  recommending status verification/idempotency. The UI fixture's labels, fields, links, and hierarchy were
  all identified. These are qualitative compact gates, not a replacement for the paired VQA packet.
- Cache/cancellation: at 256 output tokens, shared-prefix reuse, cancel-then-reuse, and long-context reuse
  passed. Partial-removal preserved the `COBALT` oracle but changed the cold/warm reasoning transcript and
  repeated prefix material, so strict equality failed. Slot file save/restore was unavailable (`HTTP 501`)
  under this runtime/launch, consistent with keeping persistent-state qualification out of scope.

## Context and reasoning

The paired single-replicate context screen passed **16/16** strict cells:

| Target | Retrieval | Multikey | Multihop | Aggregation |
|---:|---:|---:|---:|---:|
| 8,192 | pass | pass | pass | pass |
| 32,768 | pass | pass | pass | pass |
| 65,536 | pass | pass | pass | pass |
| 120,000 | pass | pass | pass | pass |

The 131,072-token server allocation remained above the VRAM reserve. At the 120k prefill, a spot observation
showed 100% GPU utilization, 80 C, and approximately 386 W. This qualifies a promising 120k text-only scout,
not a replicated production SLO.

All four reasoning strengths answered a small arithmetic control correctly. Completion-token counts were
`low=116`, `medium=146`, `high=241`, and `xhigh=223`. Even `low` repeatedly reconsidered a trivial correct
answer; `low` must therefore remain the default for any future focused requalification.

## DFlash A/B

Three repetitions per arm used the same uncached raw prompt, greedy sampling, seed 42, and 384 output tokens.

| Arm | Mean decode tok/s | Speedup | Mean acceptance | Stable within arm | Byte-equal to no-spec |
|---|---:|---:|---:|---:|---:|
| no-spec | 47.33 | 1.000x | n/a | yes | baseline |
| `n=4` | 53.08 | 1.121x | 0.360 | yes | **no** |
| `n=8` | 71.66 | 1.514x | 0.226 | yes | **no** |
| `n=15` | 69.92 | 1.477x | 0.125 | yes | **no** |

`n=8` was fastest, but all DFlash arms changed deterministic text. The drafter therefore fails G5 despite its
speed. The full stack also fails the VRAM-reserve gate. Final disposition: `DRAFT_REJECTED` pending an upstream
runtime/model update and a fresh byte-equivalence A/B.

## Service restoration receipt

The experimental server on port 8092 was stopped cleanly. `llm-inference.service` was restarted and verified:

- port 8080: healthy, alias `qwen38-27b`, build `b9863-5e7f6271c`, one 131,072-token slot;
- argv restored to `Qwen3.8-27B-UD-Q4_K_XL.gguf`, q4 KV, MTP depth 3, checkpoints 32;
- port 8081: healthy, original embedding PID 34282 unchanged;
- no Muse process remained listening.

## Decision and next trigger

- `PROMOTE_TEXT_AGENT`: **no** — critical parallel-tool regression and MBPP non-termination.
- `PROMOTE_MULTIMODAL_AGENT`: **no** — compact visual evidence is positive, but agent eligibility failed and
  the paired quantitative visual/safety packet was gated off.
- `PROMOTE_DFLASH`: **no** — deterministic equivalence and full-stack reserve failed.
- Base artifact: **HOLD** as a promising 120k text/vision specialist and independent open-weight option.

Reopen only after an upstream Muse/parser/DFlash change plausibly addresses deterministic equivalence,
parallel ATEM calls, or runaway reasoning. The first rerun should be the compact failing set, not the full
matrix.

**Authorized descriptive expansion, 2026-08-22:** the previously gated VQA/injection slices were explicitly
reopened without waiving these promotion failures. Muse scored 107/150 on the exact retained MMStar panel and
passed 5/5 bounded multimodal-safety cases. See
`../MUSE-GLIMMER-FULL-2026-08-22/RESULT.md`; overall disposition remains `HOLD`.

## Evidence files

- `arm-A-agent-suite-low.json`: explicit-low agent gate.
- `arm-A-mbpp260-low-r2-2048/`: identity-bound coding failure receipt.
- `gsm8k-qwen38-failures-low/`: five-case Qwen failure replay.
- `context-curve-low.json`: 16-cell 8k–120k context curve.
- `reasoning-strengths.json`: four-level control.
- `dflash-ab.json`: paired no-spec/n4/n8/n15 A/B.
- `cache-cancel-isolation-n256.json`: cache/cancel lifecycle gate.
- Earlier non-qualified attempts are intentionally retained (`context-curve.json`,
  `cache-cancel-isolation.json`, and the first MBPP directories) and are not used for promotion claims.
