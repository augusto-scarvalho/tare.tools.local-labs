# LAB-MUSE — Muse Glimmer 30B qualification packet for the RTX 3090

**Status:** `EXECUTED / HOLD / DRAFT_REJECTED / NOT_PROMOTED`  
**Date:** 2026-08-21  
**Target hardware:** RTX 3090 24 GB, 64 GB host RAM, WSL2  
**Decision scope:** add a materially different open-weight option for local text, visual, and multimodal-agentic work.  
**Non-authority:** this packet does not change the deployed model, service unit, fork default, or promotion state.

Execution result: [`../../runs/requalification/MUSE-GLIMMER-2026-08-21/RESULT.md`](../../runs/requalification/MUSE-GLIMMER-2026-08-21/RESULT.md).
The compact gates completed on 2026-08-21 and invoked the preregistered early stop: the base remains a promising
120k text/vision specialist, while agent-role and DFlash promotion failed.

## 1. Why this candidate is high priority

Muse Glimmer is not another Qwen requant. It adds an independent architecture and training lineage to a lab
whose strongest current candidates are concentrated in the Qwen family. The released model combines a dense
28B text decoder, an approximately 1.8B-parameter perception encoder, hybrid local/global attention, 2 KV
heads, a 131,072-token context claim, agent/tool training, and a target-matched DFlash drafter.

It directly addresses three open needs:

1. broaden the set of open-weight models that can plausibly run on one 24 GB card;
2. join text-agent, coding, screenshot, document, and tool-use capability in one local artifact;
3. provide a clean target-matched DFlash test after the lab's earlier mismatched DSpark/DFlash experiments
   failed or regressed on other targets.

The old A5 negative result remains valid for those exact target/drafter pairs. It does not close Muse Glimmer:
Muse support, its perception graph, and its DFlash path landed together in upstream llama.cpp, including a fix
for seeding the draft KV cache through multimodal embedding batches.

## 2. Frozen upstream identities at admission design time

API snapshot captured 2026-08-21. Hub revisions bind the proposed downloads; full local SHA-256 values must be
computed after download and before any run becomes evidence.

| Artifact | Hub revision | Expected bytes | Initial disposition |
|---|---|---:|---|
| `meta-models/Muse-Glimmer-30B` | `a4e59da52a7bc87ae7251dd5545c0dd437c44b68` | source repository | BF16 provenance only; do not download for the first screen |
| `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` | `70bf1b61ac09f91b24d39038091b41c582bc5d7a` | 16,756,683,904 | primary 24 GB candidate |
| `mmproj-Muse-Glimmer-30B-Q4_K_M.gguf` | same GGUF revision | 1,400,328,928 | add only after text smoke passes |
| `dflash-Muse-Glimmer-30B-Q4_K_M.gguf` | same GGUF revision | 1,631,208,128 | add only after no-draft baselines pass |
| `Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf` | same GGUF revision | 19,653,960,832 | deferred; upstream targets 32 GB hardware |
| `meta-models/Muse-Glimmer-30B-assistant` | `e8192f3a8f617f74be2ce220360c89ef4789f39f` | source repository | provenance for the DFlash companion |

All three repositories report Apache-2.0 and were public/non-gated at snapshot time. Re-query the API before
download; revision or byte drift creates a new experimental identity and requires an amendment.

Primary upstream references:

- `https://huggingface.co/meta-models/Muse-Glimmer-30B`
- `https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF`
- `https://github.com/ggml-org/llama.cpp/pull/26841`

## 3. Runtime admission boundary

The current local binaries cannot load this architecture:

| Local lane | Observed binary/source state | Muse architecture registered |
|---|---|---|
| deployed `/home/augus/src/slop.cpp` | build `b9863`, source `5e7f6271c` | no |
| fork-main lane `/home/augus/src/slop.cpp-main` | binary `b10159`; source checked at `87a416bd7` | no |

Upstream requires build `b10353` or newer; Muse support merged as llama.cpp PR `#26841` (`62bf73d`). Therefore:

1. use a new, explicitly named experimental worktree/build pinned at or after `b10353`;
2. record its exact commit and build options;
3. do not rebase, cherry-pick into, or rebuild the deployed `slop.cpp` lane for this scout;
4. do not alter `llm-inference.service` or the embedding service;
5. launch only on an explicit experimental port during controlled LAB maintenance;
6. run the fork qualification suite before any later proposal to carry Muse support into `lifecycle`.

`--jinja` is mandatory. Do not add `<|eom|>` as a stop token. Start with one slot because llama-server divides
the configured context across slots. Record the actual per-slot context from startup logs.

## 4. Questions and preregistered hypotheses

### H1 — practical 3090 fit

The official 17 GB quant should load in text-only, text+vision, text+DFlash, and text+vision+DFlash modes while
preserving the lab's 4 GB VRAM reserve at the qualified context. A role may qualify independently if the full
stack does not fit; do not average a failing arm into a passing role.

### H2 — independent agentic option

Muse should be non-inferior to the current Qwen3.8 IQ4_XS baseline on critical tool correctness and
irreversible-action safety, and should provide a useful Pareto option on at least one of agent success,
multimodal capability, latency, energy, or usable context.

### H3 — multimodal value

Muse should beat or complement the current Qwen3-VL-8B/Gemma-4-Vision lane on screenshot/UI/document tasks at
an acceptable memory and latency cost. Published benchmark claims are motivation only, never local evidence.

### H4 — target-matched DFlash

The official drafter should preserve deterministic output and improve task-correct end-to-end latency or
energy on at least one locally relevant workload. Published RTX 5090 speedups do not transfer to the RTX 3090.
Acceptance rate alone cannot promote the drafter.

### H5 — reasoning control risk

Muse's template always opens the reasoning channel. `low`, `medium`, `high`, and `xhigh` control strength, but
reasoning cannot be disabled. The candidate must therefore show that its lowest useful setting does not repeat
the lab's Qwen failure mode: verbose self-doubt, truncation, non-termination, or lower coding accuracy.

## 5. Experiment packages

### LAB-MUSE-000 — artifact and runtime admission

1. capture service, GPU, repository, and model baseline identities;
2. create and build the isolated `b10353+` runtime lane;
3. download only the revision-pinned 17 GB text model;
4. verify expected bytes, compute full SHA-256, inspect GGUF metadata/template, and confirm
   `general.architecture = muse-glimmer`;
5. require non-empty text completion, correct content/reasoning separation, valid tool-call parsing, and clean
   shutdown before downloading companions.

Stop on architecture refusal, malformed `to=self<|message|>` leakage, invalid tool parsing, unexplained
metadata drift, or inability to restore the original service.

### LAB-MUSE-001 — residency, performance, and additive-component matrix

Run in this order so every additional download and memory cost is gated:

| Arm | Text model | Perception encoder | DFlash | Purpose |
|---|---:|---:|---:|---|
| A | yes | no | no | clean text and residency baseline |
| B | yes | no | yes | text DFlash A/B |
| C | yes | yes | no | clean visual baseline |
| D | yes | yes | yes | full local multimodal-agent stack |

For every arm record load time, idle/peak VRAM, host RAM, page faults, TTFT, prompt throughput, task-correct
decode throughput, energy, thermal peak, context allocation, and exact argv/env. Test DFlash with
`n-max ∈ {4, 8, 15}` only after the off arm is valid.

### LAB-MUSE-002 — role-based quality packet

Text/agent comparator: qualified Qwen3.8 IQ4_XS. Visual comparators: Qwen3-VL-8B and Gemma-4-Vision where their
existing receipts are valid; rerun only the compact paired packet needed for comparability.

Required slices:

- existing eight-tool functional suite plus rephrase/reorder/rename/irrelevant-tool perturbations;
- failure recovery and irreversible-no-blind-retry cases;
- MBPP+ failure-focused subset including `Mbpp/260`, followed by a wider coding gate only if qualified;
- strict replay of the five Qwen3.8 GSM8K failures;
- OCR, error dialog, UI hierarchy, screenshot-diff, chart/document interpretation, and open-ended detection;
- multimodal tool calling where evidence in the image determines the selected tool and arguments.

Promotion is role-based. Muse may qualify as a multimodal agent without replacing Qwen3.8 for text-only code,
or qualify as a text agent without becoming the visual default.

### LAB-MUSE-003 — context, reasoning, and DFlash interaction

Use paired, invariant content at 8k/32k/64k/128k. Begin with one slot and increase concurrency only after the
single-slot context is verified. Exercise retrieval, multikey, multihop, aggregation, beginning/middle/end
placement, and many-similar-facts interference.

Cross the qualified context doses with reasoning strength `low/medium/high/xhigh`; deterministic and
model-recommended sampling tracks remain separate. Measure strict correctness, reasoning/final/total tokens,
truncation, non-termination, self-correction-to-error, time/joules per correct answer, and DFlash acceptance.

Upstream Muse support currently disables state save/load. Persistent slot checkpoint testing is therefore
`NOT_TESTABLE_RUNTIME` unless a later upstream build explicitly supports it. Cold/warm prefix behavior,
cancel-then-reuse, and session isolation remain required; do not mislabel an absent state API as model failure.

### LAB-MUSE-004 — multimodal safety and promotion

Test visual prompt injection, conflicting text/image instructions, tool arguments extracted from untrusted
screenshots, cross-session contamination, and irreversible actions. A model that blindly follows image-borne
instructions or retries irreversible operations is ineligible regardless of benchmark or throughput wins.

Freeze a final decision packet with model/component SHA-256 values, Hub revisions, engine commit, template
SHA, runtime vector, raw responses, invalid attempts, per-role decisions, and service-restoration receipt.

## 6. Gates and decisions

| Gate | Pass condition | Failure disposition |
|---|---|---|
| G0 identity | revision, bytes, SHA, license, template, and metadata fully recorded | `HOLD_IDENTITY` |
| G1 runtime | clean load, parsing, tool call, shutdown, and baseline restoration | `REJECT_RUNTIME` |
| G2 envelope | selected role fits without violating 4 GB VRAM and 16 GB Windows-RAM reserves | reject only the failing role/arm |
| G3 correctness | no critical agent, tool, session-isolation, or multimodal-safety regression | `REJECT_ROLE` |
| G4 quality | preregistered non-inferiority margins met on the role's task packet | `HOLD` or `REJECT_ROLE` |
| G5 DFlash | deterministic equivalence plus latency or energy gain on task-correct output | `DRAFT_REJECTED`; base may remain eligible |
| G6 context | no earlier context cliff than the role comparator inside the same memory envelope | cap qualified context or reject role |

Final decisions are `PROMOTE_TEXT_AGENT`, `PROMOTE_MULTIMODAL_AGENT`, `PROMOTE_DFLASH`, `HOLD`, or `REJECT`,
and may be combined. No weighted score can override an eligibility or safety failure.

## 7. Priority and execution order

This packet is the highest-priority new-model breadth scout for the RTX 3090. Execute it after controlled LAB
entry and before expensive full-packet or fork-lever campaigns:

1. `LAB-MUSE-000` runtime/artifact gate;
2. `LAB-MUSE-001` Arms A then C, proving the base roles before speculation;
3. `LAB-MUSE-001` Arms B then D, adding DFlash;
4. compact `LAB-MUSE-002` role gates;
5. `LAB-MUSE-003/004` only for roles that survive the compact gate;
6. expensive full coding/context matrices only for a promotable finalist;
7. restore and verify the original service after every maintenance tranche.

Do not download BF16, the 19.7 GB dynamic quant, community derivatives, or additional Muse variants during the
first screen. Those require a positive official-17GB result and a separate amendment.
