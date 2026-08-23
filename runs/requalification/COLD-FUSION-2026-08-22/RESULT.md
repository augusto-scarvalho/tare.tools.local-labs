# Cold Fusion Qwen3.8 practical candidate — result

**Execution:** 2026-08-22  
**Status:** `EXECUTED / REJECT_BASE_ROLE / MTP_NOT_RUN`  
**Deployment decision:** no change; historical Qwen3.8 deployment restored.  
**Retained artifact:** revision-pinned candidate remains available for future, differently scoped research.

## Decision

The exact Cold Fusion `NEO-MTP-IQ4_XS` candidate is rejected as a general base-role replacement on this
RTX 3090 packet. It fits and is operationally clean, but fails three compact correctness discriminators:

- context exactness: 9/12, with aggregation failing at 8k, 32k, and 64k;
- `Mbpp/260`: terminated normally but failed both EvalPlus Base and Plus tests;
- historical GSM8K failure replay: 1/5, below the preregistered 3/5 floor.

The full 378-task MBPP+, 100-task GSM8K, replicated context packet, and embedded-MTP A/B were not opened.
This is the preregistered early stop, not missing evidence. MTP cannot rescue or promote a base artifact that
failed its role-correctness gate.

## Artifact admission

| Field | Value |
|---|---|
| repository | `DavidAU/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NM-DAU-NEO-MAX-MTP-GGUF` |
| revision | `27a5cb2cce434341c2a8a4a50130268e0eccae34` |
| file | `Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NM-DAU-NEO-MAX-NEO-MTP-IQ4_XS.gguf` |
| bytes | 17,033,680,384 |
| SHA-256 | `523bf4fbe2a2e0ce7aa54f812d85746294483b579443dd6e50e8ab684d7852f9` |
| local directory | `/home/augus/models/qwen38-27b/cold-fusion-27a5cb2c/` |
| architecture / training context | `qwen35` / 262,144 |
| tensors | 866; 64 base blocks plus block 64 `nextn` |
| MTP inventory | `eh_proj` Q8_0 plus three F32 norms |
| output tensor | BF16, shape 5,120 × 248,320 |

The repository revision changed after the prior assessment, but the target blob retained the exact frozen
size and SHA-256. The file was downloaded by immutable revision and independently hashed before loading.

## Runtime and residency

The no-spec arm used llama.cpp `b9863-5e7f6271c`, external `qwen-sharp.jinja`, full GPU offload,
FlashAttention, q4_0/q4_0 KV, one slot, 65,536 context tokens, and no vision projector.

Measured GPU memory was 19,126 MiB used and 5,197 MiB free, passing the 4,096 MiB reserve gate.

## Compact results

| Gate | Result | Decision |
|---|---:|---|
| agent functional suite | 8/8; no blind retry | PASS |
| cache/cancel/reuse | 4/4; all warm outputs matched cold oracles | PASS |
| context 8k | 3/4 | FAIL aggregation format |
| context 32k | 3/4 | FAIL aggregation format |
| context 64k | 3/4 | FAIL aggregation arithmetic |
| `Mbpp/260` | stopped at 180 tokens; Base fail; Plus fail | FAIL |
| five historical GSM failures | 1/5 strict; 5/5 format; 0 truncations | FAIL floor |

The aggregation instance had expected total `217`. At 8k and 32k the model replied
`19+54+45+65+34 = 217`, violating the explicit exact-only format. At 64k it replied `157`, so the long cell
also contains a substantive arithmetic error. Retrieval, multikey, and multihop were 9/9 across depths.

## Reasoning control

The fixed `gsm8k/153` control was correct and terminated in every valid mode:

| Mode | Completion tokens | Reasoning chars | Wall time | Strict |
|---|---:|---:|---:|---:|
| off | 182 | 0 | 4.953 s | PASS |
| low | 475 | 791 | 12.312 s | PASS |
| medium | 477 | 737 | 12.469 s | PASS |
| xhigh | 473 | 967 | 12.360 s | PASS |

On this control, enabling reasoning used about 2.6 times the completion tokens and 2.5 times the wall time
of instruct/off. It does not substantiate a reasoning-efficiency gain. One task cannot estimate the model
card's distributional claim, and the broad claim test was correctly stopped after eligibility failures.

The directories `reasoning-low/`, `reasoning-medium/`, and `reasoning-xhigh/` are preserved invalid attempts:
the client sent only `reasoning_strength`, while the Qwen template reads `reasoning_effort`, causing identical
outputs. The valid reruns are suffixed `-valid`; the harness now sends both compatible names.

## Restoration receipt

After stopping the experimental process:

- port 8092 had no remaining listener;
- `llm-inference.service` was active and healthy on port 8080;
- alias `qwen38-27b`, model `Qwen3.8-27B-UD-Q4_K_XL.gguf`, build `b9863-5e7f6271c`;
- context 131,072, one slot, q4_0/q4_0 KV, MTP `n-max=3`, and 32 context checkpoints;
- the independent embedding server remained healthy on port 8081 with its original PID 34282.

## Evidence map

- frozen design and stop rules: `PRE_REGISTRATION.md`
- agent and cache receipts: `nospec-agent.json`, `nospec-cache.json`
- context rows: `nospec-context.json`
- MBPP response, identity, EvalPlus raw result, and score: `nospec-mbpp260/`
- GSM failure replay: `nospec-gsm5/`
- valid reasoning controls: `reasoning-off/`, `reasoning-low-valid/`, `reasoning-medium-valid/`,
  `reasoning-xhigh-valid/`

**Authorized descriptive expansion, 2026-08-22:** Stage C was explicitly reopened without waiving the failed
base-role prerequisite. The nine-cell off/n2/n3 A/B completed but changed code/prose output bytes, slowed the
tiny answer, and truncated the prose oracle in every arm. Decision `MTP_REJECTED`; see
`../COLD-FUSION-MTP-2026-08-22/RESULT.md`.
