# Qwen3.8 HauhauCS Aggressive requalification result

## Verdict

**QUALIFIED CANDIDATE / NOT AUTO-PROMOTED.** The revision-pinned Q4_K_P
candidate met the two requested primary criteria in the tested coding/instruct
regime:

- it scored **56/60 (93.3%)** on the exact historical HumanEval+ subset, versus
  **53/60 (88.3%)** for Fable-TC and **57/60 (95.0%)** for vanilla Qwen3.8;
- on the paired 44-prompt benign over-refusal panel it produced **44/44 comply,
  0 hedged, 0 refuse**, versus vanilla Qwen3.8's **24 comply, 18 hedged, 2
  refuse**.

This is evidence of a material coding-accuracy improvement over Fable-TC and a
large reduction in alignment friction versus vanilla Qwen3.8. It is not yet a
claim of universal intelligence superiority outside the measured coding regime.

The candidate was deliberately not promoted. The final active endpoint was
restored to Fable-TC, preserving the prior serving decision until the owner
chooses otherwise.

## Frozen artifact identity

- Source: `HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF`
- Revision: `993a5971fda8f30dd1b7eb2654792ba4415c7460`
- File: `Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf`
- Bytes: `17,923,393,664`
- SHA-256: `ba36dc3c2b2ff5e0aa5d71092a8894546996a6a119ae391803dda07cdc08516d`
- Release-manifest signature: verified with the published Ed25519 key before
  download.
- GGUF metadata observed live: 27,320,697,856 parameters, native context
  262,144, embedded chat template and embedded NextN/MTP head.

Engine used for runtime and tests:

- `llama-server` b10165, commit `71676e46c`
- SHA-256: `efb2f06c19d26605a1934c0a9ed5b65dd69034e8765f2d29d0426b7a011cfbe2`

The optional third-party FastMTP sidecar/patch was not installed or used. All
MTP measurements below use the GGUF's native embedded head on the already
qualified local engine.

## Runtime gates

The familiar Qwen3.8 profile was applied through a temporary systemd drop-in:

- full GPU offload; FlashAttention on;
- one slot; batch 2048; ubatch 512;
- KV cache q4_0; 32 context checkpoints;
- native MTP, `draft-n-max=3`;
- Jinja, metrics, no multimodal projector;
- embeddings remained active and independently healthy on port 8081.

Results:

| Gate | Result |
|---|---:|
| 32k load | PASS, healthy in 6 s |
| 32k idle VRAM reserve | 4,054 MiB |
| 23,449-token needle | PASS, exact `JADE-7319` |
| 23k prefill | 1,239.0 tok/s |
| 131,072 context load | PASS, healthy in 6 s |
| 131k idle VRAM reserve | 1,402 MiB |
| 72,049-token needle | PASS, exact `JADE-7319` |
| 72k prefill | 1,005.4 tok/s |
| Embedding endpoint during both gates | HTTP 200 |

The 131k profile therefore fits, but its approximately 1.4 GiB idle reserve is
tight and should remain a single-slot profile.

## Coding intelligence gate

Configuration:

- exact nested `market-r0` 60-task subset; subset equality against the recorded
  Fable run was checked before generation;
- official EvalPlus scorer and sandbox;
- instruct mode (`enable_thinking=false`), greedy decode, context 16,384;
- MTP off during quality generation to avoid mixing a model-quality result with
  speculative-decoding behavior;
- maximum 6,144 generated tokens; all 60 answered; no reasoning-token or empty
  answer failures.

| Model | HumanEval base | HumanEval+ | Delta vs Fable-TC |
|---|---:|---:|---:|
| Fable-TC historical | 54/60 (90.0%) | 53/60 (88.3%) | reference |
| **HauhauCS Aggressive Q4_K_P** | **58/60 (96.7%)** | **56/60 (93.3%)** | **+3 / +5.0 pp** |
| Vanilla Qwen3.8 historical | 59/60 (98.3%) | 57/60 (95.0%) | +1 vs candidate |

Candidate HumanEval+ failures: `HumanEval/77`, `HumanEval/97`,
`HumanEval/145`, and `HumanEval/154`.

Generation-side observations with speculative decode disabled: median 41.0
tok/s, median 173.5 generated tokens, 60/60 answered.

Primary evidence:

- `runs/a2/hauhaucs-r0-instruct__qwen38-hauhaucs-aggressive-q4kp__humaneval.json`
  (`d8b1095e...e958b2`)
- `runs/a2/hauhaucs-r0-instruct__qwen38-hauhaucs-aggressive-q4kp__humaneval__samples.padded_eval_results.json`
  (`5e0b622e...c51a3`)

## Alignment-friction gate

Both candidate and vanilla Qwen3.8 were run locally with the same b10165 engine,
instruct mode, context 8,192, greedy sampling, 256-token cap and the same 44
benign-but-commonly-over-refused prompts.

| Model | Comply | Hedged | Refuse | Discriminating tier comply |
|---|---:|---:|---:|---:|
| **HauhauCS Aggressive** | **44/44** | **0/44** | **0/44** | **24/24** |
| Vanilla Qwen3.8 | 24/44 | 18/44 | 2/44 | 8/24 |

The first pass of the old classifier reported three candidate refusals. Manual
inspection showed that all three were in-character answers mocking quoted
refusal phrases such as `"I can't help with that"`. The classifier was corrected
to remove short quoted spans before testing assistant-voice refusal signatures;
the same corrected classifier was then applied to both files. Its self-check
passes. The raw generations remain unchanged.

Evidence:

- `runs/a2/refusal__hauhaucs-r0-instruct__qwen38-hauhaucs-aggressive-q4kp.json`
  (`bf128b4a...bc6dd`)
- `runs/a2/refusal__vanilla38-r0-instruct__qwen38-27b-vanilla-q4xl.json`
  (`56d2ec1e...a7ba`)

## Native-MTP serving speed

On the final 131k candidate profile, five post-warmup forced 256-token coding
generations produced:

- median: **91.37 tok/s**;
- mean: **91.13 tok/s**;
- native-MTP acceptance: **940/995 = 94.47%**;
- individual range: 90.27-91.83 tok/s.

This speed result measures the candidate with native MTP enabled. It is not an
A/B against the optional FastMTP sidecar and does not justify installing that
third-party patch.

## Final serving state

Verified at 2026-08-23 17:07 -03:00:

- port 8080: `fable-tc-l1.0`, context 8,192, b10159;
- port 8081: embedding health HTTP 200;
- `llm-inference.service` and `llm-embedding.service`: active;
- inference service result `success`, `NRestarts=0`;
- temporary `serve-hauhaucs-aggressive.conf`: absent.

The candidate and its two prepared 32k/131k drop-ins remain on disk for an
explicit future promotion. No commit or push was performed.

## Remaining optional work

The normal-question gate is now complete. It found a Portuguese-language
adherence loss without a system prompt (38/48 versus vanilla's 43/48), reduced
to no measurable loss with a matched fixed language prompt (43/48 versus 44/48).
The subsequent Fable-TC control scored 44/48 without and 45/48 with that prompt;
against Fable, the prompted candidate is classified `POSSIBLE_SMALL_LOSS` because
of a two-task reading-category deficit despite trailing by only two overall.
See `../QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/RESULT.md`.

Additional work before making it a broad default would be:

1. agent/tool-calling regression suite against vanilla Qwen3.8;
2. math/general-reasoning gate (for example the existing GSM8K-200 portfolio);
3. a multi-hour reliability soak at 131k while embeddings remain active;
4. only after those pass, an isolated native-MTP versus FastMTP A/B.

The Portuguese locale branch is now closed without a weight edit. The selected
generic contract scored 48/48 for both HauhauCS and Fable-TC on a new frozen blind
panel and is available through the loopback locale proxy on port 8082. See
`../QWEN38-HAUHAUCS-LOCALE-CONTROL-2026-08-23/RESULT.md`.
