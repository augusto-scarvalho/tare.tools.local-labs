# HauhauCS PT-BR locale-control result

## Verdict

**PROMOTE CONTRACT V2 / DO NOT ABLATE OR TRAIN WEIGHTS.** A frozen generic
language contract eliminated the measured locale gap on a new blind panel:

- HauhauCS dev: 48/48;
- HauhauCS frozen test: 48/48;
- Fable-TC frozen test under the same contract: 48/48.

This satisfies the pre-registered operational sufficiency gate with zero overall
or category deficit. A weight intervention would now add cost and regression risk
without an observed error left to fix.

Selected contract: `contract_v2.txt`, SHA-256
`b66b8c553061be0fd8afb11a2096ead161bd1eb61e511c1b8e1ae9daa6bac143`.

## Why this is not an ablation

The original failure was conditional language adherence: semantically correct
answers such as `No` or an English clarification question were emitted for
Portuguese prompts. That is missing positive control, not an unwanted refusal
direction that can safely be subtracted.

The HauhauCS repository publishes only GGUF artifacts and identifies
`Qwen/Qwen3.8-27B` as the base. No trainable HauhauCS safetensors checkpoint was
available. A possible fallback was to train a locale LoRA on the official base
and transfer it at runtime to the related HauhauCS GGUF, but the perfect blind
contract result stopped that branch before downloads, package installation or
training.

Direct mixing with Fable-TC was rejected because Fable is a Qwen3.6-derived
checkpoint, not a same-base Qwen3.8 task vector.

## Frozen experiment

Two independent 48-task Portuguese panels were created before generation:

| Panel | SHA-256 | Use |
|---|---|---|
| Dev | `37515ad2...a189da737` | select a generic contract |
| Test | `75a0f623...c7b09632f` | one frozen candidate/Fable comparison |

Each panel contains 12 binary PT answers, 8 Portuguese lexical answers, 8
clarification questions, 8 reading/arithmetic answers, 8 exact structured-format
tasks and 4 constrained summaries. No prompt from the original normal-QA panel
was reused.

Both contracts reached 48/48 on dev after the declared grader correction. V2 was
selected before test because it was shorter: 370 decoded characters versus 513.

| Arm | Score | Median wall | Empty/malformed |
|---|---:|---:|---:|
| HauhauCS dev, v2 | 48/48 | 0.328 s | 0 |
| HauhauCS blind test, v2 | 48/48 | 0.328 s | 0 |
| Fable-TC blind test, v2 | 48/48 | 0.321 s | 0 |

All six categories were perfect in both blind arms.

## Grader correction

The first v1 dev summary reported 43/48. Manual inspection showed that all five
failures were valid short Portuguese clarification questions; the grader merely
required an arbitrary noun to be repeated. Before v2 or test generation:

- the original dev and untouched test task files were preserved with suffix
  `.v0-grader`;
- clarification grading changed to one terminal question, a Portuguese
  interrogative signal, no declared English phrase and at most 14 words;
- prompts and raw v1 generations remained unchanged;
- a derivative receipt regraded v1 as 48/48.

This correction removes false negatives rather than changing model behavior.

## Original-panel post-hoc confirmation

The selected contract was also replayed on the original 48 questions after the
blind test:

| Model | Old generic PT prompt | Selected contract v2 |
|---|---:|---:|
| HauhauCS | 43/48 | **44/48** |
| Fable-TC | **45/48** | 43/48 |

Under v2, HauhauCS corrected the original English `No`/clarification failures and
finished one point above Fable. The remaining failures were shared reading,
string-reversal and strict-summary-grader cases, not language drift. This replay
is explicitly post-hoc and does not replace the blind 48/48 tie.

## Operational integration

The raw llama-server stays on port 8080. `llm-locale-proxy.service` is installed
and active on loopback port 8082. It injects the exact selected contract into
OpenAI-compatible `/v1/chat/completions` requests and passes other paths through.

Use `http://127.0.0.1:8082/v1` for locale-controlled local clients. Existing
system instructions are retained after the contract. Streaming and non-streaming
requests were verified; responses include `X-Locale-Contract: qwen38-ptbr-v2`.

The proxy is intentionally loopback-only. No firewall exposure was added.

Validation:

- proxy unit tests: 5/5 pass;
- live `/health`: HTTP 200;
- live non-streaming PT answer: HTTP 200 with contract header;
- live streaming: six SSE events ending in `[DONE]`;
- service result `success`, `NRestarts=0`.

## Final serving state

Verified at 2026-08-23 17:59 -03:00:

- 8080: `fable-tc-l1.0`, context 8,192, engine b10159;
- 8081: embedding healthy, original PID 203666 preserved;
- 8082: locale proxy active on loopback;
- inference and embedding services active;
- no HauhauCS promotion, LoRA training, model download, commit or push.

Primary response hashes:

- HauhauCS blind test: `55de953e28ea9eb04ba5d3d4ec52296ed0311868736de356ccc8e81f23cfd746`
- Fable blind test: `db23eb3f5de51e67e92f6ea0e3143ea4b56fee9a5b2ec6b82b17307df39c8fb6`
- HauhauCS original replay: `a840c99c88fac0abf13b9415c0014457d00284c222b82886eae49fd030c96dfc`
- Fable original replay: `9fcef280b738461395642e7d56b223d308a2abc0d7d5e578e547050ef3a2a329`
