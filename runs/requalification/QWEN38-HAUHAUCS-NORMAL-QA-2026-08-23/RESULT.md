# Qwen3.8 HauhauCS versus vanilla and Fable-TC — normal-question result

## Bottom line

**Operational update:** a subsequently frozen, stronger generic locale contract
reached 48/48 for both HauhauCS and Fable-TC on a new blind Portuguese panel. On
this original panel it produced 44/48 for HauhauCS and 43/48 for Fable, with no
remaining candidate-only language-drift failure. This supersedes the weaker
prompt as the recommended mitigation, while leaving every primary no-prompt
conclusion below intact. See
`../QWEN38-HAUHAUCS-LOCALE-CONTROL-2026-08-23/RESULT.md`.

There are two distinct answers:

1. **Without a language system prompt: `MATERIAL_LOSS` under the frozen strict
   Portuguese rubric.** HauhauCS scored 38/48 versus vanilla's 43/48. All five
   exclusive candidate failures were semantically correct responses emitted in
   English to Portuguese prompts, so this is a real locale/instruction-adherence
   loss, not evidence of five missing pieces of knowledge.
2. **With the same verified UTF-8 language instruction on both models:
   `NO_MEASURABLE_LOSS`.** HauhauCS scored 43/48 versus vanilla's 44/48. It
   trailed by one overall and by no more than one in any category, satisfying
   the frozen no-measurable-loss rule.

For an English-first or content-only workload, this panel gives no evidence that
HauhauCS is generally less intelligent. For a Portuguese-first endpoint with no
fixed system instruction, vanilla is materially more reliable about staying in
the requested language.

The later same-task Fable-TC arm sharpens the operational answer:

- without a system prompt, Fable-TC scored **44/48** versus HauhauCS's **38/48**;
- with the same verified UTF-8 language prompt, Fable-TC scored **45/48** versus
  HauhauCS's **43/48**.

Thus Fable-TC is the strongest of the three on this short Portuguese-first panel.
HauhauCS still has the stronger measured coding result: 56/60 HumanEval+ versus
Fable-TC's historical 53/60 on the exact same subset.

## Frozen primary comparison — no system prompt

Both models received the same 48 prompts, greedy sampling, direct instruct mode,
context 8,192, MTP off, engine b10165, and deterministic graders.

| Category | Vanilla | HauhauCS | Candidate delta |
|---|---:|---:|---:|
| Facts | 10/10 | 9/10 | -1 |
| Math/logic | 10/10 | 9/10 | -1 |
| Reading | 6/8 | 5/8 | -1 |
| Instruction | 7/8 | 7/8 | 0 |
| Calibration | 6/6 | 5/6 | -1 |
| Summary | 4/6 | 3/6 | -1 |
| **Total** | **43/48** | **38/48** | **-5** |

Paired outcomes: 38 both pass, 5 both fail, 5 vanilla-only passes and 0
candidate-only passes. The two-sided exact McNemar/binomial p-value is 0.0625.
The score difference meets the pre-registered `MATERIAL_LOSS` threshold even
though this small panel does not cross a conventional 0.05 significance cutoff.

Manual error inspection was performed only after scoring. The candidate's five
exclusive misses were `Mitochondria`, `No`, `No`, an English clarification
question, and an English but otherwise correct summary. This post-hoc review
localizes the observed loss to Portuguese-language adherence; it does not replace
the primary deterministic score.

## Matched operational mitigation — identical UTF-8 system prompt

After the primary result was frozen, both models were replayed once with the
same instruction loaded from `system_prompt_pt.txt`. The stored prompt in both
clean summaries is byte-equivalent after UTF-8 decoding to that file.

| Category | Vanilla + prompt | HauhauCS + prompt | Candidate delta |
|---|---:|---:|---:|
| Facts | 10/10 | 10/10 | 0 |
| Math/logic | 10/10 | 9/10 | -1 |
| Reading | 7/8 | 6/8 | -1 |
| Instruction | 7/8 | 7/8 | 0 |
| Calibration | 6/6 | 5/6 | -1 |
| Summary | 4/6 | 6/6 | +2 |
| **Total** | **44/48** | **43/48** | **-1** |

Paired outcomes: 41 both pass, 2 both fail, 3 vanilla-only passes and 2
candidate-only passes. The two-sided exact McNemar/binomial p-value is 1.0.
Both medians were 0.437 s wall time and 3 answer tokens; both answered 48/48.

The mitigation is useful but imperfect: HauhauCS still answered `No` on two
Portuguese exact-answer tasks and emitted one English clarification question.
Therefore the recommended deployment rule is to keep the Portuguese language
instruction fixed at the gateway/client and accept a small residual locale edge
for vanilla. The candidate's summary score was stronger in this replay.

## Fable-TC same-task comparison

Fable-TC was added after the original HauhauCS-versus-vanilla question was
answered. It used the same 48 frozen tasks, b10165 executable, instruct template,
greedy sampling, context 8,192, MTP off, q4_0 KV cache and deterministic grader.

| Category | Fable no prompt | HauhauCS no prompt | Fable + prompt | HauhauCS + prompt |
|---|---:|---:|---:|---:|
| Facts | 10/10 | 9/10 | 10/10 | 10/10 |
| Math/logic | 10/10 | 9/10 | 10/10 | 9/10 |
| Reading | 7/8 | 5/8 | 8/8 | 6/8 |
| Instruction | 7/8 | 7/8 | 7/8 | 7/8 |
| Calibration | 6/6 | 5/6 | 6/6 | 5/6 |
| Summary | 4/6 | 3/6 | 4/6 | 6/6 |
| **Total** | **44/48** | **38/48** | **45/48** | **43/48** |

No-prompt paired outcomes were 38 both pass, 4 both fail, 6 Fable-only and 0
HauhauCS-only; two-sided exact McNemar/binomial p=0.03125. This is a significant
and `MATERIAL_LOSS` result under the frozen rule.

With the prompt, paired outcomes were 41 both pass, 1 both fail, 4 Fable-only and
2 HauhauCS-only; two-sided exact p=0.6875. HauhauCS trails by two overall, but its
two-task reading deficit makes this `POSSIBLE_SMALL_LOSS` rather than
`NO_MEASURABLE_LOSS` under the frozen category rule. HauhauCS wins the constrained
summary category 6/6 to 4/6; Fable wins reading 8/8 to 6/8.

The practical profile is therefore:

- **Fable-TC:** better default for ordinary short Portuguese requests, and it
  does not depend on a language system prompt to remain stable;
- **HauhauCS:** better measured coding model and much closer to Fable on ordinary
  questions once the language instruction is fixed, but still less reliable on
  terse Portuguese exact answers and reading;
- **Vanilla Qwen3.8:** between them on the raw panel (43/48) and essentially tied
  with HauhauCS under the prompt (44/48 versus 43/48).

The available serving-speed figures are not a controlled A/B: HauhauCS measured
91.37 tok/s median over five 256-token coding generations, while the Fable-TC
serving receipt records 77.11 tok/s on one 128-token deterministic generation.
They suggest HauhauCS may decode faster in these prepared profiles, but do not
establish a matched throughput delta.

## Invalidated exploratory runs and harness correction

The first accented prompt attempt (`pt-system`) suffered U+FFFD corruption while
PowerShell transported the text through argv. Two following files tagged
`pt-system-utf8` are also invalid: an already-running Fable-TC occupied port 8080,
and the old runner mistook its health response for the requested model. All three
are retained as receipts and excluded from conclusions.

Before the clean replay, the runner was hardened to:

- refuse to start when its port is occupied;
- verify `/props.model_path` against the requested registry model after health;
- read the system prompt directly from a UTF-8 file;
- stop only processes containing the experimental profile's exact model path,
  rather than using a host-wide `pkill -f llama-server` that could kill the
  independent embedding server.

The clean arms are identified by tag `pt-system-utf8-clean`. The embedding PID
remained 203666 through both clean arms and the final restoration.

## Evidence identity

- Task set: `tasks.jsonl`, SHA-256
  `56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f`
- Primary vanilla responses: SHA-256
  `da4bd5d62edbba403d74bf8ac43c6f366e75cbdb37449801dd51f46c0129be48`
- Primary candidate responses: SHA-256
  `34719bad3bf7bc0dd9d8e4a95a20f8402cedd3f8dcbab4dd01e9ae2f0dddda60`
- Clean prompted vanilla responses: SHA-256
  `105869310686890e4d210c3cf50173fe93d599ac5ace204a472cf6ef14698a70`
- Clean prompted candidate responses: SHA-256
  `0f32d3da93e60ff630d63bffb4d9156102fa90c253ef8a3956e3dff402567fd3`
- Fable-TC no-prompt responses: SHA-256
  `4f08bad5a74e43d98ea2e1d796daa31dc22f100b3c592deeef2fd3eed9b36d4c`
- Fable-TC clean prompted responses: SHA-256
  `9443128b09f048664f1e1062aeb61b0aba050f1f967d3ac0ac7934de2f6cbb52`
- Engine: llama.cpp/slop.cpp b10165, commit `71676e46c`

## Final serving state

Verified at 2026-08-23 17:36 -03:00:

- port 8080: `fable-tc-l1.0`, context 8,192, engine b10159;
- port 8081: Nomic Embed Text v1.5 healthy, original PID 203666 preserved;
- `llm-inference.service`: active/running, result `success`, `NRestarts=0`;
- temporary HauhauCS systemd drop-in: absent.

No commit or push was performed.
