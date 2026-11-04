# MiniCPM5-2B original: native compaction and tool calls

Date: 2026-09-14. Continuation of the MiniCPM investigation, with the **original
OpenBMB release**, as the operator clarified. The NPU2 derivative was not used.

## Result

Deterministic compaction works in the native Runtime, including with the real
model, without an auxiliary inference call. This does not qualify MiniCPM as an
autonomous implementer, nor does it show an end-to-end saving. The model is still
erratic at editing, at interpreting tools and at following the steps.

There were six experimental sessions and 140 physical generations, including four
summarization calls. No aggregate cap on calls, time, stagnation or compaction
was applied; there were zero retries. The produced code was not repaired by hand.

## Observed configuration

- Official repository: `openbmb/MiniCPM5-2B-GGUF`.
- Revision: `2079a22f3beaa4e306449978533478fe0522f4b3`.
- File: `MiniCPM5-2B-Q8_0.gguf`.
- SHA-256: `c5415f8989bf88a8288f1b55a3cc371af53c07b0faa220a63bd7a990cfaba078`.
- slop.cpp `b10165-71676e46c`, the artifact's Jinja template, thinking enabled.
- `/props`: effective window **32768**, one session; this is not a claim about the
  original model's maximum. The reply identity was checked on every generation.
- GPU, flash attention, KV K/V `q8_0`, batch 512, ubatch 128; seed 42;
  temperature 1, top-p .95, top-k 40, min-p 0, repeat penalty 1.05, repeat-last-n 64.
- Output per reply 4096; trigger .8, target .5, margin 64, last full turn
  preserved. The server cache stayed available between runs.

## Implementation

Runtime: `ContextPolicy.compaction` accepts `model` (the compatible default) or
`deterministic`. The host injects the projector; there is no import of the Kernel/OS.
The whole candidate is counted before the atomic activation. Instructions, required
sources, explicit user messages, recent turns and the original journal stay
preserved. Excess state is not silently dropped to force admission. Resuming
neither invents inference nor repeats effects.

Kernel: the `native-task-state/1` projector derives observed files/hashes, edits,
checks, searches, refusals and uncertain effects. It states that it does not
summarize earlier semantic decisions. Refusals for incomplete arguments now
report missing/unexpected fields and valid repositories. The final descriptions
explain root-relative paths and line numbers that are for display only.
OpenAI-compatible summarization gets its own instructions and offers no tools;
the Converse contract was not changed by this change.

OS: explicit selection in `context_policy.compaction`. The new mode is optional;
the tests do not justify making it mandatory for all work.

Code kept locally, with no publication or global install in this round:

| Repository | Base | Delivery |
|---|---|---|
| agent-runtime | `2554110` | `978bd981` |
| kernel | `cac5099` | `c027c60a` |
| os | `ee12f0f` | `aac4731f` |

Shared branch: `fix/deterministic-context-20260914`. The implementation must be
integrated/published with compatible versions of the three packages; earlier
published minimums do not deliver the new mode. The old checkouts with WIP were kept.

Final validation: Runtime **209** tests, Kernel **109** focused, OS **54** native
integration; Runtime `compileall` and `git diff --check` passed.

## Real sessions

Directed tasks with explicit paths: no discovery, SpecGraph or broad-search
comparison. Checkers were frozen before each run.

The pressure test uses three 200-line files with markers, requests in stages and a
final edit of `answer.json`. It is a synthetic rehearsal of context accumulation and
recovery. The checker inspects the file; the journal was also inspected to detect a
premature edit. That extra audit was done after the runs, without turning failures
into passes.

| Variant | Task/mode | Generations | Session seconds | Input + output tokens | Compactions / LLM summaries | Result |
|---|---|---:|---:|---:|---:|---|
| Initial | Unicode / deterministic | 4 | 7.17 | 6414 | 0 / 0 | File and 6 test groups correct |
| Initial | Duration / deterministic | 63 | 131.14 | 836600 | 0 / 0 | Failed; ended by proposing a delivery with a SyntaxError |
| Initial | Pressure / deterministic | 19 | 49.45 | 273195 | 2 / 0 | File correct; no early edit |
| Initial | Pressure / model | 11 | 47.81 | 140226 | 1 / 1 | Valid summary; edit refused for a wrong path; file empty |
| Clear descriptions | Pressure / deterministic | 23 | 77.56 | 339399 | 3 / 0 | File correct, but it tried to edit before the authorized stage |
| Clear descriptions | Pressure / model | 20 | 80.31 | 264613 | 2 / 3 | File correct, edited prematurely; session blocked by a candidate that was too large |

The last two runs are not full passes. A one-off improvement in a call is not a
demonstrated improvement of the whole task. In these two rehearsals the final
descriptions removed the refusals for missing fields and for a path with a
repository prefix, but they did not remove behavioral errors.

The parser received 7 valid and 14 invalid examples, including `2ms`, `1hour`,
`2ss`, `2m garbage` and `1hfoo`. There were 29 refusals for missing arguments and five
for a search without a unique match. The wrong code and the incomplete arguments
came from the model; the refused edits were not applied. The final proposal was
not treated as approval of the work.

## Compaction and cost

In the first deterministic pressure session, requests went from
**29159 to 10598** and from **28610 to 10532** tokens. In the final variant they were
**29312 → 10528**, **30308 → 10730** and **29432 → 10642**. There was no LLM summary.
The final model variant triggered two summaries; the third candidate did not fit,
because the last turn held three large reads that had to stay whole.

The configured trigger was 26214 tokens (80%). The check runs before the next
call, after the tool result arrives; one read can take the request from under 80%
to 89–92%. Those requests were compacted before sending, not dispatched with the
excess context.

The deterministic checkpoint dropped prose and caused rereads; one of the model
summaries kept the markers correctly. So "zero summary calls" does not mean
"lower total cost". The weighted decode rates were about 142–190 tok/s; these are
server rates, not rates of the whole flow.

Tool-argument tokens are in the provider's output, and tool results appear in the
following inputs. They were not added again. Real summaries are in the same total.
Cache, counts and effects remain in the receipts. The session timer excludes
fixture preparation and the final independent check; energy was not measured. The
rehearsal supports neither a percentage of end-to-end saving nor a causal inference
from one trajectory per variant.

## Evidence and next use

The summary, patches, Git references and receipts were kept in the maintainer's
private evidence store (`.reconciliation/evidence-store/reports/native-deterministic-20260914/`);
these raw records are not published in this repository. `RESULTS.json` identifies
each run; requests, replies, journals, profiles and checkers are kept in the
consolidated archive that `retention.json` points to.

The temporary server was shut down. The Qwen backend stayed healthy with the same
PID. The duplicate 2.68 GB copy of the weights was removed after checking the
historical archive that can restore it, without creating another weights file.

Prudent use of this evidence: keep MiniCPM as a candidate for small tasks with a
verifiable result and a narrow scope. Its quality as a subagent under real
parallelism is still unqualified. Do not make deterministic compaction a universal
default, and do not try to make up for every reasoning failure with more harness rules.
