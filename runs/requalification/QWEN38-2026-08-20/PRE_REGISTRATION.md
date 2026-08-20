# QWEN38-2026-08-20 — evidence requalification

**Status:** PRE-REGISTERED, before outcome-bearing runs.  The preliminary 2026-08-20 MQAR,
NIAH, and GSM8K artifacts remain append-only historical scouts; this campaign writes to a new
namespace and does not overwrite them.

## Declared design

Applicable method cards: deterministic scorer qualification, dose-response, blocked order,
negative controls, exact substrate identity, and finite-sample intervals.  This is a single-model
requalification, not a model promotion and not an engine A/B.  Evidence can reach grade 2
(attributive) here; grade 3 requires an independently repeated run or a frozen second substrate.

Hard constraints are evaluated lexicographically: valid instrument and stable server identity
first; only valid rows enter correctness; throughput is descriptive and cannot compensate for a
quality failure.

## Shared factors

- **control:** workload pressure (`P`, prompt-token target, or fixed task subset).
- **noise_context:** probe depth and run order; explicitly blocked and balanced.
- **hard_to_change:** live llama-server build, GGUF, quant, KV format, context limit, chat template.
- **nuisance:** cache warmth and concurrent host load; prompt cache is disabled and order alternates.
- **prohibited:** changing model/server/template during a run. Identity drift invalidates the run.

Every runner records exact source SHA-256, Git HEAD, model GGUF SHA-256, llama.cpp build, process
argv, model metadata, context limit, and chat-template SHA-256 before and after measurement.

## RQ-1 — MQAR strict capacity/depth curve

- **hypothesis:** Qwen3.8-27B retains exact associative mappings through `P=2048` within the live
  32k context, without a depth-specific cliff.
- **metric:** complete-reply exact match; format adherence is separate. Wilson 95% intervals expose
  denominators. Substrings, prose wrappers, and partial matches are failures.
- **baseline:** `P=4`, same deterministic fixture family and five probe depths.
- **design:** `P={4,32,128,512,1024,2048}`, depths `{.10,.25,.50,.75,.90}`, 8 replicates/depth
  (`n=40/P`); block order alternates forward/reverse.
- **successCriteria:** low-dose exact >=95%; no high-pressure dose below 90%; no depth cell shows a
  reproducible isolated collapse. These are requalification thresholds, not a deployment promotion.
- **abandonCriteria:** low-dose competence <95%, server identity drift, scorer self-test failure, or
  prompt exceeding the server context. A high-dose drop opens a bounded confirmation at the adjacent
  doses; it does not authorize mechanism work immediately.
- **reversalPlan:** measurement-only; stop requests. No runtime or model mutation.

## RQ-2 — token-calibrated NIAH

- **hypothesis:** exact retrieval remains high through 30,000 *actual chat-template tokens* and is
  not an artifact of needle depth or code hallucination.
- **metric:** complete-reply exact code match. Negative prompts must return exact `NOT_PRESENT`.
- **baseline:** 4,096 actual prompt tokens.
- **design:** targets `{4096,8192,16384,24576,30000}`, depths `{.10,.50,.75,.90}`, 3 positive
  replicates/depth plus 2 negative controls/target. Prompt length is calibrated through
  `/apply-template` then `/tokenize`, tolerance <=64 tokens below target.
- **successCriteria:** >=95% overall positive retrieval, 100% negative controls, and no target/depth
  cell with repeated failures.
- **abandonCriteria:** token calibration is unstable, false positives occur, server identity drifts,
  or the prompt approaches the context limit without the declared reserve.
- **reversalPlan:** measurement-only.

## RQ-3 — GSM8K seeded strict subset

- **hypothesis:** the preliminary 29/30 signal survives a pre-declared seeded sample without relying
  on the first rows or a permissive last-number fallback.
- **metric:** primary strict accuracy requires the final non-empty line to be exactly
  `#### <number>`; lenient last-number accuracy is diagnostic only. Format adherence and truncation
  are separate.
- **baseline:** preliminary 29/30 is historical context only, not a statistical control arm.
- **design:** deterministic shuffled `n=100` subset from the canonical 1,319-row test export,
  seed `20260820`, greedy instruct mode, max 512 tokens. Dataset file and score-bearing logical
  content (including gold answers) are hashed.
- **successCriteria:** Wilson lower 95% bound >=85%, format adherence >=98%, truncation <=1%.
- **abandonCriteria:** dataset/identity mismatch, scorer self-test failure, >1% truncation, or server
  drift. A strict-vs-lenient gap is an interface failure and is not silently repaired.
- **reversalPlan:** measurement-only.

## Ordering and commit boundary

The scorer changes, self-tests, runner, and this pre-registration are committed before any
outcome-bearing request. Results are committed separately. Nothing is pushed automatically.
