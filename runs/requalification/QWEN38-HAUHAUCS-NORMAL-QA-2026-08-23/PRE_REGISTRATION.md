# Qwen3.8 HauhauCS versus vanilla — normal-question A/B pre-registration

## Question

Does `qwen38-hauhaucs-aggressive-q4kp` lose ordinary-answer quality relative
to `qwen38-27b-vanilla-q4xl` when both run in the same direct instruct regime?

## Frozen comparison

- 48 paired prompts in `tasks.jsonl`, fixed before either model is run.
- Categories: facts (10), math/logic (10), reading comprehension (8),
  structured instruction following (8), calibration (6), constrained summary
  (6).
- Portuguese-first, with a small English slice.
- Engine: b10165 (`71676e46c`), full GPU offload, FlashAttention, one slot,
  context 8,192, batch 2,048, ubatch 512, q4_0 KV.
- Sampling: greedy, `enable_thinking=false`, MTP off, cache reuse off.
- Same prompts, limits, grader and runner for both models.

## Scoring

- Primary: total deterministic rubric passes out of 48.
- Secondary: paired category counts, exclusive wins/losses, response-token
  median, wall-time median, malformed/empty responses.
- Exact-answer tasks use accent/case/punctuation-insensitive normalization.
- JSON and line-structure tasks are parsed structurally.
- Summaries require declared facts, reject declared hallucinations where
  applicable, and enforce their word limit.
- No LLM judge is used.

## Decision rule

- `NO_MEASURABLE_LOSS` if the candidate trails vanilla by at most two tasks
  overall and has no category deficit larger than one.
- `POSSIBLE_SMALL_LOSS` if it trails by three or four tasks, or any category by
  two, without severe failures.
- `MATERIAL_LOSS` if it trails by at least five tasks, produces malformed/empty
  answers, or loses any category by at least three.

This gate measures ordinary short-answer behavior, not current-world knowledge,
medical/legal safety, long-form writing taste, or open-ended agent behavior.
