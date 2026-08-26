# Aborted during first real request

The active `fable-tc-l1.0` server returned `finish_reason=length`, 128 completion tokens entirely in `reasoning_content`, and an empty final `content` for the first frozen prompt. The exact `/tokenize` endpoint therefore returned zero pieces and the harness stopped before writing a receipt or any result.

Frozen implementation hashes are retained in `PIPELINE.json`. After transition to `BLOCKED`, the shared runner gained an explicit `max_tokens` argument so a clean successor can bind a larger budget under a new digest; the failed package itself remains unexecuted.
