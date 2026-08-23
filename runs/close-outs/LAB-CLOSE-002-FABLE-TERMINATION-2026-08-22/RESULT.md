# LAB-CLOSE-002 — Fable Fusion termination qualification

**Status:** `COMPLETE / INSTRUCT-ONLY SAFE / THINKING AGENTIC DISQUALIFIED`

Fable-Fusion-711 terminates reliably on the four bounded trivial prompts only when
thinking is disabled and normal EOS handling is allowed. Thinking mode naturally
terminated just 6/16 greedy+sampled cells (37.5%). Larger budgets and sampling did
not repair the repeated loops, and an explicit `</think>` stop did not produce final
answers. The artifact is disqualified for a thinking-enabled agentic role.

## Results

| Arm | Natural stop | Length | Final content | Predicted tokens by prompt order |
|---|---:|---:|---:|---|
| instruct greedy, cap 512 | 4/4 | 0/4 | 4/4 | 2, 7, 4, 21 |
| instruct greedy, cap 2048 | 4/4 | 0/4 | 4/4 | 21, 2, 7, 4 |
| thinking greedy, cap 512 | 1/4 | 3/4 | 1/4 | 512, 512, 512, 143 |
| thinking greedy, cap 2048 | 2/4 | 2/4 | 2/4 | 143, 2048, 1674, 2048 |
| thinking sampled, cap 512 | 1/4 | 3/4 | 1/4 | 512, 512, 118, 512 |
| thinking sampled, cap 2048 | 2/4 | 2/4 | 2/4 | 2048, 1584, 2048, 118 |
| thinking + explicit `</think>` stop, cap 2048 | 0/4 natural; 2/4 forced | 2/4 | **0/4** | 1669, 2048, 140, 2048 |
| instruct + ignore EOS, cap 512 | 0/4 | 4/4 | 4/4 partial | 512, 512, 512, 512 |

The order in each row is the executed rotated order; task identities are retained in
`receipts.jsonl`.

Two prompts—Python one-liner and the 17-prime question—hit `length` in thinking mode
at both 512 and 2,048. “Say hello in exactly three words” was only rescued after
1,674 greedy or 1,584 sampled tokens, which is termination but not practical bounded
agent behavior. The Paris prompt consistently terminated around 118–143 thinking
tokens.

The explicit-stop arm exposes an important scoring trap: two cells reported
`finish_reason=stop`, but all four had empty final `content`. The stop either cut the
reasoning at `</think>` or never appeared before the token limit; it cannot be counted
as a completed answer. The ignore-EOS diagnostic made even safe instruct responses
run to 512, confirming that the natural instruct result is real EOS-driven termination.

## Decision

- **Instruct-only:** bounded termination-safe on this 8-cell trivial-prompt panel.
- **Thinking-enabled agentic role:** **DISQUALIFIED**. Natural termination 37.5% is
  far below the preregistered 95% gate, with repeated 2,048-token loops.
- Do not use stop strings to mask the failure; they can create `stop` receipts without
  a final answer.
- This close-out does not reopen the artifact's weak historical 40% HumanEval+ result
  and does not promote an instruct role on quality grounds.

## Identity, validity and restoration

- Artifact bytes: 18,498,575,840; SHA-256
  `c796c2c011eaa0edf06395ff49cda5bfd4843ad52b86b58a83296dfc33849e4e`.
- Engine: `5e7f6271c06b9104862ab799278a1b7f1323a449` (`b9863`).
- Runtime: one 8,192-token slot, q4_0/q4_0 KV, all GPU layers, FlashAttention,
  Jinja, no speculative MTP.
- 32/32 requests completed operationally and were scored with the qualified
  `flag_truncated` primitive.
- The isolated experimental service was stopped. Canonical Qwen3.8 is active/healthy
  on 8080, embedding is healthy on 8081, and board power remains 420 W.

## Evidence

- `PRE_REGISTRATION.md`: frozen identity, matrix and role gate.
- `receipts.jsonl`: all raw OpenAI responses and derived fields, SHA-256
  `4e1c7fdd4d5a40d0760cc5ad6adaf30b75132142ce50df27fdba6234382cb359`.
- `summary.json`: aggregate decision, SHA-256
  `ca474cd1cbe027f8d929561871af523ad3cf431fb6e6b1f88eb698d3317ef448`.
- `server.log`: experimental server receipt, SHA-256
  `d7be09ba9fe526d617ec7d7ab8cdfbf343ec7d04b9836528158688fdf9211505`.
- Harness: `tools/benchmarks/fable_termination_matrix.py`.
- Launcher: `tools/scripts_sh/launch_fable_termination_server.sh`.

