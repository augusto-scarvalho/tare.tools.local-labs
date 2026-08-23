# LAB-CLOSE-002 — Fable Fusion termination qualification

Frozen on 2026-08-22 before launching the experimental server.

## Question

Can Fable-Fusion-711 reliably terminate trivial requests across budget, EOS, explicit
stop and sampling controls, or must it be disqualified from a thinking-enabled
agentic role?

## Identity and runtime

- Artifact: `Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf`
- Bytes: 18,498,575,840
- SHA-256: `c796c2c011eaa0edf06395ff49cda5bfd4843ad52b86b58a83296dfc33849e4e`
- Engine: `slop.cpp` commit `5e7f6271c06b9104862ab799278a1b7f1323a449`
- One slot, 8,192 context, q4_0/q4_0 KV, FlashAttention, all GPU layers, Jinja,
  embedded MTP disabled so termination is attributed to the base artifact/template.

## Matrix

Four frozen trivial prompts per arm:

1. instruct greedy at caps 512 and 2,048;
2. thinking greedy at caps 512 and 2,048;
3. thinking sampled (`temperature=0.6`, `top_p=0.95`, seed 42) at caps 512 and 2,048;
4. thinking greedy cap 2,048 with explicit `stop=["</think>"]`;
5. instruct greedy cap 512 with `ignore_eos=true` as an EOS-mechanism diagnostic.

This yields 32 requests. `cache_prompt=false` throughout.

## Scoring and decision

- Use the qualified `flag_truncated` rule: `finish_reason=length` or predicted tokens
  reaching the cap is truncated/non-terminating.
- Natural termination requires `finish_reason=stop` without an explicit stop string.
- Explicit-stop termination is reported separately and is not natural EOS. It counts
  as role-safe only if non-empty final content exists; cutting at `</think>` with only
  reasoning is not a completed answer.
- The ignore-EOS arm is diagnostic and excluded from role scoring.
- Instruct is bounded-safe only at 8/8 natural terminations.
- Thinking-enabled agentic eligibility requires at least 95% natural termination over
  the 16 greedy+sampled thinking cells, with no prompt hitting `length` at both caps.
  Failure means `DISQUALIFIED` for the thinking-enabled agentic role; an independently
  safe instruct-only role may still be reported.
- Preserve every response and restore the canonical 8080 service afterward. Port 8081
  must remain healthy.

