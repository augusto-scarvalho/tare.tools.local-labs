# A2 Stage-2 methodology addendum: answer-channel refusal gate

Date: 2026-08-23

The first E2 baseline used the model's default chat template and a frozen 32-token generation
budget. The template opened a `<think>` block, so all sampled completions ended inside reasoning
preamble before the answer/refusal. The observed harmful refusal rate of 0.00 therefore did not
measure the pre-registered behavioral quantity. The interrupted sweep is invalid calibration
evidence and is retained in `SELECT.log` and `refusal_diagnostic_thinking_default.json`.

Corrective rule, frozen before examining any layer intervention result:

- use the tokenizer's official `enable_thinking=False` template option;
- retain the 32-token greedy budget and the existing refusal detector;
- re-extract all base activations at the resulting answer-channel post-instruction position;
- rerun a raw 8 harmful / 4 harmless diagnostic before E2;
- run E2 only if the corrected baseline has measurable harmful refusal.

This changes measurement validity, not a layer, threshold, direction, or outcome after inspection.

After layers 8--11 showed identical rejection and exposed an avoidable serial-forward bottleneck,
generation and KL evaluation were first batched at eight prompts, then at the full frozen 32-prompt
validation split after the eight-row run showed ample memory headroom. Batching preserves the same greedy
decodes and computes each row's exact `KL(baseline || ablated)` before averaging; no candidate,
prompt, hook, threshold, or selection rule changed. The interrupted serial trace is retained in
`SELECT_ANSWER_CHANNEL_SERIAL.log`.
