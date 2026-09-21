# LitJEV on resident Qwen: bounded feasibility diagnostic

Frozen before inference. User-requested evaluation, no production promotion.
Upstream revision: e7fb109a7466da9709028eb9c4e9f16eaeb4e2a3.
Use upstream question rendering and restricted candidate normalization with the
existing qualified qwen38 GGUF via gateway /completion. This is a backend port
of the single-question fast method, not upstream Transformers/H100 replication.
The native API may require one sampled token to expose first-position logprobs;
record that token honestly, never call that zero-token inference. No reasoning.

First one transport control (choose literal beta among alpha/beta/gamma), then
freeze/run the existing six-case OpenJEV/Laya protocol unchanged plus six newly
authored synthetic cases (three scenarios, two orders): unknown prices require
abstention; unmatched cohorts require abstention; equal price/quality favors
lower complete-delivery time. Gold labels excluded from inference.
No prompt tuning or retries based on answers. Missing candidate logprobs fails
that case; never assign an invented zero. At most one capability control and
12 labelled inference calls, 120 s/request, 600 s total, no automatic retry.
The top-128 pre-sampling logprobs must contain every candidate; normalize in
log space. Check full suffix token boundary and input length <=4096 tokens.
No grammar, logit bias, answer forcing or post-sampling probabilities.

All generation through gateway port8080 and its existing exclusive GPU lease.
Port18080 only tokenization/template inspection (no inference). Refuse a busy
image queue/lease, recheck between cases, preserve user jobs. No service restart.
Record health/process identity, placement memory at 200ms intervals, latency,
raw responses, exact prompts/token IDs, configuration and upstream file hashes.
If initial text residency is empty, unload our qwen through normal image-lease
supervisor on completion; preserve initially resident qwen. Never stop other
models/jobs. Embedding service is observed only and remains unchanged.

Pass: complete valid distributions, no truncation, 6/6 original cases and 6/6
new cases, both orders stable. Even a pass is synthetic diagnostic evidence,
not calibration, real task savings, multi-question branch validation, or routing
authority. No new model weights, global environment or TUI defaults.
