# BACKLOG-SLX08-PHYSICAL-PREFILL-04 implementation blocker

Observed on 2026-08-29 before any experimental request or service mutation.

The clean `slop.cpp` worktree at commit `34b3dac7c` contains no callable
selected-block attention/prefill route, request control or telemetry field.
Feature-specific searches across `common/`, `src/`, `ggml/` and `tools/server/`
found DFlash speculative decoding and ordinary dense prefill kernels only.
DFlash injects draft-side KV state and cannot serve as the preregistered
selected-block attention treatment.

Consequently an OFF/ON campaign cannot currently satisfy:

- `physical_selected_block_prefill_requests >= 64`;
- `selected_block_route_observation_rate == 1.0`;
- `median_retained_attention_fraction == 0.5`.

Launching dense requests with an `ON` label would violate the preregistration
and create a false positive. No runner, GPU inference, service handoff or TTFT
measurement was started.

Unblock only after a separately reviewed `slop.cpp` implementation exposes an
explicit selected-block prefill control and per-request physical telemetry,
with fixture/build tests demonstrating that OFF is dense and ON actually
retains the requested block set. The immutable build and its tests must then be
bound before returning this packet to implementation.
