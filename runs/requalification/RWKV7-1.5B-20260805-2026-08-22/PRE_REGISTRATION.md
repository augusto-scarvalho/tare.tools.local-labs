# RWKV7 1.5B official checkpoint — pre-registration

Frozen before GPU execution: 2026-08-22.

## Objective

Qualify an official, small, open-architecture recurrent checkpoint as an additional RTX 3090
option. This is a mechanism/runtime breadth candidate, not a quality comparison against the 27B
serving incumbent.

## Fixed artifact

- Hub: `RWKV/RWKV7-1.5B-20260805` (publisher card identifies it as the BlinkDL official release)
- Hub revision: `d2d414ff676d9d9c40a3d7b5c6faec7d2dd76e13`
- Local snapshot: `/home/augus/models/rwkv7-1.5b/official-d2d414f`
- Weights: BF16, 1,527,668,736 serialized parameters
- Runtime: release-bundled `inference/` code; `backend=auto`, state `float32`
- GPU: single RTX 3090; embedding service on 8081 remains untouched

The manifest-declared weights SHA-256 (`84ccbb...a14f`) must match the local file before a
qualified run. The source-owner weight license is unasserted by the card; therefore this experiment
may qualify local research behavior but must not label the weights Apache-2.0 or deployment-cleared.

## Gates

1. **Identity/admission:** pinned snapshot; local weights/config/tokenizer hashes match the bundled
   release manifest.
2. **Load/fit:** one-GPU BF16 load succeeds and peak/resident allocation leaves at least 4 GiB free.
3. **Runtime:** `auto` reports its resolved backend; generation returns non-empty finite output.
4. **Recurrent continuation:** suffix logits from full-sequence execution and two-part cached
   continuation agree with max absolute difference `<= 5e-2` in BF16/FP32-state execution.
5. **Constant state:** unique tensor storage bytes of returned recurrent state are equal after
   sequence lengths 32, 256, and 1,024 (zero growth tolerance).
6. **State isolation:** a fresh-state rerun of the same suffix agrees with the original fresh-state
   run under the same `5e-2` tolerance; accidental cross-prompt state reuse must be detectable.
7. **Bounded behavior:** four deterministic instruct-style prompts, maximum 128 new tokens each;
   report natural EOS rate and non-empty rate without promoting base-model quality.

## Decision rule

- `QUALIFIED_MECHANISM` only if gates 1–6 pass. Gate 7 is reported descriptively.
- `HOLD_RUNTIME` for runtime/dependency/kernel failure.
- `REJECT_MECHANISM` for reproducible parity, state-growth, or isolation failure.
- Weight-license status remains an independent `BLOCKED_FOR_DEPLOYMENT` until the owner asserts it.

After execution, restore machine mode to `SERVE`, restore `llm-inference.service`, verify ports 8080
and 8081, and restore the GPU power limit to 420 W.

