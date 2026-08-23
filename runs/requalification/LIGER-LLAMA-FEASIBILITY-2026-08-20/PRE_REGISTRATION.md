# LIGER-LLAMA-FEASIBILITY-2026-08-20 — pre-registration

**Status:** PRE-REGISTERED before Llama state-dict construction or tensor execution.

## Rationale and frozen substrate

The preceding Qwen3 extension stopped at its transfer gate because it adds
projection biases and drops Q/K-normalization weights. That result does not test
the paper's original Llama-3 path. This follow-up reuses the already qualified,
isolated substrate without changing versions:

- Liger: `0b364eb81d2159cc0fd9818b95d2d07d75522043`
- FLA: `72aa949f27dba47767f13226c45de29600d77312`
- lm-evaluation-harness: `1ba35e623b9bd9ca48df926f1a028043e159a6f2`
- Python 3.10.20, PyTorch 2.5.1, Triton 3.1.0, Transformers 4.52.4,
  FLA 0.3.0, FlashAttention 2.7.4.post1
- runner seed: `20260820`; RTX 3090, BF16

No pretrained checkpoint or benchmark data is needed. The reduced model has two
layers, hidden size 128, four attention heads, two KV heads, head dimension 32,
and a 256-token vocabulary.

## Known source-review risks

Before this pre-registration, static reading showed that the Llama attention path
uses bias-free Q/K/V projections, matching ordinary Llama config. It also showed
two cache risks whose outcomes have not been executed: recurrent cache offset is
updated from `q.shape[1]`, and the 64-token local-attention branch does not visibly
store K/V in the FLA cache. These are declared falsification targets, not post-hoc
explanations.

## Gates and metrics

1. **Transfer:** base Llama and Liger state dicts must have zero missing,
   unexpected, or shape-mismatched tensors under identical config.
2. **Construction:** after exact base-state transfer, a deterministic BF16 CUDA
   batch (`B=1`, `T=8`) must produce logits of shape `[1,8,256]`; loss, logits,
   and all trained gradients must be finite.
3. **Repeatability:** the construction microcase runs twice in fresh processes;
   both pass/fail vectors must match and scalar errors must agree within `1e-3`.
4. **Recurrence:** eval logits from a full `T=8` causal forward and eight cached
   single-token forwards must satisfy `max_abs <= 5e-2` and `max_rel <= 5e-2`.
   Cache sequence length after each token must equal `1..8` exactly.

The decision is lexicographic. Transfer failure stops before CUDA. Construction
failure stops before recurrence. Recurrence failure blocks checkpoint download,
quality evaluation, and fine-tuning even if full-sequence forward succeeds.

## Controls and interpretation

- Common base weights are copied exactly before any forward.
- Dropout is zero and model mode is explicit.
- Full and tokenwise paths receive identical token IDs and positions.
- GPU memory and server health are recorded; the existing Qwen38 server remains
  running unless the tiny model cannot allocate. A server stop is not authorized
  merely to rescue a failing kernel.
- Passing this campaign establishes only mechanical feasibility of the upstream
  Llama path. It does not establish pretrained quality or the paper's benchmark
  recovery claims.

## Reversal

No upstream file is modified. Artifacts are JSON receipts under this run
namespace. A cache or kernel fix requires a separate forked campaign with an
explicit patch digest. Nothing is pushed remotely.
