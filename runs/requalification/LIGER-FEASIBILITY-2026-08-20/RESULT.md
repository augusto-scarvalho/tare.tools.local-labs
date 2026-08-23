# LIGER-FEASIBILITY-2026-08-20 — result

**Decision:** BLOCKED at the pre-registered static-compatibility gate. No model
weights were downloaded, and the construction, forward/backward, recurrence,
or fine-tuning gates were not run.

## What passed

- The Liger source, FLA gitlink, and lm-evaluation-harness gitlink were recovered
  at their exact declared SHAs and all three worktrees were clean.
- The absent upstream `.gitmodules` file required a documented manual mapping to
  the two repositories named by the README. The commits themselves were reachable.
- An isolated Python 3.10.20 environment imported the pinned stack, including
  PyTorch 2.5.1, Triton 3.1.0, Transformers 4.52.4, FLA 0.3.0, and
  FlashAttention 2.7.4.post1. The complete resolved environment is frozen beside
  this report.

## Blocking result

A two-layer reduced Qwen3 base and the upstream Qwen3 Liger candidate used the
same vocabulary, hidden size, head geometry, MLP size, and attention-bias config.
The candidate nevertheless had six tensors absent from the base:

- `q_proj.bias`, `k_proj.bias`, and `v_proj.bias` in each of two layers.

The base had four tensors absent from the candidate:

- `q_norm.weight` and `k_norm.weight` in each of two layers.

There were no shape mismatches among common tensors. A real
`load_state_dict(strict=False)` confirmed the same six missing and four
unexpected keys. At Qwen3-8B scale this pattern would repeat across every layer:
the upstream implementation hard-codes Q/K/V biases even when
`attention_bias=False`, while replacing Qwen3 attention also discards its learned
Q/K normalization weights.

This violates the declared gate of zero unexplained state-transfer differences:
the conversion would silently leave new projection biases initialized outside
the base checkpoint and drop trained base tensors. Later tensor success could
not compensate for that loss of attribution, so the runner stopped fail-closed.

## Interpretation

This result applies to the late Qwen3 extension at upstream commit
`0b364eb81d2159cc0fd9818b95d2d07d75522043`; it does not refute the paper's
original Llama-3-based Liger mechanism. The next bounded experiment should test
the original Llama path under the same transfer gate. If that passes, recurrence
parity must still be measured before any checkpoint download or fine-tuning.

A Qwen3 compatibility patch would be a new forked mechanism, not an upstream
replication. It would need to remove the unintended biases, explicitly decide how
to preserve or retire Q/K norms, and separately qualify cache semantics.
