# LIGER-LLAMA-FEASIBILITY-2026-08-20 — result

**Decision:** BLOCKED at construction. The original Llama path transfers all
base weights exactly, but cannot execute under the repository's Qwen-era
Transformers 4.52.4 environment. Recurrence was not run.

## Repeated result

Two fresh processes using runner SHA-256
`725765d9d566dcf4ce69a4a148d1d19fe92a7ef03a38034ef1e1ca7b998d15ae`
produced the same gate vector and error:

- provenance: PASS for all three exact repository SHAs;
- state transfer: PASS, 361,088 parameters on both sides, zero missing,
  unexpected, or shape-mismatched tensors;
- construction: FAIL with `ValueError: too many values to unpack (expected 2)`;
- recurrence: not run fail-closed.

A third diagnostic run changed only the runner's exception receipt to preserve a
traceback. It reproduced the same result. The traceback locates the failure in
Transformers 4.52.4 `LlamaDecoderLayer.forward`: that API expects attention to
return two values, while upstream `LigerGatedLinearAttention.forward` returns
three (`output`, `weights`, `past_key_value`).

The FLA call also emitted the same warning in every process: the input looks
head-first while the pinned kernel defaults to sequence-first. That warning is a
separate unresolved risk; the outer Transformers unpack failure occurs before a
valid construction result can qualify its numerical effect.

## Source-history diagnosis

The repository initially pinned Transformers 4.47.1. Commit
`a17bb0650a8dbe9830ecd2a1529bf8d1776c4774` removed that pin and says 4.52.4 was
tested for the Qwen2.5 update, but it changed only imports/decorators in the Llama
file and did not update the three-value attention return contract. Thus the
repository's current unpinned environment is not a single qualified substrate
for all advertised architecture paths.

The next bounded test should use the original declared Transformers 4.47.1 stack
in a new environment/campaign. Passing there would establish only that the
historical Llama path constructs; cached recurrence still has its own declared
falsification targets.
