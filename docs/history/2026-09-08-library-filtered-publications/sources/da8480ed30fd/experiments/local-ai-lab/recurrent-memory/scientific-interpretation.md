# Recurrent Memory — scientific interpretation

[← Empirical history](empirical-history.md) · [Research-line index](README.md) · [Navigation](../../../NAVIGATION.md)

The post-06C result reopens **historical recurrent state as a memory substrate**, not the old synthetic Memory Caching recipe.

## Branch taxonomy

A. recover old states — snapshot retrieval / DART-like mechanisms;
B. avoid losing access — state expansion/alternative update mechanisms;
C. increase memory capacity — sparse/structured memory;
D. improve write/erase policy — newer gated/delta mechanisms.

Recent work such as DART, StateX, Sparse Delta Memory, Gated DeltaNet-2 and other recurrent-memory systems belongs to RESEARCH/WATCH. Their mechanisms are not substitutes for our measured causal chain.

## Permanent lab rules

- recurrent state ownership is model semantics;
- Cache API ≠ checkpoint-semantics proof;
- full state includes recurrent/conv/frontier metadata as required by model;
- snapshot bytes at wrong temporal boundary can be corrupted semantically while deserializing correctly;
- feature enabled ≠ mechanism exercised;
- quality benchmark ≠ lifecycle proof;
- execution identity includes model/runtime/kernel/dtype/chunk semantics.

Serving-runtime issues around recurrent state, prefix hits, replay/copy semantics are relevant engineering evidence but do not override the experiment.

---

**Next gate:** [RNN-06D →](backlog-rnn-06d.md) · **Related learning research:** [Adaptive Learning](../../../research/context/adaptive-learning-cross-project-evolution.md)
