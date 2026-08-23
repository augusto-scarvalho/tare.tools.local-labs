# RNN-09 — RetNet retention mechanism qualification

Scope: qualify the RetNet retention identity already recorded in the research ledger:
parallel decayed retention versus recurrent constant-state retention. This is a mechanism
microbenchmark, not an official-checkpoint reproduction and not a model-quality claim.

Frozen before results:

- CPU PyTorch, seed 20260821, float64 primary parity.
- Shapes `B=2`, `d_k=d_v=16`; sequence lengths 1, 7, 64, 513; decay 0.97.
- Gates: parallel↔recurrent max absolute error ≤1e-10; chunkwise recurrence ≤1e-10 for
  chunks 1, 3, 16, 128; save/reload bit-exact; batch isolation bit-exact; leaked state
  must be detectable and explicit reset must recover standalone output; q/k/v gradient
  parity ≤1e-9; float32 recurrent state finite at 4096 tokens.
- Timing is descriptive only (single CPU thread, medians) because the Python recurrent
  loop is not a production kernel. The complexity/state-size comparison is the mechanism
  result; timing cannot promote an engine.
- Any gate failure blocks a reproduction claim and remains recorded; tolerances are not
  loosened after seeing output.

