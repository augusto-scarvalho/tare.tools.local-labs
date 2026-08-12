# RNN-06C dir — note on the two `BLOCKED_BY_06B2` markers

This directory contains, from the PREVIOUS train (RNN-06 Fixed-Length State Load + Historical
Information, end HEAD `41ecfb78`), two markers:
- `BLOCKED.json` and `BLOCKED_BY_06B2.md` — committed by `42e449e`.

Those markers were CORRECT for that train: its Backlog-1 gate
(`FIXED_LENGTH_STATE_LOAD_REGION`) was BLOCKED, so RNN-06C was `BLOCKED_BY_06B2` and did not run.

They are **superseded** by the current train (RNN-06 Controlled State-Load Perturbation +
Historical Information). Here Backlog-1 RNN-06B3 minted
`STATE_LOAD_FORGETTING_PERTURBATION = QUALIFIED`, which OPENED the dependency gate, so RNN-06C
**executed** and produced:
- `HISTORICAL_INFO_RESULTS.json` + `HISTORICAL_INFO_DECISION.md` →
  **`HISTORICAL_STATE_INFORMATION = QUALIFIED`**.

The prior `BLOCKED*` files are left byte-unchanged (no history rewrite; they remain the prior
train's deliverable, already bundled in `RNN-06-state-load-historical-info-train-audit-bundle.zip`
SHA-256 `3d770241…`). The current train's audit bundle includes ONLY the executed 06C artifacts
plus this note, not the superseded `BLOCKED*` markers.
