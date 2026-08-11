# RNN-05B — Audit Reconciliation (interpretation / provenance / terminology, 2026-08-11)

**RNN-05B = ACCEPTED_WITH_AUDIT_CLARIFICATIONS.** The load-bearing numerical results stand and are accepted:
LA additive collapse; DN/GDN non-additive state; complete toy-module matrix+conv checkpoint/restore; frozen
post-hoc MC negative; trained frozen reader NO_EFFECT; strong train×inference interaction on GDN; same-weight
fixed-position cache-count sweep; weak directional historical-state signal in DN/GDN. This note **supersedes
interpretation only** — every RNN-05B raw artifact (`run.log`, `rnn05b_results.json`, `rnn05b_outcomes.json`,
`rnn05b_summary.csv`, `substrate_selftest.json`) is **immutable and unchanged**. Machine-readable form:
`rnn05b_audit_reconciliation.json`. Start HEAD `25c28f0`. **No training, no benchmark rerun, no Qwen, not
pushed.**

## 1. Memory-axis qualification defect
The executed calibration band was `0.30 < GDN_base < 0.96`, but the recorded `select_rule` string said
`(0.30,0.90)`. With `D=36 GDN=0.9407` / `D=40 GDN=0.9675`, D=36 satisfies 0.96 but **not** 0.90, and the final
3000-step GDN baselines were ~0.97–0.98 → severe ceiling limitation. Raw calibration preserved.
- `MEMORY_AXIS_ORIGINAL = QUALIFIED_BY_EXECUTED_0.96_RULE`
- `MEMORY_AXIS_INTENDED = NOT_QUALIFIED`
- `MEMORY_AXIS_FINAL_INTERPRETATION = CEILING_LIMITED / NOT_QUALIFIED_FOR_POSITIVE_GAIN_DETECTION`
**Code fix (future-proofing):** `ops/rnn_mc_05b.py` now derives the recorded rule string from the single-source
constants `CAL_BAND_LO/HI/TGT`, so recorded and executed bounds cannot diverge; `calibration_rule_selfcheck()`
asserts identity → **PASS** (`calibration_rule_selfcheck.json`). Raw JSON/logs untouched.

## 2. Co-adaptation wording
Retract "H1 is decisive" and "MC helps when train/inference are matched." Use:
- `TRAIN_INFERENCE_MC_INTERACTION = STRONG`
- `MC_COADAPTATION_DEPENDENCE = SUPPORTED`
- `NET_MC_BENEFIT_AFTER_COADAPTATION = NOT_DETECTED`
Both mismatched cells (B, C) degrade strongly; matched MC-training + MC-inference (D) recovers **approximately
the ordinary single/single baseline** (A). The experiment demonstrates **regime dependence, not a net MC
quality benefit.** GDN 3-seed interaction preserved: **0.3025 / 0.3432 / 0.2902**.

## 3. Reference-parity evidence class
Parity compared a **local** sequential-scan port vs a **local** chunk-parallel port; both derive from the
pinned Qwen recurrence; the FLA/upstream executable was **not** invoked (no FLA installed, per packet).
- `LOCAL_DUAL_IMPLEMENTATION_PARITY = PASS`
- `UPSTREAM_EXECUTABLE_PARITY = NOT_QUALIFIED`
- `REFERENCE_PARITY_SCOPE = LOCAL_PORTS_ONLY` (now emitted by `reference_parity()`). Numerical parity unchanged.

## 4. Qwen mapping wording
Replace "maps 1:1 to Qwen cache" with **`STRUCTURALLY_ANALOGOUS_TO_QWEN_TWO_PART_CACHE`**. The toy complete
state is `recurrent matrix + raw conv boundary buffer (kernel-1)`; RNN-01 observed the real Qwen cache as
`recurrent_states + conv_states`, but exact representation parity is unproven.
- `QWEN_CACHE_ROLE_MAPPING = SUPPORTED` · `QWEN_CACHE_REPRESENTATION_PARITY = NOT_PROVEN`.

## 5. Historical snapshot terminology
Distinguish `FULL_RESTORABLE_SEQUENCE_CHECKPOINT` (matrix **+** conv boundary — the live restorable complete
state) from `HISTORICAL_RECURRENT_STATE_SNAPSHOT` (matrix-only — what the MC historical cache stores per
segment). Matrix-only historical entries are **not** full checkpoints. No numerical result changes.

## 6. Isolation / branch evidence scope
- `STATELESS_REQUEST_EXECUTION = PASS` — forward(B)/forward(A)/forward(B) proves no model-global persistent
  state (weaker than a request-owned runtime cache lifecycle).
- `CHECKPOINT_CONTINUATION = PASS`.
- `SINGLE_BLOB_FORK_BRANCHING = PASS` — the original branch helper recomputed the prefix separately per
  branch (was `NOT_DIRECTLY_TESTED`). A tiny CPU test (`single_blob_fork`, seconds, no training) now
  serializes **one** `{S, conv}` blob and restores it **twice** with two different suffixes; each branch
  matches its own full run (err ≤ ~1e-6) and the branches diverge (~3.09). Results in `single_blob_fork.json`.

## 7. Pure-cache interpretation
`PURE_CACHE_COUNT_CURVE = QUALIFIED` (weights, positions, segmentation fixed). Retained-K curves
(K=1,2,4,8): LA `0.997→0.997→0.989→0.976` (**degrades**); DN `0.971→0.972→0.974→0.975` (**small monotonic
improvement**); GDN `0.981→0.982→0.985→0.984` (**improves K1→K4, slight decrease at K8**). →
`DN_GDN_HISTORICAL_COMPLEMENTARITY = WEAK_DIRECTIONAL_SIGNAL`. **H3 is NOT labeled positive.**

## 8. Raw run vs final derived provenance
`run.log` ends with the first-pass classifier (`*_MC_SIGNAL = NEGATIVE`, gate `DEFER`); the derived JSON uses
the refined classifier (`NO_EFFECT_NAIVE_MC_NEGATIVE`, gate `CONDITIONAL / DEFER`). Metrics did **not** change;
only classification semantics did.
- `RAW_RUN_CLASSIFICATION = HISTORICAL_FIRST_PASS` · `FINAL_DERIVED_CLASSIFICATION = SUPERSEDING_POST_ANALYSIS`.
- **For the current architectural decision:** `QWEN_GDN_TRANSPLANT_GATE = DEFER` until the ceiling-limited H3
  test is resolved. `run.log` is not modified.

## 9. GDN collapsibility wording
Do not generalize "GDN final state ≈ recent state" — that varied by probe/config. Preserve only
`GDN_ADDITIVE_COLLAPSE = NO` plus the measured per-artifact distances: harness (d_k=64) GDN rel-err vs final
`sum 1.086 / mean 0.668 / last 0.561 / weighted-lstsq 0.261`; the d_k=24 substrate self-test had `last≈0`
(config-specific). DN (d_k=64) `sum 0.626 / weighted-lstsq 0.384`. LA `sum ~1.8e-7 / weighted ~1.9e-7` (YES).

## Net
No numerical result changed. RNN-05B is **ACCEPTED_WITH_AUDIT_CLARIFICATIONS**; the frozen-backbone MC negative
and the strong (but net-benefit-absent) train×inference interaction stand; `QWEN_GDN_TRANSPLANT_GATE = DEFER`
pending the ceiling-limited H3 test (RNN-05B-EXT). Do not auto-start RNN-05B-EXT.
