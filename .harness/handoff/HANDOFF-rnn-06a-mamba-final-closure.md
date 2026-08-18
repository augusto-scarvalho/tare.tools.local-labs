# HANDOFF — RNN-06A-MAMBA Final Audit Reconciliation / Closure

**Packet:** RNN-06A-MAMBA · **Closure micro-packet** (append-only interpretation reconciliation).
**Author:** Claude (Opus 4.8) via Claude Code. **Date:** 2026-08-11.
**Nature:** interpretation reconciliation ONLY. No GPU rerun, no measured-result change, no
threshold change, no edit to historical protocol/results/decision, nothing pushed.

## HEAD / commit boundary
- **before-closure HEAD:** `2b19d06824dbefd5b91f578a4fee614020eb1212`
- **closure commit (after HEAD):** `0560c3da12491b2ac3b5fb69213d565cf747c1cb` — `closure(rnn): RNN-06A-MAMBA append-only audit reconciliation`
- **branch:** `master` · **upstream:** none · **pushed:** NO
- Prior 06A commits **intact, not rewritten:** `2fd8b4e` (protocol) → `8a02147` (runner) →
  `7cc91f3` (results) → `ddef963` (decision) → `1ff0f56` (git evidence) → `2b19d068` (bundle
  tool) → **`0560c3da12491b2ac3b5fb69213d565cf747c1cb`** (this closure). No amend/rebase.

## Files added by the closure commit (append-only, tracked)
- `runs/rnn/RNN-06A-MAMBA/AUDIT_RECONCILIATION.md` (new)
- `runs/rnn/RNN-06A-MAMBA/git_evidence_closure.txt` (new)
- `ops/rnn_06a_mamba_bundle_final.py` (new — superset bundle builder)

Untracked closure deliverables (per `.harness/handoff/` convention): **this handoff** and
`runs/rnn/RNN-06A-MAMBA/RNN-06A-MAMBA-final-audit-bundle.zip`.

## Git status confirmation
Working tree clean except derived bundles (`RNN-06A-MAMBA-audit-bundle.zip`,
`…-final-audit-bundle.zip`) and pre-existing untracked helpers (RNN-04/05*/08*/P0). Historical
06A + P0 artifacts byte-unchanged: `LIFECYCLE_RESULTS.json` `d10527b6`, `PRE_REGISTRATION.md`
`50c8a41d`, `LIFECYCLE_DECISION.md` `ff2dd32d`, `LIFECYCLE_MATRIX.csv` `26693353`; P0
`P0_RESULTS_MAMBA2.json` `d35db764`. Verified unchanged (§below).

## Original protocol/results/decision unchanged
`NO_PREREGISTRATION_EDIT = TRUE`, `NO_RESULTS_EDIT = TRUE`, `NO_MATRIX_EDIT = TRUE`,
`NO_DECISION_ARTIFACT_REWRITTEN = TRUE`. The verdict `FROZEN_BACKBONE_LIFECYCLE = NOT_QUALIFIED`
is permanent and unchanged.

## Claim E reconciliation (the material correction)
The preregistered PRIMARY isolation test (alone-vs-in-batch, predeclared BIT_EXACT, load-bearing)
was **not met as BIT_EXACT**: `P_alone_vs_in_batch` / `Q_alone_vs_in_batch` = `BOUNDED_DIFFERENCE`
(max_abs 0.5, benign batched-GEMM). The runner substituted a non-preregistered neighbor-invariance
test (`[P,Q]` vs `[P,Q2]`), which passed **BIT_EXACT** (logits + per-row state) and drove `E_ok`.
- `E_ORIGINAL_ALONE_VS_BATCH_CRITERION = NOT_MET_AS_BIT_EXACT`
- `E_NEIGHBOR_INVARIANCE = PASS_BIT_EXACT`
- `E_PREREGISTERED_GATE_CONFORMANCE = DEVIATION`
- **"sole blocker = Claim B" is corrected/withdrawn.** Under the literal preregistered gate the
  verdict is over-determined by **both** Claim B and Claim E (alone-vs-batch). Verdict unchanged.

## Claim B historical status
`CLAIM_B_ORIGINAL_GATE = FAIL`. Preserved: a2 `NOT_EQUIVALENT` (max_abs 0.625 > preregistered 0.5);
a5/a8 `BOUNDED_DIFFERENCE` (0.375). Argmax 100% is secondary and does not make historical B pass.
No threshold relaxed. Root cause narrowed: `PREFILL_PADDING_OR_REDUCTION_ORDER_MECHANISM =
PLAUSIBLE / NOT_CAUSALLY_PROVEN` (pre-decode/pre-restore divergence is SUPPORTED; exact mechanism
not ablated).

## Recurrent-cache lifecycle vs autonomous-generation checkpoint
`RECURRENT_CACHE_CHECKPOINT_RESTORE_ON_SEGMENTED_PATH = BIT_EXACT` (conv+ssm, 48 layers). BUT
Claim C/D seeded the compared continuation with a prefix last-logit carried from OUTSIDE the
serialized recurrent snapshot → `COMPLETE_AUTOREGRESSIVE_GENERATION_CHECKPOINT = NOT_PROVEN_BY_06A`.
Deterministic known-token continuation from the recurrent tensors is unaffected.

## State-size economics flag
`state_bytes_per_sequence = 52,002,816` (≈49.59 MiB bf16; ssm dominates). Earlier O(1–10 MB)
"constant-size" estimates superseded for this candidate/backend. `HISTORICAL_SNAPSHOT_ECONOMICS =
REQUIRES_RECALCULATION_BEFORE_06C_06D`.

## Confirmations
- `NO_GPU_RERUN = TRUE` · `NO_MEASURED_RESULT_MODIFIED = TRUE` · `NO_THRESHOLD_CHANGED = TRUE`
- `NO_HISTORICAL_ARTIFACT_REWRITTEN = TRUE` · `NO_HISTORICAL_COMMIT_REWRITTEN = TRUE`
- `NOTHING_PUSHED = TRUE`
- Boundaries: `GDN_COMPATIBILITY_GAP = OPEN`, `QWEN_GDN_TRANSPLANT_GATE = DEFER`, no 06B/06A-EXT/
  06A2 started, no historical-state, no Memory Caching, no backend change, no Qwen, no serving change.
- `PACKET_AUDIT_FILE = NOT_PRESENT_IN_UPLOADS` (transcribed from request; P0-precedent-consistent).

## Exactly one next recommendation (NOT executed)
**OPEN `RNN-06A2-MAMBA-CONTINUATION` IN A NEW SESSION** — a NEW independently preregistered
experiment (operational continuation contract + criterion justified from RNN-06B/C BEFORE
outcomes; new deterministic held-out sequences; includes generation-frontier metadata for a
complete generation checkpoint). Do NOT repair GDN / run 06B / test historical state / change backend.

**STOP after closure. Do NOT start RNN-06A2 or RNN-06B.**
