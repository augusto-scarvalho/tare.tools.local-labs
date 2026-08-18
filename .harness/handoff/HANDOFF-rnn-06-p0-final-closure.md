# HANDOFF — RNN-06-P0 Final Provenance Closure

**Packet:** RNN-06-P0 — Frozen-Checkpoint BASE Regime Scout · **Closure micro-packet** (append-only).
**Date:** 2026-08-11 · **Author:** Claude (Opus 4.8) via Claude Code.
**Nature:** provenance/evidence reconciliation ONLY. No GPU rerun, no measured-result change, nothing pushed.

## HEAD / commit boundary
- **before-closure HEAD:** `2607c9921036efc3532e34866dd96737c1babc58`
- **closure commit (after HEAD):** `1b0cd2c297578c4a1f4a681c5fc2f0ddce66b170` — `closure(rnn): RNN-06-P0 append-only provenance reconciliation`
- **branch:** `master` · **upstream:** none · **pushed:** NO
- Prior P0 commits **intact, not rewritten:** `7d7feed` (pre-run protocol) → `46098e5` (results) → `2607c99` (handoff-fill) → **`1b0cd2c`** (this closure). No amend/rebase.

## Files changed by the closure commit (append-only, tracked)
- `runs/rnn/RNN-06-P0/AUDIT_RECONCILIATION.md` (new)
- `runs/rnn/RNN-06-P0/git_evidence_closure.txt` (new)
- `ops/rnn_06_p0_bundle_final.py` (new)

Untracked closure deliverables (per the `.harness/handoff/` convention, not committed): **this handoff** and `runs/rnn/RNN-06-P0/RNN-06-P0-final-audit-bundle.zip`.

## Final Git status (relevant)
Working tree clean except derived bundles: `runs/rnn/RNN-06-P0/RNN-06-P0-audit-bundle.zip` (first) and `…-final-audit-bundle.zip` (this) are untracked deliverables; unrelated pre-existing untracked helpers under RNN-04/05*/08* unchanged.

## Provenance resolution (full detail in AUDIT_RECONCILIATION.md)
- **`46098e5` exists** — full `46098e552caaca9fe9a71e2cc73e1d4765dd910e`, parent `7d7feed`, tree `fa023582296e225f7781f1201085b45883806879`; 14 files (+1246/−26).
- **Handoff self-reference = case A + B.** `46098e5` committed `HANDOFF.md` with the placeholder `<RESULTS_COMMIT>`; the **delivered** handoff (containing `46098e5`) is in the **later** commit `2607c99` and is clean at HEAD. Not uncommitted (≠ C). Confirmed via the first ZIP's manifest: delivered `HANDOFF.md` sha256 `82990be1…` == the `2607c99` blob.
- **Artifact bytes (Git blob SHA-1 · content SHA-256):**
  - `P0_PROTOCOL.md` `ab9b9ef7` · `3122af44…` (46098e5)
  - `P0_RESULTS_DELTANET.json` `54f1a161` · `5b454330…` (46098e5)
  - `P0_RESULTS_MAMBA2.json` `d35db764` · `97a6c945…` (46098e5)
  - `P0_CURVES.csv` committed-LF `e913b360` · `8edad565…` (1597 B); working-tree/first-ZIP-CRLF · `b98b2979…` (1610 B) — `.gitattributes eol=lf`, data-identical.
  - `P0_DECISION.md` `97362796` · `7869cf3e…` (46098e5)
  - `HANDOFF.md` `a07d29e9` · `82990be1…` (**2607c99**)
  - `ops/rnn_06_p0_mqar.py` `00eaeb39` · `a5023872…` (46098e5)
- Measured artifacts (`P0_RESULTS_*`, `P0_CURVES.csv`) are byte-identical between HEAD and `46098e5` (blob-equality verified) — the closure did not touch them.

## Executed-source identity status
**`PER_CANDIDATE_EXECUTED_SOURCE_IDENTITY = NOT_PROVEN`** (no runtime source hash / no reflog of on-disk bytes at execution instant). Best-candidate committed blobs (HIGH-confidence, corroborated by committed result-config fields, **not** cryptographically bound):
- **DeltaNet sweep** → blob `ae1c27d0` @ `7d7feed`, sha256 `fa90087a…` (24241 B). Corroboration: `P0_RESULTS_DELTANET.json` has `impl=<ABSENT>`, `autobatch_budget=<ABSENT>`, `config_overrides_applied={}` → pre-adaptation script.
- **Mamba-2 sweep** → blob `00eaeb39` @ `46098e5`, sha256 `a5023872…` (26456 B). Corroboration: `P0_RESULTS_MAMBA2.json` has `impl="transformers"`, `autobatch_budget=1536`, `config_overrides_applied={chunk_size:256→32}` → post-adaptation script.
- Remediation carried to 06A/06B: the runner must self-record its source SHA into the results JSON **before** any outcome-bearing run.

## Confirmations
- `NO_GPU_RERUN = TRUE`
- `NO_MEASURED_RESULT_MODIFIED = TRUE` (no `P0_RESULTS_*`/curve/threshold/classification change)
- `NO_HISTORICAL_COMMIT_REWRITTEN = TRUE`
- `NOTHING_PUSHED = TRUE`
- Accepted statuses unchanged: GDN `MODEL_NOT_RUNNABLE` (phenomenon NOT TESTED); DeltaNet `NOT_FOUND_WITHIN_BUDGET`; Mamba-2 `PLAUSIBLE` (exploratory); `FIXED_BACKBONE_GRADED_REGION = NOT_QUALIFIED`; `QWEN_GDN_TRANSPLANT_GATE = DEFER`; `GDN_COMPATIBILITY_GAP = OPEN`.

## Exactly one next recommendation (NOT executed)
**OPEN `RNN-06A-MAMBA` IN A NEW SESSION** — State Observability & Lifecycle Qualification on `AntonV/mamba2-1.3b-hf` (pin exact backend/kernels/source-SHA before outcome-bearing work). GDN compatibility remains a **separate** OPEN gap so it cannot delay the Mamba lifecycle experiment.

**STOP after closure. Do NOT start RNN-06A.**
