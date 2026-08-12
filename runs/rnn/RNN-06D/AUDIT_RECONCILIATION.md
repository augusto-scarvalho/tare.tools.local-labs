# RNN-06D — AUDIT RECONCILIATION (append-only; historical artifacts NOT modified)

**Date:** 2026-08-12. **Provenance:** materialized from the RNN-06T packet Section 0 specification of
`AUDIT_RECONCILIATION_RNN-06D_2026-08-12.md` (that file was not present in the repo/workspace at
train start) **plus independent re-verification against the committed RNN-06D arrays**. The repository
is authoritative; this reconciliation is evidence appended alongside — it does **not** rewrite
`D0_DECISION.md`, `D1_DECISION.md`, `RECOVERY_CEILING_RESULTS.json`, or `RECOVERY_UTILITY_RESULTS.json`.

## Verified Git chain (repo truth)

RNN-06D commits `139045f..8f4b18a` (append-only; no amend/rebase): `d75369c` protocol+lib → `75e5940`
calibration+AMENDMENT 1 (v2)+freeze K=4 → `62ebee0` D0 prereg+identities → `eac2a1e` D0 ceiling runner
→ `d70f486` D0 results → `de1a012` D0 decision → `7e9a97e` D1 prereg → `8c7b21b` D1 runner → `f7f385b`
D1 results → `57f61d4` D1 decision → `8f4b18a` evidence/handoff/bundle. Subject
`AntonV/mamba2-1.3b-hf`@`703e19a4`, transformers-native naive bf16 cs=32.

## Preserved primary verdicts (unchanged)

- `RNN_06D0_RECOVERY_CEILING = QUALIFIED`
- `RNN_06D1_PREREGISTERED_GATE = QUALIFIED_PARAMETER_FREE`

## Appended auditor reconciliation verdicts

- `D0_ORIGINAL_PREREG_CONFORMANCE = DEVIATION`
- `D0_V2_CONFIRMATORY_QUALIFICATION = ACCEPTABLE_PROSPECTIVE_AFTER_AMENDMENT`
- `PARAMETER_FREE_HISTORICAL_RECOVERY_EXISTS = SUPPORTED_STRONGLY`
- `ADAPTIVE_SELECTOR_INCREMENTAL_ADVANTAGE = NOT_QUALIFIED`
- `END_TO_END_CAPTURE_PLUS_RECOVERY_UTILITY = NOT_QUALIFIED`
- `SINGLE_PASS_HISTORICAL_CAPTURE_PARITY = NOT_TESTED`
- `OFFICIAL_MAMBA_TRANSPORTABILITY = OPEN`

## Corrected wording on the construction amendment (binding)

The v1→v2 construction amendment (all-load → sentinel-pre/load-post) occurred **AFTER exploratory
calibration outcomes but BEFORE qualification outcomes**. It is **incorrect** to describe it as made
"before any outcome": exploratory calibration on the v1 construction had already produced outcomes
(ORACLE_PROXIMAL 0.15–0.40 << TAU_PROX 0.75) which motivated the amendment. This is why
`D0_ORIGINAL_PREREG_CONFORMANCE = DEVIATION` and the v2 qualification is scored as
`ACCEPTABLE_PROSPECTIVE_AFTER_AMENDMENT` (prospective on a fresh disjoint qualification set, thresholds
unchanged) rather than as strictly conformant to the original pre-registration.

> Correction note: earlier RNN-06D wording ("BEFORE any outcome" in the calibration commit message and
> lib comment, and "before any calibration or qualification outcome" in `TRAIN_PROTOCOL.md`) is
> superseded by the wording above for the purpose of conformance classification. The historical files
> are left byte-unchanged; this note is the authoritative reconciliation.

## Independent audit control — ALWAYS_SLOT_76 (re-derived from committed D0 arrays)

Computed here directly from `runs/rnn/RNN-06D/D0_READOUTS.npz` (schedule [38,76,115,153], N=192),
i.e. the non-adaptive control "always read out the fixed slot-76 snapshot":

| quantity | value (re-verified) | auditor value |
|---|---:|---:|
| ALWAYS_SLOT_76 accuracy | **0.770833** | ≈ 0.770833 |
| Δ vs FINAL (0.130208) | **+0.640625** | ≈ +0.640625 |
| recovered | **123** | 123 |
| harmed | **0** | 0 |
| MAX_CONFIDENCE (0.833333) − ALWAYS_SLOT_76 | **+0.062500** | ≈ +0.0625 |

**This is an AUDIT-DERIVED diagnostic, not a historical preregistered D1 arm.** D1 is not rewritten.

**Interpretation (the reason for the reconciliation).** The D1 headline (MAX_CONFIDENCE Δ = +0.703 vs
FINAL) is overwhelmingly attributable to *using a fixed early/middle historical snapshot at all*: a
trivial non-adaptive ALWAYS_SLOT_76 control already achieves Δ = +0.641 vs FINAL. The *adaptive*
confidence selector adds only **+0.0625** over that fixed control, and that adaptive-vs-fixed contrast
was **not a preregistered D1 comparison** (D1's frozen contrasts were method-vs-FINAL). Hence
`PARAMETER_FREE_HISTORICAL_RECOVERY_EXISTS = SUPPORTED_STRONGLY` (recovery from history is real and
large) but `ADAPTIVE_SELECTOR_INCREMENTAL_ADVANTAGE = NOT_QUALIFIED` (the adaptive component's
incremental value over a fixed slot is small and untested prospectively). RNN-06D also never measured
end-to-end capture economics (`END_TO_END_...= NOT_QUALIFIED`) and never captured true single-pass
in-run snapshots (`SINGLE_PASS_HISTORICAL_CAPTURE_PARITY = NOT_TESTED`; 06D snapshots were independent
re-prefills of prefixes). These four gaps define the RNN-06T train.
