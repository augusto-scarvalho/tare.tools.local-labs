# BACKLOG-BLIND-NUMERIC-RELABEL-02 preregistration

Task: Blind policy-consistent amendment of the complete numeric relabel
Evidence class: `human_calibration`

## Hypothesis

A fresh blind adjudicator applying one target-specific rule will resolve the five audited task families: six non-target responses become null and the mixed-number conclusion is preserved verbatim, without changing the other 761 labels.

## Frozen inputs

- `runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-01/raw/receipt.json`
- `runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-01/raw/final_blind_labels.jsonl`
- `runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-01/raw/sealed_mapping.json`
- `runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-01/raw/inter_rater_agreement.json`
- `runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-01/REVIEW.json`

All R1 labels, mapping, agreement, receipt and independent review are frozen by SHA-256 in `tools/research/run_blind_numeric_relabel_r2.py`. The five-row amendment bundle exposes only opaque ID, question and response.

## Command

```powershell
python tools/research/run_blind_numeric_relabel_r2.py --prepare
python tools/research/run_blind_numeric_relabel_r2.py --finalize
```

## Factors

One fresh blind adjudicator receives all seven arm rows belonging to the five task IDs named by the independent R1 audit when either retained response exhibits the same target/subquantity ambiguity. Frozen rule: return a value only when the response concludes the quantity requested by the question; a concluded subquantity is null. Preserve a mixed number as written except whitespace normalization; do not calculate an improper fraction. The remaining 761 R1 labels and the 96-record agreement result are immutable. No test-model inference, GPU, training or service mutation.

## Acceptance gates

- `source_integrity`: `r1_sources_verified eq True`
- `label_coverage`: `final_labels eq 768`
- `amendment_blinding`: `gold_or_arm_fields_exposed eq False`
- `amendment_coverage`: `amended_policy_cases eq 5`
- `inherited_agreement`: `inherited_exact_inter_rater_agreement ge 0.85`
- `target_policy`: `unresolved_target_policy_cases eq 0`
- `representation_policy`: `unregistered_numeric_recodings eq 0`
- `adjudication`: `unresolved_amendments eq 0`

## Abort conditions

Abort on source mismatch, gold/arm/source leakage, amendment task families other than the five frozen IDs or rows other than the seven frozen records, missing/duplicate label, non-target subtotal retained, mixed-number arithmetic recoding, mutation of any other R1 label, or incomplete provenance. Downstream scientific claims remain forbidden.

## Allowed claims

- `BLIND_NUMERIC_RELABEL_COMPLETED_R2`
- `BLIND_NUMERIC_RELABEL_NOT_VALIDATED_R2`

Claims outside these codes are forbidden even if a metric looks favorable.
