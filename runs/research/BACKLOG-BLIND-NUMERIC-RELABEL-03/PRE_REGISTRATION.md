# BACKLOG-BLIND-NUMERIC-RELABEL-03 preregistration

Task: Two-rater blind resolution of the final numeric-label policy conflict
Evidence class: `human_calibration`

## Hypothesis

Two fresh blind raters applying the audited target-specific policy will independently identify full-trace gsm8k/865 as the explicit target value 75 and full-trace gsm8k/774 as null because it concludes driving time rather than total time away.

## Frozen inputs

- `runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-02/raw/receipt.json`
- `runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-02/raw/final_blind_labels.jsonl`
- `runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-01/raw/sealed_mapping.json`
- `runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-02/REVIEW.json`

R2 receipt/labels/review and the sealed R1 mapping are frozen by SHA-256 in `tools/research/run_blind_numeric_relabel_r3.py`. Each rater input exposes only opaque ID, question and response.

## Command

```powershell
python tools/research/run_blind_numeric_relabel_r3.py --prepare
python tools/research/run_blind_numeric_relabel_r3.py --finalize
```

## Factors

Exactly two frozen records are independently labeled by two new blind raters. Frozen rule: an explicit assertion of the requested quantity is the label even if it appears before a later wrong total; a value that covers only a different duration component is null. Any disagreement requires a third blind adjudication. The other 766 labels are byte-preserved. No test-model inference, GPU, training or service mutation.

## Acceptance gates

- `source_integrity`: `r2_sources_verified eq True`
- `label_coverage`: `final_labels eq 768`
- `blinding`: `gold_or_arm_fields_exposed eq False`
- `double_coverage`: `double_labeled_records eq 2`
- `rater_independence`: `independent_raters ge 2`
- `policy_resolution`: `unresolved_policy_records eq 0`
- `preservation`: `other_label_mutations eq 0`

## Abort conditions

Abort on source mismatch, field leakage, fewer than two raters, missing/duplicate IDs, unresolved disagreement, labels other than 75 for the explicit yellow-car assertion or null for travel-only duration, mutation of another label, or incomplete provenance.

## Allowed claims

- `BLIND_NUMERIC_RELABEL_COMPLETED_R3`
- `BLIND_NUMERIC_RELABEL_NOT_VALIDATED_R3`

Claims outside these codes are forbidden even if a metric looks favorable.
