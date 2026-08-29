# BACKLOG-BLIND-NUMERIC-RELABEL-01 preregistration

Task: Blind complete semantic relabel of retained trace and Q8 numeric outputs
Evidence class: `human_calibration`

## Hypothesis

Three independent semantic raters, blinded to arm and gold, can extract the concluded numeric value for all 768 retained responses with at least 85% exact agreement on a 96-record overlap and complete adjudication of disagreements.

## Frozen inputs

- `runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/receipt.json`
- `runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/student_samples.json`
- `runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02/raw/receipt.json`
- `runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02/raw/samples.jsonl`
- `runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-RESCORE-03/REVIEW.json`
- `runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-03/REVIEW.json`
- `workloads/gsm8k.jsonl`

All listed experiment sources and both R3 reviews are frozen by SHA-256 in `tools/research/run_blind_numeric_relabel.py`. Blind inputs expose only opaque record ID, question and response. The sealed mapping is not an annotator input.

## Command

```powershell
python tools/research/run_blind_numeric_relabel.py --prepare
python tools/research/run_blind_numeric_relabel.py --wait-finalize
```

## Factors

Exactly 768 shuffled records: 512 trace-deployment and 256 Q8-utility outputs. Three raters each receive 256 primary records plus 32 overlap records in batches of at most 32. Shuffle/opaque-ID seed is 2026082813. Labels contain only concluded numeric value or null, confidence and a short gold-blind rationale. Every disagreement is sent to a fourth blind adjudicator. No model inference under test, training, GPU or service mutation occurs.

## Acceptance gates

- `source_integrity`: `frozen_sources_verified eq True`
- `bundle_coverage`: `blind_records eq 768`
- `gold_blinding`: `gold_or_arm_fields_exposed eq False`
- `primary_coverage`: `primary_labels eq 768`
- `overlap_coverage`: `double_labeled_records ge 96`
- `rater_independence`: `independent_raters ge 3`
- `agreement`: `exact_inter_rater_agreement ge 0.85`
- `adjudication`: `unresolved_disagreements eq 0`

## Abort conditions

Abort on source hash mismatch, leaked gold/arm/original-label fields, missing or duplicate record IDs, malformed numeric labels, fewer than 768 primary labels, fewer than 96 double labels, fewer than three distinct raters, agreement below 0.85, unresolved disagreement, or incomplete provenance. Do not infer human annotation or either downstream scientific claim.

## Allowed claims

- `BLIND_NUMERIC_RELABEL_COMPLETED_R1`
- `BLIND_NUMERIC_RELABEL_NOT_VALIDATED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.
