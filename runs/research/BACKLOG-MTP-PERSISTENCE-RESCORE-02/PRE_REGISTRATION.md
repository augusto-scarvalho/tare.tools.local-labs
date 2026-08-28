# BACKLOG-MTP-PERSISTENCE-RESCORE-02 preregistration

Task: Correct the MTP persistence false negative against current cache telemetry
Evidence class: `mechanism_research`
Executor: Codex executor
Date: 2026-08-26

## Hypothesis

The 44 source observations are physical persistence successes and were marked
as failures only because the first runner required the stale
`timings.cache_n > 0` field. The current runtime's authoritative evidence is a
successful nonzero save/restore lifecycle plus `tokens_cached > 0`, exact
cold/restored content and the frozen oracle. If any source row lacks those
signals, or priming was not material in all 20 primed rows, this correction is
invalid.

## Frozen inputs

- Source samples: `runs/research/BACKLOG-MTP-PERSISTENCE-01/raw/samples.jsonl`, 2,681,187 bytes, SHA-256 `ae235a397007db32e6c5adc4228f26023fd0a7a5efa4a93cab8d7ba9e6b1b34d`.
- Source receipt: `runs/research/BACKLOG-MTP-PERSISTENCE-01/raw/receipt.json`, SHA-256 `6a52f8690b9f20c6583bdfa80e947ca4f7469e3297b24e046195a799c4116079`.
- Source result: `runs/research/BACKLOG-MTP-PERSISTENCE-01/RESULT.md`, SHA-256 `d49a9cffd05669d64d5d8bc0959250871f5c2179951243aa099a4ed52d160f02`.
- Source runner: `tools/research/run_mtp_persistence_first_instance.py`, SHA-256 `2a70897b1a9d73fb5fbf77159fa8c6706a41786c95bd428061e8963b8abde7b1`.

## Command

```powershell
python tools/research/run_mtp_persistence_rescore.py --outdir runs/research/BACKLOG-MTP-PERSISTENCE-RESCORE-02
```

## Factors

- Read-only rescore of exactly 44 immutable JSONL rows: four no-spec controls,
  20 unprimed MTP and 20 primed MTP observations.
- Physical success requires HTTP 200 lifecycle operations, positive `n_saved`
  and `n_restored`, positive top-level `tokens_cached`, exact cold/restored
  content and the `MAGNOLIA` oracle.
- Material priming requires positive `draft_n_accepted` before the erased
  persistence sequence.
- The source `pass` and `restored_cache_n` fields are analyzed as the treatment
  defect and never reused as corrected truth.
- No GPU requests or service changes are permitted.

## Acceptance gates

- `source_coverage`: `rescored_observations eq 44`
- `physical_persistence`: `physical_persistence_successes eq 44`
- `semantic_parity`: `exact_semantic_successes eq 44`
- `priming_materiality`: `material_primed_observations eq 20`
- `legacy_false_negative`: `legacy_false_negative_confirmed eq True`
- `historical_failure`: `unprimed_physical_failures eq 0`

## Abort conditions

- Any frozen source hash differs or the JSONL has other than 44 unique indices.
- Treatment counts differ from 4/20/20 or any required raw response field is
  absent.
- The original receipt does not contain the expected failed cache-dependent
  gates.
- The rescore attempts network, GPU or service mutation.
- Provenance or receipt construction is incomplete.

## Allowed claims

- `MTP_PERSISTENCE_HISTORICAL_FAILURE_NOT_REPRODUCED_R2`
- `MTP_PERSISTENCE_RESCORE_INVALID_R2`

Claims outside these codes are forbidden even if a metric looks favorable.
The executor stops at `EXECUTED`; independent review must decide whether to
supersede the original result text.
