# BACKLOG-AGY-SYSTEM-BLOCKERS-03 preregistration

Task: Semantically classify six AGY integration blockers
Evidence class: `mechanism_research`

## Hypothesis

The predecessor's literal, path-restricted search produced at least one false
negative: the immutable candidate commit contains a CUDA GDN cache fusion that
materially elides a recurrent-state copy. A whole-tree semantic inspection will
classify all six claims without treating nearby mechanisms as full matches.

## Frozen inputs

- `runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-02/raw/receipt.json`
- `runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-02/raw/samples.json`
- `tools/research/run_agy_system_blockers_r2.py`

- Admission: `17b6d1d1cc7e8f2ca761f7077635555bf136da1270ab24a093e3997e17660d53`.
- R2 receipt: `06255331280cc995c17cbba2b4dc78443690ba1530c0243b81884ca1f3c04af0`.
- R2 samples: `33bdab6b182d9998ab0ce0634ad7632491e215be5dc27c966e80adafb14abc6f`.
- R2 runner: `77d6f704d23a6d60eedf5f3bccdc9f0c7d7b808d3ebf62515cbd9fe8c7767060`.
- Candidate WSL commit: `87a416bd75d5a64e66e55846b779c0a54eca21bd`.

## Command

```powershell
python tools/research/run_agy_system_blockers_r3.py --outdir runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-03
```

## Factors

- Six frozen predecessor claims: SLX-03, SLX-07, REP-04, REP-05, RETRO-01 and SLX-08.
- Read-only inspection of every tracked path at the exact candidate commit plus
  a bounded model inventory.
- Positive classifications require feature-specific source anchors; lexical
  near-matches are retained but cannot satisfy materialization.
- Service identity is normalized to PID, executable, argv, restart count and
  health, excluding timestamps and serialized systemd metadata.

## Acceptance gates

- `scope_coverage`: `classified_items eq 6`
- `semantic_evidence`: `items_with_semantic_evidence eq 6`
- `false_negative_detection`: `confirmed_predecessor_false_negatives ge 1`
- `runtime_integrity`: `runtime_unchanged eq True`

## Abort conditions

Abort on candidate commit drift, predecessor hash drift, source-command error,
missing evidence for any item, service mutation, or unhealthy 8080/8081
baseline. This is read-only and must not stop or restart either service.

## Allowed claims

- `AGY_SYSTEM_BLOCKER_FALSE_NEGATIVE_CONFIRMED_R3`
- `AGY_SYSTEM_BLOCKER_FALSE_NEGATIVE_NOT_CONFIRMED_R3`

Claims outside these codes are forbidden even if a metric looks favorable.
