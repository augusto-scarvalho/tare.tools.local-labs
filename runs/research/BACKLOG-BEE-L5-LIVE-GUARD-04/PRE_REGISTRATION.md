# BACKLOG-BEE-L5-LIVE-GUARD-04 preregistration

Task: Rerun BEE-L5 live guard with byte-piece normalization
Evidence class: `serving_runtime`

## Hypothesis

Deterministically decoding integer byte-array pieces with `bytes(piece).decode("utf-8", errors="replace")` will permit the otherwise unchanged R3 protocol to finish. All nine gates and all scientific factors remain unchanged.

## Frozen inputs

- `runs/research/BACKLOG-BEE-L5-LIVE-GUARD-03/PRE_REGISTRATION.md`
- `runs/research/BACKLOG-BEE-L5-LIVE-GUARD-03/ABORTED.md`
- `tools/research/run_bee_l5_live_guard.py`
- `tools/research/run_bee_l5_live_guard_r3.py`
- `runs/research/BEE-L5-REASONING-LOOP-GUARD-2026-08-25/raw/receipt.json`
- `tools/analysis/reasoning_loop_guard.py`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/teacher_samples.json`

- R3 preregistration/abort/launcher SHA-256: `89e7fb7ad4e521426f48c2dc8965fc692c21e09cfe3bf8104da6fe435b10055b`, `7d006d848d240ce27454381a35a8fd2559532062fbff49b916325e83664658b2`, `f32ab039a1e1b39f7c70b0d7ac87cc9b9f4ec6ef141848a3c59686142530cdea`.
- Core, historical receipt/guard, and teacher hashes remain those frozen by R3.

## Command

```powershell
python tools/research/run_bee_l5_live_guard_r4.py --outdir runs/research/BACKLOG-BEE-L5-LIVE-GUARD-04
```

## Factors

- Identical 128 real teacher traces, 25 live baseline/stream pairs, 128-token budget, prompts, independent pathology definition, historical guard, and four-slot `draft-mtp` route.
- Sole delta: string pieces pass unchanged; integer arrays are decoded as bytes with UTF-8 replacement.

## Acceptance gates

- `legitimate_coverage`: `real_legitimate_traces eq 128`
- `pathology_coverage`: `live_pathological_baselines eq 25`
- `sensitivity`: `sensitivity_tpr ge 0.95`
- `specificity`: `false_alarm_fpr le 0.02`
- `physical_intervention`: `stream_aborts_confirmed eq 25`
- `token_savings`: `median_token_savings ge 0.8`
- `guard_overhead`: `guard_p95_us_per_token le 2.0`
- `service_integrity`: `service_restarts eq 0`
- `idle_recovery`: `idle_slots_after eq 4`

## Abort conditions

- All R3 abort conditions remain binding; any unexpected piece type aborts. No scientific factor may change after observation.

## Allowed claims

- `BEE_L5_LIVE_GUARD_QUALIFIED_R4`
- `BEE_L5_FALSE_POSITIVE_CONFIRMED_R4`

Claims outside these codes are forbidden even if a metric looks favorable.

No natural prevalence, server integration, answer preservation, deployment, or out-of-panel claim is allowed.
