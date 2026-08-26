# BACKLOG-BEE-L5-LIVE-GUARD-03 preregistration

Task: Rerun BEE-L5 live guard with direct-file import bootstrap
Evidence class: `serving_runtime`

## Hypothesis

Adding the repository root to `sys.path` before importing the frozen R2 wrapper will permit the unchanged R2 physical protocol to execute. All nine R2 gates, panels, prompts, pathology definition, guard settings, and service invariants remain binding.

## Frozen inputs

- `runs/research/BACKLOG-BEE-L5-LIVE-GUARD-02/PRE_REGISTRATION.md`
- `runs/research/BACKLOG-BEE-L5-LIVE-GUARD-02/ABORTED.md`
- `tools/research/run_bee_l5_live_guard.py`
- `tools/research/run_bee_l5_live_guard_r2.py`
- `runs/research/BEE-L5-REASONING-LOOP-GUARD-2026-08-25/raw/receipt.json`
- `tools/analysis/reasoning_loop_guard.py`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/teacher_samples.json`

- R2 preregistration/abort/wrapper SHA-256: `b4731587185a2bbd6d2720b78bad7eeb951e01e181e0c571ddc915f163f975e2`, `57783ba9b1d080b104ce696082c46babcfde3f9dfae265f0f350405de74ec3ad`, `3a0d325f614708f2029fcc1f40c3c904f80a3d4f6f9c4320ec6cf664f122d118`.
- R1 core, historical receipt/guard, and teacher panel retain the hashes frozen in R2.

## Command

```powershell
python tools/research/run_bee_l5_live_guard_r3.py --outdir runs/research/BACKLOG-BEE-L5-LIVE-GUARD-03
```

## Factors

- Identical to R2: 128 real teacher traces and 25 paired live 128-token baseline/stream requests on the active four-slot `draft-mtp` route.
- The sole implementation delta is inserting the repository root in `sys.path` before importing R2. R2 supplies corrected `tokens[*].piece` parsing.

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

- Every R2 abort condition remains binding. No threshold, prompt, sample, guard, pathology definition, or runtime control may change after observation.

## Allowed claims

- `BEE_L5_LIVE_GUARD_QUALIFIED_R3`
- `BEE_L5_FALSE_POSITIVE_CONFIRMED_R3`

Claims outside these codes are forbidden even if a metric looks favorable.

No natural prevalence, server integration, answer preservation, deployment, or out-of-panel claim is allowed.
