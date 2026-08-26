# BACKLOG-AGY-SYSTEM-BLOCKERS-02 preregistration

Task: Repeat six-item integration blocker audit with direct WSL argv
Evidence class: `proxy_realization`

## Hypothesis

Direct `wsl -e` argv transport will preserve every frozen R1 regex literally and allow the unchanged six-item integration audit to finish. All four R1 gates and claim limits remain binding.

## Frozen inputs

- `runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-01/PRE_REGISTRATION.md`
- `runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-01/ABORTED.md`
- `tools/research/run_agy_system_blockers.py`

- R1 preregistration/abort/implementation: `a6ce6a423f56d9b3328943fd984bd1df84b735667dc1af747da858f7248355ef`, `549a3a9ab7e6e335242577cd9cdf37feb51ad5076ed00939c84d3318b20c67ae`, `064b113ff68b935e576ee75ac64033cef66d61b1e5ad94d87642231dc1f53b2e`.
- All predecessor receipt/probe identities remain exactly those frozen in R1.

## Command

```powershell
python tools/research/run_agy_system_blockers_r2.py --outdir runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-02
```

## Factors

- Identical six items, patterns, deployed binary, candidate source/model inventory, unlock criteria and read-only controls as R1.
- Sole delta: every direct Linux command uses `wsl -d Ubuntu-24.04 -e <argv>` so regex metacharacters reach `git grep` literally.

## Acceptance gates

- `scope_coverage`: `audited_items eq 6`
- `objective_absence`: `missing_integration_count eq 6`
- `probe_classification`: `proxy_only_predecessors eq 6`
- `runtime_integrity`: `runtime_unchanged eq 1`

## Abort conditions

- All R1 abort conditions remain binding; any search transport failure or ambiguous positive match aborts.

## Allowed claims

- `AGY_SYSTEM_BLOCKERS_REGISTERED_R2`

Claims outside these codes are forbidden even if a metric looks favorable.

This is a current integration blocker audit, not a performance experiment or scientific rejection.
