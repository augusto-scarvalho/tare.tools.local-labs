# BACKLOG-GDN02-LEARNED-STATE-03 preregistration

Task: Requalify retained GDN state with literal WSL model identity
Evidence class: `mechanism_research`

## Hypothesis

Passing the frozen model directory as the literal POSIX string
`/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe` will remove the R2
pre-load abort while leaving its three-cell learned-state treatment, retained
vectors, 147 collateral cosines, independent scorer and scientific gates unchanged.

## Frozen inputs

- R2 aborted terminal: `4ab5f6a79bd1b6b73d895add25d0d1a5b282cc6e3bb0674230d77fbe50570e6b`
- R2 runner: `da6520358d880ac8b4fd94265face19e0324a28dd538051d7f5a16fc9c378dc9`
- R2 worker: `c89d21fa26c26197e6facac92cc13f175502e898c3eceffcb6313f839521f9e1`
- retained scorer: `a488bcd69196a3186a5ef2aeedb1427e80b409c77e8a9d753942c740f934f91e`
- GSM8K corpus: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`
- model shard: `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`

## Command

```powershell
python tools/research/run_gdn02_learned_state_r3.py --outdir runs/research/BACKLOG-GDN02-LEARNED-STATE-03
```

## Factors

- All R2 scientific factors, records, layers, formulas, evidence counts and hardware remain fixed.
- Sole correction: the Windows controller stores the WSL model path as `str`, never `pathlib.Path`.
- Harness and watcher own terminal completion; R2 abort remains immutable.

## Acceptance gates

- `learned_states`: `learned_gdn_layer_cells ge 3`
- `retained_cells`: `retained_decisive_layer_cells eq 3`
- `retained_collateral`: `retained_collateral_cosines eq 147`
- `independent_recompute`: `recomputed_metric_match_rate eq 1.0`
- `target_leakage`: `median_old_fact_leakage_pct le 5.0`
- `collateral_retention`: `median_collateral_retention_pct ge 90.0`
- `update_fidelity`: `median_updated_fact_fidelity_pct ge 95.0`
- `state_materiality`: `distinct_recurrent_state_conditions ge 3`

## Abort conditions

All R2 abort conditions remain. Additionally abort if argv contains backslashes,
the literal directory or model shard is absent, or the worker resolves a
different model identity.

## Allowed claims

- `GDN02_RETAINED_WSL_IDENTITY_QUALIFIED_R3`
- `GDN02_RETAINED_WSL_IDENTITY_REJECTED_R3`

Claims outside these codes are forbidden. Scope remains representation-level.
