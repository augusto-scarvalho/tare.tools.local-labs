# BACKLOG-GDN02-LEARNED-STATE-02 preregistration

Task: Requalify learned GDN state editing with retained decisive tensors
Evidence class: `mechanism_research`

## Hypothesis

Repeating the frozen three learned Qwen3.5 GatedDeltaNet cells will reproduce
the R1 negative indication, and retaining all decisive target vectors plus all
49 collateral cosine values per layer will permit exact independent
recomputation of every gate.

## Frozen inputs

- `runs/research/BACKLOG-GDN02-LEARNED-STATE-01/raw/receipt.json`: `3222aceaa925b48fda0b9eb684e32f5dac917e4e86a80c6fc65279cddbb7f236`
- `docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md`: `a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04`
- `tools/research/run_gdn02_learned_state.py`: `c6f37fccf63cff34d14a4b76595acd9bf50835ec2b3400f18c4742611d4e0fcb`
- `tools/research/gdn02_learned_state_worker.py`: `9d09cf2b93c870f14dadf20662a7f8298b14e60bedf91e2ca78ccca17d5970cf`
- `workloads/gsm8k.jsonl`: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`
- Qwen3.5 model shard: `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`

## Command

```powershell
python tools/research/run_gdn02_learned_state_r2.py --outdir runs/research/BACKLOG-GDN02-LEARNED-STATE-02
```

## Factors

- Same 50 records, target index 5, old/new values 41/42 and learned layers 0/1/2 as R1.
- Baseline, append-only correction and replace-in-place oracle remain token-count matched.
- Retain target baseline/correction/oracle output vectors for every layer in a
  safetensors bundle and retain 49 individual collateral cosines per layer.
- A separate scorer reopens the bundle and recomputes three cells, 147
  collateral cosines and aggregate metrics before sealing.
- RTX 3090 inference; fixed model/corpus; no serving process mutation.

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

Abort on input/model mismatch, missing learned GDN modules, token-length
mismatch, fewer than 50 records, state/output capture failure, missing or
nonfinite retained tensor/cosine, wrong tensor count/shape, scorer disagreement,
OOM, service identity change or harness failure.

## Allowed claims

- `GDN02_LEARNED_STATE_WITH_RETAINED_TENSORS_QUALIFIED_R2`
- `GDN02_LEARNED_STATE_WITH_RETAINED_TENSORS_REJECTED_R2`

Claims outside these codes are forbidden even if a metric looks favorable.
This remains representation-level learned-state evidence, not full-model
knowledge editing, serving integration or permanent erasure.
