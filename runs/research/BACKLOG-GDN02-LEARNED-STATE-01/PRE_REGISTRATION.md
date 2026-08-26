# BACKLOG-GDN02-LEARNED-STATE-01 preregistration

Task: Requalify GDN-02 on learned Qwen3.5 GatedDeltaNet states
Evidence class: `mechanism_research`

## Hypothesis

Across learned Qwen3.5 GatedDeltaNet layers 0, 1 and 2, append-only correction of one record among 50 will either reproduce the historical rejection or reverse it. Qualification requires median old-fact leakage at most 5%, collateral retention at least 90%, and update fidelity at least 95% against a replace-in-place oracle. Failure of any gate retains the negative on this frozen representation-level screen.

## Frozen inputs

- `runs/research/GDN-02-ERASE-RETENTION-2026-08-25/PRE_REGISTRATION.md`
- `runs/research/GDN-02-ERASE-RETENTION-2026-08-25/RESULT.md`
- `runs/research/GDN-02-ERASE-RETENTION-2026-08-25/raw/receipt.json`
- `tools/probes/gdn02_erase_retention_lab.py`
- `runs/research/BACKLOG-NEGATIVE-KV-REAL-SCREEN-02/raw/model_hash.json`
- `workloads/gsm8k.jsonl`

- Historical preregistration/result/receipt/probe SHA-256: `9b72e8acb64f365a3d2bbed82eca1381e3640635a26073d5cd6c274762c1cb4b`, `677560ec0bd1e191cc7c187a594fa320f236b0ae501646563183695bf197c766`, `0494fc573181c9dfa6d16371f38faf72461da8942f8c8659f8a40ad8d7266e33`, `e34cdcfdcb6f9df13bc87aeb67c549922c7015eb24c3268ca2d13f6fdc786ae7`.
- Model-hash ledger SHA-256: `45f10080c70897cb106b21013bc4953f6a5696a27296098972d60ca132fad1ec`; safetensors shard SHA-256: `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`.
- Frozen GSM8K workload SHA-256: `68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77`.

## Command

```powershell
python tools/research/run_gdn02_learned_state.py --outdir runs/research/BACKLOG-GDN02-LEARNED-STATE-01
```

## Factors

- Take the first 50 unique frozen GSM8K task IDs as record keys. Record values are their frozen numeric answers except target index 5, whose old/new values are fixed to `41`/`42`.
- Conditions use token-count-matched templates: baseline contains old target plus a latest-record reaffirmation of `41`; treatment contains old target plus latest-record correction to `42`; oracle replaces the original target with `42` and reaffirms `42`.
- Query all 50 keys under baseline and treatment. Query the target under oracle. Tokenized sequence lengths for corresponding conditions must match exactly.
- Evaluate the actual learned `Qwen3_5GatedDeltaNet` modules at layers 0, 1 and 2 using the checkpoint's embeddings, layer-specific input RMS norm, convolution, learned Q/K/V projections, learned beta/decay gates and official chunk recurrent kernel.
- Force the official recurrent kernel to return its final physical recurrent state only for hashing/materiality; do not alter outputs or learned parameters.
- For each layer: `B` is target-query baseline vector, `C` correction vector and `O` replace-in-place oracle vector. Leakage is `100*d(C,O)/(d(C,O)+d(C,B))`. Update fidelity is `100*max(0,1-d(C,O)/d(B,O))`. Abort if `d(B,O)<1e-4`.
- Collateral retention is mean cosine similarity between baseline and treatment final-query vectors for the other 49 records, times 100.
- Primary metrics are medians across the three learned layer cells. Recurrent state hashes for baseline, treatment and oracle must be distinct in every cell.

## Acceptance gates

- `learned_states`: `learned_gdn_layer_cells ge 3`
- `target_leakage`: `median_old_fact_leakage_pct le 5.0`
- `collateral_retention`: `median_collateral_retention_pct ge 90.0`
- `update_fidelity`: `median_updated_fact_fidelity_pct ge 95.0`
- `state_materiality`: `distinct_recurrent_state_conditions ge 3`

## Abort conditions

- Source/model/corpus mismatch, missing learned GatedDeltaNet modules, fewer than 50 records, unequal condition token lengths, state/output capture failure, nonfinite metric, oracle materiality below `1e-4`, OOM, or service health loss.
- Any post-hoc change to records, target, templates, layers, formulas or gates.

## Allowed claims

- `GDN02_LEARNED_STATE_QUALIFIED_R1`
- `GDN02_NEGATIVE_RETAINED_R1`
- `GDN02_FALSE_NEGATIVE_CONFIRMED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.

This is a learned recurrent-state representation test, not a full-model knowledge-editing claim.
