# BACKLOG-DISTILL01-FLEET-REAL-01 preregistration

Task: Recompute DISTILL-01 routed-fleet superiority from clean process-isolated real generations
Evidence class: `mechanism_research`

## Hypothesis

Applying the original DISTILL-01 routing rule to clean, process-isolated real generations will reproduce at least 20% relative composite gain over `target_all_linear`, with `target_mlp_only` scoring at least 15/32 math and `target_attn_only` at least 5/16 QA. Failure of any rule falsifies the historical promotion on the frozen panels.

This evaluates routing composition of saved adapters, not a causal effect of distillation training.

## Frozen inputs

- `runs/research/BACKLOG-ADAPT-REQUAL-02/PRE_REGISTRATION.md`
- `runs/research/BACKLOG-ADAPT-REQUAL-02/raw/receipt.json`
- `runs/research/BACKLOG-ADAPT-REQUAL-02/raw/samples.jsonl`
- `runs/research/BACKLOG-ADAPT-REQUAL-02/raw/artifact_hashes.json`
- `runs/research/DISTILL-01-FLEET-DISTILLATION-2026-08-25/PRE_REGISTRATION.md`
- `runs/research/DISTILL-01-FLEET-DISTILLATION-2026-08-25/RESULT.md`
- `runs/research/DISTILL-01-FLEET-DISTILLATION-2026-08-25/raw/receipt.json`
- `tools/probes/distill01_fleet_distillation.py`

- Admission SHA-256: `cdc85abf2dd596df8643a563abddc12dfc5ad2c82b8e8bd0d01e7b41df4542e5`.
- Clean successor preregistration/receipt/samples/artifact-ledger SHA-256: `9b528e0f9542778fd8dbbc2dcddf730a3e63f46aa130a2b5849e1f99181f0ffd`, `8bc38d1f2cb5ef60f53ddb989e5c0aa1104b81359efbc5a2c8e4bbd0d92bc876`, `8900194aa5abc38092f7e5d99122c7322de8781c5aff4ef402d812fb6dfb2a8c`, `b19fa60e5d122219934a1563cdf231dac0a847393327d35b214763711582c5fc`.
- Original DISTILL-01 preregistration/result/receipt/probe SHA-256: `dd5889aaf3767b67bd24dd805775dce3c0bf20b3291e600311adc7793bf86c10`, `a663e7807c06faf5aaa47ee26de64eae7ccbe0bf05e0580cc3f3b8e8e21ab3ab`, `e71f1831345356b6e1dc5d20f960b533daf1dc7e2f61d15fc869d430e120a8a9`, `ccadd6e28e8ad8bbb9c40e7f512aa3cb5f260f335c21c2711b37ab25169008cc`.

## Command

```powershell
python tools/research/run_distill01_fleet_real.py --outdir runs/research/BACKLOG-DISTILL01-FLEET-REAL-01
```

## Factors

- Required real arms: `target_mlp_only`, `target_attn_only`, `target_all_linear`; exactly 48 samples per arm, comprising the same 32 math and 16 protected-QA IDs.
- Fleet rule frozen from the predecessor: take math outputs only from `target_mlp_only` and QA outputs only from `target_attn_only`.
- Monolith control: take both panels from `target_all_linear`.
- Independently re-extract every math answer and regrade every QA output using the implementation-bound scorers; stored `correct` flags are not trusted.
- Primary gain is `(fleet_total - monolith_total) / monolith_total`; threshold `>=0.20`. Secondary mandatory thresholds remain math `>=15` and QA `>=5`.
- The source receipt must verify 17 clean-base workers and the exact saved adapter identities. This successor performs no new model inference because its estimand is a deterministic routing composition over already reexecuted physical generations.
- Raw output contains a copied semantic projection of all 144 selected samples plus independent scores and source hashes; original evidence is not edited.

## Acceptance gates

- `source_execution`: `source_real_execution_verified eq True`
- `arm_coverage`: `complete_required_arms eq 3`
- `sample_coverage`: `complete_required_samples eq 144`
- `fleet_gain`: `fleet_gain_over_monolith ge 0.2`
- `math_specialist`: `fleet_math_correct ge 15`
- `qa_specialist`: `fleet_qa_correct ge 5`
- `independent_scoring`: `independent_rescore_match eq True`

## Abort conditions

- Any frozen source hash, source receipt fingerprint, adapter identity, panel membership or scorer binding differs.
- Any required arm has fewer than 32 unique math plus 16 unique QA samples.
- Stored correctness flags disagree with independent rescoring.
- Routing rules, denominators or thresholds change after scores are observed.
- Provenance or independent recomputation is incomplete.

## Allowed claims

- `DISTILL01_FLEET_QUALIFIED_R1`
- `DISTILL01_FALSE_POSITIVE_CONFIRMED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.
