# BACKLOG-BEE-L3-REAL-TELEMETRY-01 preregistration

Task: Audit adaptive MTP controller with paired live K0 K2 K4 telemetry
Evidence class: `serving_runtime`

## Hypothesis

Replaying the frozen adaptive controller over paired physical K0/K2/K4 observations will retain the historical claims: at least 1.25x speedup over K0, at least 15% gain over static K4, and at least 95% throughput protection on the bottom acceptance quartile, with exact output parity. Failure of any gate confirms the simulation-only qualification was a false positive.

## Frozen inputs

- `runs/research/BEE-L3-MTP-CONTROLLER-2026-08-25/PRE_REGISTRATION.md`
- `runs/research/BEE-L3-MTP-CONTROLLER-2026-08-25/RESULT.md`
- `runs/research/BEE-L3-MTP-CONTROLLER-2026-08-25/raw/receipt.json`
- `tools/analysis/adaptive_mtp_controller.py`
- `tests/test_adaptive_mtp_controller.py`
- `runs/research/BACKLOG-ADAPT-TRAIN-01/raw/samples.jsonl`

- Historical prereg/result/receipt/controller/test: `cfb661dca692d6d857fa9e854e6f00a436e42675929ebfb740dd384015e067a4`, `a26f33d2d93f097ed425be19723d6096ad7faba16f90080aa4dad938408d9720`, `d13d51e10ceda38fe611cf4c4b6efaa12ac1fa22461b0aaa2a6fe6f1e61fee6d`, `7ede4879ddf8d94a5efcbd4b4b2ca2ab0ed70353004822d83fbe442b2a986e9e`, `c5074759e4cbb4c1379468a15ca4d051b7b2ac80dffc38f981f55d0db0a3f12a`.
- Frozen prompt source: `runs/research/BACKLOG-ADAPT-TRAIN-01/raw/samples.jsonl`, SHA-256 `243311c37ff240d97f63539c4e85f3a9ec7272ea8eaa1279d31c7c38d44d50c4`; select the first 48 unique base-arm prompts in file order.
- Active binary/model/ExecStart are captured and hashed before maintenance.

## Command

```powershell
python tools/research/run_bee_l3_real_telemetry.py --outdir runs/research/BACKLOG-BEE-L3-REAL-TELEMETRY-01
```

## Factors

- For every prompt, run deterministic `n_predict=64`, `temperature=0`, `top_k=1`, `seed=0`, `cache_prompt=false` under physical K4, K0 and K2 routes using the same binary/model and four slots.
- K4 is the original systemd route. K0 omits speculative flags; K2 uses `draft-mtp` with `spec-draft-n-max=2`. Root systemd handoff and exact path/argv restoration are mandatory; 8081 remains untouched.
- Require all three outputs for each prompt to match exactly. Record wall latency, server predicted timing, draft count and accepted draft count.
- Feed actual K4 `(accepted,drafted)` observations sequentially to the unmodified controller. Map recommended K 0..1 to measured K0, 2..3 to measured K2, and 4 to measured K4 for that same prompt. This is a counterfactual replay over physical arms, not live request-level switching.
- Define low-acceptance prompts before scoring as the bottom quartile of K4 accepted/drafted rate (stable tie break by task id). Compute adaptive replay throughput from measured per-cell wall latency and compare with K0/K4.

## Acceptance gates

- `arm_coverage`: `live_requests eq 144`
- `semantic_parity`: `paired_exact_parity_rate eq 1.0`
- `mtp_telemetry`: `k4_requests_with_drafts ge 40`
- `global_speedup`: `adaptive_replay_speedup_over_k0 ge 1.25`
- `static_gain`: `adaptive_replay_gain_over_k4 ge 0.15`
- `low_protection`: `low_acceptance_protection ge 0.95`
- `service_restore`: `original_service_restored eq 1`
- `embedding_integrity`: `embedding_health eq 200`

## Abort conditions

- Hash drift, fewer than 48 unique prompts, any route/start/request error, missing draft telemetry, output mismatch during baseline collection, 8081 failure, or inability to restore immutable systemd path/argv aborts.
- No controller setting, K mapping, prompt, sample, decode control, threshold, or low-acceptance definition may change after observation.

## Allowed claims

- `BEE_L3_REAL_TELEMETRY_QUALIFIED_R1`
- `BEE_L3_FALSE_POSITIVE_CONFIRMED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.

No live per-request K switching, production integration, out-of-panel generalization, or simulated-cost-as-hardware-latency claim is permitted.
