# BACKLOG-SPEC01-LIVE-HYBRID-01 preregistration

Task: Materialize SPEC-01 combined ngram and MTP serving route
Evidence class: `serving_runtime`

## Hypothesis

The deployed combined `draft-mtp,ngram-cache` route will preserve all paired greedy outputs, deliver at least 3.0x wall-throughput speedup over active MTP alone, expose enough per-proposer telemetry to substantiate the historical 31.64% n-gram share, and use draft tokens on at least 25/30 structured prompts. Any failed gate rejects the synthetic promotion.

## Frozen inputs

- `runs/research/SPEC-01-SPECULATIVE-PIPELINE-2026-08-25/PRE_REGISTRATION.md`
- `runs/research/SPEC-01-SPECULATIVE-PIPELINE-2026-08-25/RESULT.md`
- `runs/research/SPEC-01-SPECULATIVE-PIPELINE-2026-08-25/raw/receipt.json`
- `tools/analysis/hybrid_speculative_engine.py`
- `tests/test_hybrid_speculative_engine.py`

- Historical prereg/result/receipt/engine/test: `62fecb2683c511546b8b37703698673f4806869dec7a7094ead33c61b77e16b1`, `e4afcf36f830ff0ebbe32a8971065d68797beed169f9d928c56a025169c619fc`, `26b0c2857396cb080659a0217dd5b88bebe08c876655caab5c57b82a0d4c7da0`, `7a3c8456498848f003476137a88b3e472ac5d1a91be476bd57ab8ac93f5d44ab`, `61c627f1d1874efbf42d25858105211042b55e182885fcbc34cf2697022f4c50`.
- Deployed binary/model/systemd identities are captured and hashed before execution.

## Command

```powershell
python tools/research/run_spec01_live_hybrid.py --outdir runs/research/BACKLOG-SPEC01-LIVE-HYBRID-01
```

## Factors

- Freeze 30 prompts indexed 0..29. Each prompt contains a deterministic 24-item repeating JSON object pattern and asks the model to continue it exactly; the generator and semantic hash are recorded before inference.
- Baseline is the active `draft-mtp` K4 route. Treatment is the same binary/model/settings with `--spec-type draft-mtp,ngram-cache --spec-draft-n-max 4`, four slots, `n_predict=128`, greedy seed 0 and no prompt cache.
- Record full responses, wall/predicted timing, aggregate draft/accepted counts, and `generation_settings["speculative.types"]`. Exact content parity is mandatory.
- `per_proposer_attribution_available=1` only if physical responses or exported metrics separately report n-gram-proposed/accepted tokens versus MTP; aggregate `draft_n` is insufficient.
- Root-stop the original service only for treatment, keep 8081 healthy, then restore immutable executable path/argv.

## Acceptance gates

- `request_coverage`: `live_requests eq 60`
- `hybrid_route`: `hybrid_route_confirmed eq 30`
- `semantic_parity`: `exact_output_rate eq 1.0`
- `historical_speedup`: `hybrid_speedup ge 3.0`
- `ngram_attribution`: `per_proposer_attribution_available eq 1`
- `draft_coverage`: `hybrid_requests_with_drafts ge 25`
- `service_restore`: `original_service_restored eq 1`
- `embedding_integrity`: `embedding_health eq 200`

## Abort conditions

- Hash/route/start/request failure, treatment not advertising both types, 8081 failure, panel drift, or inability to restore original service aborts.
- No prompt, route type, decode control, attribution rule, threshold or metric may change after observation.

## Allowed claims

- `SPEC01_LIVE_HYBRID_QUALIFIED_R1`
- `SPEC01_FALSE_POSITIVE_CONFIRMED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.

No unobserved n-gram share, forward-count reduction, production, or out-of-panel claim is permitted.
