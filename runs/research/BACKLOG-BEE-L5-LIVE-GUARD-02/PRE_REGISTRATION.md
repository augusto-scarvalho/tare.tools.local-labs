# BACKLOG-BEE-L5-LIVE-GUARD-02 preregistration

Task: Rerun BEE-L5 live guard with corrected server token-piece parser
Evidence class: `serving_runtime`

## Hypothesis

Changing only the `/tokenize` response parser from nonexistent top-level `pieces` to active `tokens[*].piece` will allow the frozen R1 protocol to run. The same nine numerical gates and the same 128 real traces, 25 live baselines, prompts, pathology rule, guard, and runtime controls remain binding.

## Frozen inputs

- `runs/research/BACKLOG-BEE-L5-LIVE-GUARD-01/PRE_REGISTRATION.md`
- `runs/research/BACKLOG-BEE-L5-LIVE-GUARD-01/ABORTED.md`
- `tools/research/run_bee_l5_live_guard.py`
- `runs/research/BEE-L5-REASONING-LOOP-GUARD-2026-08-25/raw/receipt.json`
- `tools/analysis/reasoning_loop_guard.py`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/teacher_samples.json`

- R1 preregistration/abort/implementation SHA-256: `122f9a436259697ff6aa26a5972cd7d15fe733ece6c89783a19bf4b19fb22754`, `ae1da170d097684dcfb72e8d603bcb78f5a1f19f8266b64a768fbe157d10cc52`, `7ce5e9cd47fde2d5e53c923b012d4cef2e665b27b634f917e3f6e311e79ab5f8`.
- Historical receipt/guard SHA-256: `3e279af76d4f357cc1a5fba1dff0892790d95f4813c63cc59373eee4973cd933`, `a4bf8e7f29ff4cff5dbfefbb8c3accdfad24a9e43803ed8525aa0d44366e6358`.
- Teacher samples SHA-256: `e545dfa0a35b97b00b72a1f1b32e35052a083b74d7bf8b26145ec6e87dcd102a`.

## Command

```powershell
python tools/research/run_bee_l5_live_guard_r2.py --outdir runs/research/BACKLOG-BEE-L5-LIVE-GUARD-02
```

## Factors

- Identical to frozen R1: 128 `selected["20260824"]` real teacher traces and 25 deterministic live baseline/stream pairs at 128 tokens.
- The only implementation delta is extracting exact pieces from `response["tokens"][*]["piece"]`.
- Baseline pathology remains independently defined as 128 predicted tokens, at least 40 pieces, and at most eight distinct pieces in the final 40.
- Historical guard settings remain window 32, three reversals, and three repeated four-grams; the service route must remain four-slot `draft-mtp`.

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

- All R1 abort conditions remain binding: hash, route, dataset, baseline pathology, request, service identity, and embedding health failures abort.
- No numerical threshold, prompt, sample, guard setting, or pathology criterion may change after observation.

## Allowed claims

- `BEE_L5_LIVE_GUARD_QUALIFIED_R2`
- `BEE_L5_FALSE_POSITIVE_CONFIRMED_R2`

Claims outside these codes are forbidden even if a metric looks favorable.

No natural prevalence, server-side integration, answer preservation, deployment, or out-of-panel claim is permitted.
