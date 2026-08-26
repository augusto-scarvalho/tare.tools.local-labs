# BACKLOG-BEE-L5-LIVE-GUARD-01 preregistration

Task: Audit BEE-L5 on real teacher traces and live streaming intervention  
Evidence class: `serving_runtime`

## Hypothesis

The historical BEE-L5 guard will avoid firing on at least 98% of 128 frozen, real ThinkingCap teacher traces and will fire on at least 95% of 25 independently generated live pathological streams. Closing each stream at the first trigger must save a median 80% of the frozen 128-token budget, leave all four slots idle, and not restart either service. Any failed gate confirms that the synthetic-only promotion overstated qualification.

## Frozen inputs

- `runs/research/BEE-L5-REASONING-LOOP-GUARD-2026-08-25/PRE_REGISTRATION.md`
- `runs/research/BEE-L5-REASONING-LOOP-GUARD-2026-08-25/RESULT.md`
- `runs/research/BEE-L5-REASONING-LOOP-GUARD-2026-08-25/raw/receipt.json`
- `tools/analysis/reasoning_loop_guard.py`
- `tests/test_reasoning_loop_guard.py`
- `runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/teacher_samples.json`

- Historical preregistration/result/receipt SHA-256: `22709ab07db66a4ba3506ab82dc5699f2bc4828b0e42c10882cf3d12f2a3690d`, `e3a542796c62d543b82d873e1b9ed07ad423f09cf4ef06b51b030a92febda885`, `3e279af76d4f357cc1a5fba1dff0892790d95f4813c63cc59373eee4973cd933`.
- Historical guard/test SHA-256: `a4bf8e7f29ff4cff5dbfefbb8c3accdfad24a9e43803ed8525aa0d44366e6358`, `36f187c30dcd9557183eed2d316a90fc53573cbe601bf97bb1e3164f57833c44`.
- Frozen teacher sample SHA-256: `e545dfa0a35b97b00b72a1f1b32e35052a083b74d7bf8b26145ec6e87dcd102a`; use exactly the 128 `selected["20260824"]` full traces.
- The active endpoint is `http://127.0.0.1:8080`; systemd identity, server binary, arguments, slots, logs, and 8080/8081 health are captured before and after.

## Command

```powershell
python tools/research/run_bee_l5_live_guard.py --outdir runs/research/BACKLOG-BEE-L5-LIVE-GUARD-01
```

## Factors

- Tokenize every frozen teacher trace through active `/tokenize` with pieces returned; feed those exact pieces to the unmodified historical `ReasoningLoopGuard` and record triggers and per-token latency.
- Generate 25 non-stream baselines using a frozen prompt ending with 12 repetitions of `wait let me reconsider now`, `temperature=0`, `top_k=1`, `seed=case index`, `ignore_eos=true`, and `n_predict=128`.
- A baseline is independently pathological only if it exhausts 128 predicted tokens and its last 40 token pieces contain at most eight unique pieces. Abort before scoring if any baseline fails this definition.
- Repeat each prompt through actual SSE streaming with identical controls. Feed exact returned token pieces to the guard and close the response immediately at first trigger; confirm the slot returns idle.
- No output may be edited or relabeled after observation. The historical guard implementation and settings remain unchanged.

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

- Frozen hashes mismatch; route is not four idle `draft-mtp` slots; the dataset is not 128 unique task IDs; any baseline is not independently pathological; any request errors; service/binary identity changes; or embedding health fails.
- Any threshold, prompt, model setting, guard setting, sample membership, or pathology definition changes after outputs are observed.

## Allowed claims

- `BEE_L5_LIVE_GUARD_QUALIFIED_R1`
- `BEE_L5_FALSE_POSITIVE_CONFIRMED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.

This experiment does not establish natural loop prevalence, server-side integration, final-answer preservation, production deployment, or generalization beyond the frozen traces and prompts.
