# BACKLOG-AGY-SYSTEM-BLOCKERS-01 preregistration

Task: Register objective integration blockers for six AGY system claims
Evidence class: `proxy_realization`

## Hypothesis

Each of six remaining AGY system claims lacks both a callable deployed runtime surface and a corresponding implementation in the inspected candidate source/model inventory; therefore physical rerun is objectively blocked. Any matching implementation or artifact invalidates that item's blocker and requires a successor experiment instead.

## Frozen inputs

- `runs/research/SLX-03-STATE-WRITE-ELISION-2026-08-25/raw/receipt.json`
- `tools/probes/slx03_state_write_oracle.py`
- `runs/research/SLX-07-H2O-EVICTION-2026-08-25/raw/receipt.json`
- `tools/probes/slx07_h2o_eviction_oracle.py`
- `runs/research/REP-04-KVARN-NATIVE-KERNEL-2026-08-25/raw/receipt.json`
- `tools/probes/rep04_kvarn_native_kernel.py`
- `runs/research/REP-05-LAYERWISE-PRECISION-2026-08-25/raw/receipt.json`
- `tools/probes/rep05_layerwise_kv_precision.py`
- `runs/research/RETRO-01-RECURRENT-RETROFIT-2026-08-25/raw/receipt.json`
- `tools/probes/retro01_recurrent_retrofit.py`
- `runs/research/SLX-08-SPECULATIVE-PREFILL-2026-08-25/raw/receipt.json`
- `tools/probes/slx08_speculative_prefill_oracle.py`

- SLX-03 receipt/probe: `10a7ea78bcf66991700ca40d217e1e7e83b0f2d25ee1e45fc651c3399b68d5a9`, `b398b691268baf3f933a9fd45a36ad71e86c62fc6e4fdb02467ea8e0531a3e74`.
- SLX-07 receipt/probe: `405072005b7b17897b3a46200d53f204366b2fbe148113e69039d5b6f283f198`, `98af7d6e56cec47d481154ef0ddc41e4654d0cb79f3b22466e92f9337ad3376d`.
- REP-04 receipt/probe: `66c163d2d161041caae80aed3e1c8689f48522ba3c197f96b8fef07e531898c8`, `0093f09b1b9f7055a6b0fab565014cf03db3726ed878b829ebc84e3072eb0ac8`.
- REP-05 receipt/probe: `9c23e665e02fdfc8fd03c21e58f183bfcf8f1dee435803f1b51b705c0b74e040`, `9787da10ad8026df424f295f91618d8525928d5c218c1f9d2a3f6333de5b4067`.
- RETRO-01 receipt/probe: `c3092d2fdb6a46067794af5d79e2684fd7722a383bc0f223a0fbe76f876af42f`, `2328c1b83273811117234b5f7ffd95404efcdea103f6ce98c33e5b1bf0f6874d`.
- SLX-08 receipt/probe: `f19600ed451d5ed4ad3a24b5c29ef3fbcf2a95de06ffcab0aef9a7e9152cb78a`, `5b85dd266c3fc72ae47a7cabe6e5ae3246e4aab544e87e6ee7cd47eab81bdc37`.
- Deployed binary SHA-256, full `--help`, candidate source Git identity/status, grep outputs, and model inventory are captured at execution.

## Command

```powershell
python tools/research/run_agy_system_blockers.py --outdir runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-01
```

## Factors

- Inspect the deployed binary help and printable symbol matches, plus `/home/augus/src/slop.cpp-main` at its captured Git commit, for preregistered feature-specific patterns.
- SLX-03 requires a compiled state-write elision/checkpoint cadence surface; SLX-07 requires H2O/heavy-hitter KV lifecycle code; REP-04 requires a KVarN fused kernel (generic Hadamard is insufficient); REP-05 requires per-layer KV precision configuration (global K/V types are insufficient); RETRO-01 requires a trained recurrent-retrofit checkpoint; SLX-08 requires selected-block speculative prefill integrated into TTFT.
- Inspect `/home/augus/models` for a recurrent-retrofit artifact. Record all exact commands, return codes and outputs. Do not mutate services or rerun predecessor Python proxies.
- For each item, absence in deployed surface and candidate implementation yields one objective blocker with a precise unlock criterion; any positive match aborts that blocker and sets `missing_integration_count < 6`.

## Acceptance gates

- `scope_coverage`: `audited_items eq 6`
- `objective_absence`: `missing_integration_count eq 6`
- `probe_classification`: `proxy_only_predecessors eq 6`
- `runtime_integrity`: `runtime_unchanged eq 1`

## Abort conditions

- Source Git identity unavailable, binary identity unavailable, any frozen hash mismatch, search command failure, service identity/health change, or ambiguous positive match aborts the blocker registration.
- No scientific rejection or permanent infeasibility may be inferred from implementation absence.

## Allowed claims

- `AGY_SYSTEM_BLOCKERS_REGISTERED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.

This packet registers current integration blockers only; it is not a performance experiment.
