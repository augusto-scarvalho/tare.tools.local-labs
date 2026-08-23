# LAB-OPT-001 result — qualified screen, deploy decision withheld

## Outcome

The bounded Optuna screen completed successfully: all six Round-1 configurations passed health,
4 GiB free-VRAM, telemetry, non-empty generation, and three byte-identical greedy-output gates.
The successive-halving rule advanced `n3/ub2048`, `n3/ub1024`, and `n4/ub1024`; all three completed
three counterbalanced short/long repetitions.

Within the *explicitly configured* screen, `n4/ub1024` met the frozen recommendation rule versus
`n3/ub2048`: median long-prompt throughput was 1.20% higher (1276.34 vs 1261.24 tok/s) and median
decode throughput was 7.08% higher (82.13 vs 76.70 tok/s). It left 5,048 MiB free after load,
peaked at 19,313 MiB used, and produced the same three output hashes as every other feasible cell.

## Interpretation correction

This is **not yet a deploy-default promotion**. After the run, the canonical systemd unit and binary
help were reconciled. The service does not pass `--ubatch-size`; this binary's documented default is
512, not the 2,048 value pre-registered as the incumbent. The raw Optuna result is valid for the six
explicit configurations, but its `decision.recommended` field compares against an assumed control,
not the exact live default.

The operational conclusion is therefore `QUALIFIED_SCREEN / CONTROL_MISMATCH / NO_DEFAULT_CHANGE`.
LAB-OPT-001b must compare the exact live-equivalent `n3/ub512` control with `n4/ub1024` at the
canonical 131,072-token allocation before any configuration-file change is considered.

## Evidence

- Frozen packet: `PRE_REGISTRATION.md`
- Machine-readable study and all Optuna trials: `results.json`
- Per-cell request/telemetry receipts: `trials/`
- Per-server startup logs: `logs/`
- Optuna version: 4.9.0, pinned in `requirements-experiments.txt`
- Candidate servers exited cleanly; port 8092 was freed.
- `llm-inference.service` was restored unchanged and both 8080 and 8081 returned healthy.
