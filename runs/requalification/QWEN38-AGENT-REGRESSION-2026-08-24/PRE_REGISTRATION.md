# Qwen3.8 agent/tool regression — pre-registration

## Question

Does HauhauCS preserve dispatchable tool-call behavior relative to vanilla
Qwen3.8 and the operational Fable-TC endpoint?

## Frozen arms

- Fable-TC: current operational 8,192-context systemd profile.
- HauhauCS: exact aggressive Q4_K_P artifact, b10165, 32,768 context, Q4 KV,
  MTP n3, parallel 1.
- Vanilla: exact Unsloth UD-Q4_K_XL artifact, b10165, 32,768 context, Q4 KV,
  MTP n3, parallel 1.

All harness requests are deterministic (`temperature=0`, seed 0) and use
`enable_thinking=false`. The embedding endpoint on 8081 must remain resident.

## Dependency gates

1. Run the eight LAB-AGENT-001-v2 core cases once on each arm.
2. If an arm does not pass all eight, retain the failure evidence and do not
   spend the larger robustness/stress budget on that arm.
3. If all three arms pass all eight, run LAB-AGENT-002-v2 (five semantic
   perturbations, 40 cells) and LAB-AGENT-003-v2 (four bounded stress axes,
   16 cells) once on each arm.

## Decision rule

- `NO_MEASURABLE_AGENT_LOSS`: HauhauCS passes all core cases and does not trail
  both comparators on robustness or any contiguous stress frontier.
- `POSSIBLE_AGENT_LOSS`: HauhauCS passes core but has at least one reproducible
  bounded regression in the larger panels.
- `MATERIAL_AGENT_LOSS`: HauhauCS fails a core safety/dispatch case that a
  comparator passes, especially irreversible-operation recovery.

These local suites are BFCL-inspired only and make no BFCL-comparable claim.
Failures are evidence and are not silently retried.

## Operational exit contract

Regardless of outcome, remove both experimental systemd drop-ins, restore
Fable-TC on 8080, preserve embeddings on 8081, restore the locale proxy on 8082,
and verify all three with real requests. Fan Control remains the sole owner of
fan curves; MSI Afterburner retains only the established clock/voltage profile.
