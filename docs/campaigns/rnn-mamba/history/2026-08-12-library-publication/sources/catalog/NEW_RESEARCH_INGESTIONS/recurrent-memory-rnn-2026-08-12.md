# Live Research Ingestion — Recurrent Memory / RNN / Memory Caching — 2026-08-12

**Decision:** `INTEGRATED_AS_EXPERIMENTAL_LINEAGE_WITH_AUDIT_RECONCILIATION`

## Publication targets

- Experimental lineage: [`experiments/local-llm/recurrent-memory/README.md`](../../experiments/local-llm/recurrent-memory/README.md)
- Machine ledger: [`RESULTS_LEDGER.json`](../../experiments/local-llm/recurrent-memory/2026-08-12/RESULTS_LEDGER.json)
- Independent RNN-06T audit: [`RNN-06T-AUDIT-RECONCILIATION.md`](../../experiments/local-llm/recurrent-memory/2026-08-12/RNN-06T-AUDIT-RECONCILIATION.md)
- Prospective RNN-06T2 protocol: [`RNN-06T2-PROTOCOL.md`](../../experiments/local-llm/recurrent-memory/2026-08-12/RNN-06T2-PROTOCOL.md)
- Independent RNN-06T2 audit: [`RNN-06T2-AUDIT-RECONCILIATION.md`](../../experiments/local-llm/recurrent-memory/2026-08-12/RNN-06T2-AUDIT-RECONCILIATION.md)
- Next-train design: `RNN-07A` realistic operating-point discovery + RNN-06T2 economics semantic closure
- Research pointers: [`recurrent-memory-rnn-2026-08-12-pointers.md`](recurrent-memory-rnn-2026-08-12-pointers.md)

## Authority

`EXPERIMENTAL / RESEARCH / PROPOSED` only.

This ingestion does not mint tare.tools canonical CURRENT/TARGET, promote Memory Caching into architecture, authorize Qwen work, or rewrite failed historical gates.

## Latest independent audit

`RNN-06T2` is accepted with an economics reconciliation.

Load-bearing experimental state for the exact official Mamba-2 fixed-batch subject:

```text
OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE = QUALIFIED
BATCH_SHAPE_NUMERICAL_PORTABILITY = OUT_OF_SCOPE_NOT_QUALIFIED
SINGLE_PASS_HISTORICAL_CAPTURE_T0R = QUALIFIED

HISTORICAL_RECOVERY_NARROW = QUALIFIED
ADAPTIVE_SELECTION_NARROW = DIRECTIONAL

WIDE_TARGET_RECOVERY_T1R = QUALIFIED
ADAPTIVE_SELECTION_T1R = QUALIFIED
```

The wide synthetic qualification independently reproduces:

```text
FINAL               = 0.2708333
FIXED_SLOT_153      = 0.5000000
MAX_CONFIDENCE      = 0.8125000

MAX_CONF - FINAL    = +0.5416667
95% CI              = [0.4738281, 0.609375]

MAX_CONF - SLOT153  = +0.3125000
95% CI              = [0.2395833, 0.3751302]
```

The primary adaptive contrast is positive in 3/4 preregistered strata.

## Economics reconciliation

The historical implementer mint:

`END_TO_END_RECOVERY_UTILITY_T1R = QUALIFIED`

is preserved as historical evidence but reconciled as a false green.

The economics source returned scored token IDs for FINAL arms but scored-vocabulary column indices for RECOVERY. The frozen protocol required the same answer domain for all arms and prescribed `NOT_COMPARABLE` otherwise.

Current state:

```text
END_TO_END_RECOVERY_UTILITY_T1R_HISTORICAL_MINT = RECONCILED_FALSE_GREEN
END_TO_END_RECOVERY_UTILITY_T1R = NOT_COMPARABLE
MARGINAL_STEP_PATH_TIMING_SIGNAL = POSITIVE_NON_LOAD_BEARING
```

This does not invalidate lifecycle or recovery qualification and does not require their rerun.

## Provenance caveats retained

- batch1-vs-batchB state difference `0.5` remains negative evidence; batch-shape portability is not qualified.
- a post-outcome T1R instrumentation rerun is accepted with provenance caveat; future outcome-exposed runner changes must package the exact diff.
- the T1R display `runId` is not globally unique because it was derived from a common short prefix; state/result identities are unaffected.

## Current frontier

Scientific discovery may now move to a realistic long-context operating-point scout on the same exact official Mamba-2 subject, while carrying a small economics semantic-closure task.

`RNN-07A` remains `PROPOSED` until a new implementer session executes it. No RNN-07A result is minted here.

Qwen remains `DEFER`.

## Exact source identities

- RNN-06T2 bundle SHA-256: `52fcf4d00430bb8b24da3c2cfd8b5a4c1c2473c701b2939acbd0f633e4a35426`
- external RNN-06T2 handoff SHA-256: `d6d409f7f1a7db00f01af9f6b005d467487969f4a2a94dc2ae9ba464b59cbc53`
- independent RNN-06T2 audit SHA-256: `72fab88e53391692e80803e0bddb8fecde85c309f7d02f33560e9d001aa69b48`
- RNN-07A proposed protocol SHA-256: `63941f9af9205635c1f10b8b27ce751fc17207e18d927f4bf6b3ba1cd1d09e65`

Raw workstation ZIPs are not copied into the public research tree by default; exact digests are preserved instead.
