# RNN-06T2-T0R — DECISION

**Runner:** `ops/rnn_06t2_t0r.py` (committed before outcomes).
**Substrate:** `state-spaces/mamba2-1.3b` @ `c5b59d00ec85d313adea86a08cad2a43c962dd3b`, official
`mamba_ssm` 2.2.4 fast path (chunk_size 256, 48 layers, 52,002,816 state bytes/seq), bf16, RTX 3090.
**Qualification-set SHA-256 (T0R):** `ca92cfad0d0aac4ae20aa8612f259c559ad592415a71797561b3e5909103cafe`
(seeds `{20261050,20261051,20261052,20261060}`, disjoint from all historical seeds).
**Runtime:** 174.8 s, peak VRAM 24.42 GB.

## Mints

```
OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE = QUALIFIED
BATCH_SHAPE_NUMERICAL_PORTABILITY    = OUT_OF_SCOPE_NOT_QUALIFIED
SINGLE_PASS_HISTORICAL_CAPTURE_T0R   = QUALIFIED
```

**T1R gate: OPEN** (both required gates QUALIFIED).

## Lifecycle tests (all fixed batch B=8; BIT_EXACT state, argmax-identical readout)

| Test | Result | Notes |
|---|---|---|
| A deterministic same-path replay | PASS | state bit-exact at all 4 boundaries + readout identical |
| B destroy/reload/restore/continue | PASS | vs **uninterrupted same-path** step continuation (not full prefill); final state bit-exact + readout identical |
| C real branch/fork | PASS | P reproducible from a freshly reconstructed parent (bit-exact); Q independent of P execution; P unaffected by Q; parent immutable after P and Q; branches distinct. **No `or True` tautology.** |
| D fixed-batch neighbor isolation | PASS | focal-row state bit-exact under neighbor order permutation + content replacement; readout argmax invariant |
| E reset / **reuse** equivalence | PASS | dirtied→reset→**reused** cache continuation is BIT_EXACT to a genuinely fresh cache (`364bf104…` == `364bf104…`) + readout identical. **The property RNN-06T never tested.** |
| F serialization roundtrip + **continuation** | PASS | post-roundtrip continuation BIT_EXACT to a no-roundtrip continuation + readout identical (not just immediate hash equality) |
| G fixed-batch slice ownership | PASS | focal-row state + continuation invariant to sibling changes; no batch1 claim |
| H temporal snapshot identity | PASS | 5/5 boundary checks match independent replay; full per-boundary records (conv/ssm/combined hashes, positions) |
| I weight immutability | PASS | checkpoint identity `c5b59d00…` separate from mutation sentinel (unchanged); sentinel labeled non-cryptographic |
| J backend / fast-path identity | PASS | prefill `mamba_chunk_scan_combined`=528 + `causal_conv1d_fn`=528; step `selective_state_update`=`causal_conv1d_update`=235,968; **fallback path count = 0** |

## Property B (batch-shape numerical portability) — OUT_OF_SCOPE

Diagnostic `batch1 vs batchB` focal-row state max-abs diff = **0.5** (> historical `TOL_BATCH` 0.03).
Preregistered as **OUT_OF_SCOPE_FOR_FIXED_BATCH_RECOVERY**: the operational recovery contract holds
batch shape fixed end-to-end, so this property is not required downstream. It is **not** re-labeled
"benign" — it is an unexplained, out-of-scope numerical divergence, preserved as-is. This mint does
not alter the historical RNN-06T strict verdict.

## Single-pass historical capture T0R — QUALIFIED

One trajectory per example (prefill first slot, then step decode under one `InferenceParams`),
in-run captures at boundaries `[156,308,464,616,768]`, same run continued to FINAL. For the held-out
batch, captured state hash == independent same-path replay hash at **every** boundary
(`snapshotBoundaryChecks=5`, `snapshotBoundaryFailures=0`). Per-boundary batch-correct counts
`[0,2,5,4,2]/8` show a genuine forgetting/recovery regime (mid-schedule snapshots beat FINAL) — the
downstream recovery signal, captured in a single pass with no independent prefix re-prefill.

## Authority

Prospective qualification of the fixed-batch operational lifecycle contract. Supersedes the
historical RNN-06T lifecycle mint for the purposes of gating downstream recovery work. Nothing
pushed.
