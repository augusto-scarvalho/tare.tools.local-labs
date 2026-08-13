# RNN-06T2-E1 — ECONOMICS SEMANTIC CLOSURE — PRE-REGISTRATION (frozen before rerun outcomes)

Append-only remediation of ONE semantic false-green in the RNN-06T2-T1R economics arm. The historical
economics mint (`END_TO_END_RECOVERY_UTILITY_T1R = QUALIFIED`, `runs/rnn/RNN-06T2/T1R_ECONOMICS.json`,
`T1R_ECON_run{0,1}.json`, `T1R_DECISION.md` Section 13, `FINAL_DECISION.md`) is **preserved unedited**.
E1 writes only into `runs/rnn/RNN-06T2-E1/` and adds new runners `ops/rnn_06t2_e1_econ.py` /
`ops/rnn_06t2_e1_decide.py`. No historical file is modified.

## The exact defect being fixed (and ONLY this)

In `ops/rnn_06t2_econ.py`:
- `final_fused_equiv` returns `vt[sub.argmax(-1)]` → a **scored VALUE TOKEN ID** (element of the scored
  value token set).
- `final_step_equiv` returns `L.readout(...)[0]` = `vtensor[sub.argmax(-1)]` → a **scored VALUE TOKEN ID**.
- `recovery_equiv` returns `pl.argmax(-1)[...]` → a **column index into `vt`** (range `[0, len(vt))`),
  NOT a token id. The three arms therefore did not return comparable outputs: two returned token ids,
  one returned a vocabulary column index. This is a semantic (output-domain) false-green — it does not
  change the *timings* (a single extra gather is negligible), but it means the arms were not proven to
  compute the same answer object.

**Fix (only this):** map the recovery arm's selected column index back through `vt` so it returns the
same scored VALUE TOKEN ID domain as the other two arms. Nothing else in the economics semantics,
timing method envelope, or thresholds is redefined.

## Executable output-domain assertions (must pass before any timing is trusted)

Before timing, run each arm once on the qualification batch and assert, in code (hard `assert`, run
logged):
1. `DOMAIN_MEMBERSHIP`: every prediction of every arm ∈ `set(scored_value_token_ids)` (the frozen `vt`).
   The **old** recovery output (column indices) fails this; the **fixed** output passes. This is the
   executable proof the false-green is closed.
2. `NOT_COLUMN_INDEX`: for the recovery arm, assert the returned tensor is NOT equal to the raw column
   indices `pl.argmax(-1)` unless `vt` is the identity map (it is not: scored value token ids ≫ len(vt)).
3. `DTYPE_RANGE`: all arm outputs are int64 token ids with `min >= vt.min()` and `max <= vt.max()`.
4. `CROSS_ARM_DOMAIN_IDENTITY`: the set of *possible* outputs (the `vt` set) is identical across arms.

Answer *agreement* across arms is reported descriptively (not asserted): the three arms take different
computational paths (fused prefill vs step trajectory vs restore-from-snapshot) and may differ on
borderline bf16 argmaxes, which is expected and does not affect output-domain comparability.

## Timing method (frozen)

- Every arm executes the SAME semantic task (context + target query + constrained scored VALUE TOKEN
  ID readout), unchanged from RNN-06T2 econ.
- **Randomized / interleaved cycles:** timing is collected in `WARM_ITERS` cycles; within each cycle the
  three arms are timed once each in a **shuffled order** (per-process deterministic RNG seeded by a
  disjoint E1 seed), instead of the historical "all iters of arm A, then all of B, then all of C." This
  removes any systematic thermal/clock-drift bias that could favor whichever arm ran in a fixed slot.
- **≥ 2 clean process starts** (process indices 0 and 1), each a fresh interpreter/model load. RAW
  per-iteration samples persisted; the aggregate pools all process starts.
- Warm iters per arm per process: 40 (same as historical). Batch 16, M=192, schedule as historical.

## Frozen thresholds (NOT redefined post-hoc; identical to T1R_PRE_REGISTRATION.md)

- `ENVELOPE_MS = 250.0` for the primary marginal comparator `RECOVERY_ENABLED − FINAL_STEP`
  (p95 warm, per query). This value was frozen in `runs/rnn/RNN-06T2/T1R_PRE_REGISTRATION.md` BEFORE any
  T1R outcome; E1 reuses it verbatim and does not introduce a new post-outcome threshold.
- Quality gate (context for the marginal-utility mint) reuses the already-committed wide-band recovery
  quality result (`T1R_WIDE_RESULTS.json`: MAX_CONF−FINAL Δ, net_recovery). E1 does NOT rerun recovery
  qualification; it only cites the frozen historical quality number.

## Mints (kept SEPARATE — not collapsed)

- **`ECONOMICS_OUTPUT_COMPARABILITY_E1`** ∈ {QUALIFIED, NOT_QUALIFIED}: QUALIFIED iff all four executable
  output-domain assertions pass on ≥2 process starts (i.e. all arms provably return the same scored
  VALUE TOKEN ID domain).
- **`MARGINAL_RECOVERY_UTILITY_ON_STEP_PATH_E1`** ∈ {QUALIFIED, COST_FAIL, NOT_QUALIFIED}: QUALIFIED iff
  (a) `ECONOMICS_OUTPUT_COMPARABILITY_E1 = QUALIFIED`, (b) p95(RECOVERY − FINAL_STEP) ≤ 250 ms on pooled
  warm samples, and (c) the cited frozen wide quality gate holds (net_recovery > 0 and Δ ≥ 0.05). This is
  the marginal cost of enabling recovery *on the capture-capable step path you already run*.
- **`RECOVERY_PATH_VS_FUSED_BASELINE_E1`** — descriptive characterization of `RECOVERY − FINAL_FUSED`
  (the orthogonal cost of choosing a capture-capable path at all vs the cheapest fused-prefill answer).
  Reported as a magnitude with an explicit verdict label
  ∈ {COMPETITIVE_WITH_FUSED, NOT_COMPETITIVE_WITH_FUSED}: NOT_COMPETITIVE_WITH_FUSED if the p95 gap is
  large (> ENVELOPE_MS). This is intentionally NOT the recovery premium; it is reported so no reader
  can mistake the step-vs-fused cost for the recovery machinery cost.
- **`GENERAL_END_TO_END_DEPLOYMENT_UTILITY = OPEN`** — asserted OPEN unconditionally. Whether recovery is
  worth its cost in a *real* deployment depends on whether a realistic workload exhibits a forgetting
  regime where recovery adds value — which is exactly what RNN-07A investigates and has not established.
  E1 does not and cannot close this.

## Seeds (disjoint from all prior RNN-06T2 seeds)

- E1 timing-shuffle seed base: `20261200` (process 0 uses `20261200`, process 1 uses `20261201`).
- Pool seed reused (frozen data): `20260817` (identical scored-value pools as historical econ).

## Constraints honored

No lifecycle rerun. No synthetic recovery-qualification rerun. No threshold redefinition. No Qwen, no
selector/reader training, no host-policy change. Append-only; nothing pushed.
