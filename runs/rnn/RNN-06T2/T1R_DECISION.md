# RNN-06T2-T1R — DECISION

Executed because T0R qualified both gates. Fresh, disjoint calibration/qualification data.
`MAX_CONFIDENCE` frozen. All primary comparisons paired; region-stratified bootstrap (2000 resamples).
Qualification conditional on `state-spaces/mamba2-1.3b` @ `c5b59d00…` (official `mamba_ssm` 2.2.4 fast
path). Fixed batch 16.

## Mints

```
HISTORICAL_RECOVERY_NARROW      = QUALIFIED
ADAPTIVE_SELECTION_NARROW       = DIRECTIONAL   (not required to qualify in the narrow band)
WIDE_TARGET_RECOVERY_T1R        = QUALIFIED
ADAPTIVE_SELECTION_T1R          = QUALIFIED
END_TO_END_RECOVERY_UTILITY_T1R = QUALIFIED
```

`WIDE_TARGET_RECOVERY_T1R` and `ADAPTIVE_SELECTION_T1R` are **independent** mints — not collapsed into
a single "Mamba PASS".

## Narrow band [8,64] (Section 7 — formal transport replication vs RNN-06D)

Set `T1R_NARROW_QUAL_SPEC.json` (seed 20261110, disjoint). Arms:

| arm | acc |
|---|---|
| FINAL | 0.162 |
| FIXED_SLOT_76 | 0.734 |
| MAX_CONFIDENCE | 0.839 |
| ORACLE_BEST_GOLD (diag) | 0.906 |
| all fixed | 38:0.495 · 76:0.734 · 115:0.438 · 153:0.250 |

- `FIXED_SLOT_76` vs FINAL: **+0.573** CI[0.505,0.646], net_recovery=110, robust 4/4 →
  `HISTORICAL_RECOVERY_NARROW = QUALIFIED`. The historical hypothesis holds: in the narrow band a
  single fixed checkpoint (slot76) captures most of the recovery value.
- `MAX_CONFIDENCE` vs `FIXED_SLOT_76`: +0.104 CI[0.047,0.161] but robust 2/4 <3/4 →
  `ADAPTIVE_SELECTION_NARROW = DIRECTIONAL`. As preregistered, adaptive selection is **not required**
  to qualify in the narrow band.

## Wide band [8,144] (Sections 8–11 — primary confirmatory)

Interpretation frozen before outcomes: slot153 is AFTER every target in [8,144]; the adaptive problem
is the NOT_YET_WRITTEN vs SEEN_AND_RETAINED vs ALREADY_FORGOTTEN tradeoff.

**Tie policy (Section 9).** Calibration (`T1R_WIDE_CALIBRATION.json`, seed 20261120) fixed-slot acc
`{38:0.245, 76:0.380, 115:0.458, 153:0.537}`. With `TAU_TIE = 0.02`, only slot153 is within tolerance
of the best → `CARRIED_FIXED_CONTROLS = [153]`, strongest carried = slot153. Deterministic tie-break =
highest calib acc, ties → smallest slot (no dict-order dependence).

**Qualification** (`T1R_WIDE_QUAL_SPEC.json`, seed 20261121, disjoint). Fixed-slot acc on qual
`{38:0.219, 76:0.370, 115:0.443, 153:0.500}`; FINAL 0.271; MAX_CONFIDENCE 0.813; ORACLE_BEST_GOLD
0.891.

- **`WIDE_TARGET_RECOVERY_T1R`** — MAX_CONFIDENCE vs FINAL: **+0.542** CI[0.474,0.609],
  recovery_rate 0.757, harm_rate 0.039, net_recovery=104, robust 4/4 → **QUALIFIED**
  (Δ ≥ 0.15, CI LB > 0.05, robust ≥ 3/4).
- **`ADAPTIVE_SELECTION_T1R`** — MAX_CONFIDENCE vs strongest carried fixed control (slot153):
  **+0.313** CI[0.240,0.375], robust 3/4 → **QUALIFIED** (Δ ≥ 0.05, CI LB > 0, robust ≥ 3/4).
  Selection histogram over `[slot38,slot76,slot115,slot153]` = `[56,50,48,38]` — the adaptive selector
  genuinely spreads across all four snapshots (does not collapse to slot153). Every fixed control has
  a full recovery/harm table (`recovery_harm_per_fixed_control`), including negatives (slot38 acc
  0.219). POST_HOC "MAX_CONF − best fixed on qual" is labeled descriptive, not confirmatory.

This is a materially stronger adaptive result than the RNN-06T narrow-band outcome: in the wide
forgetting regime no single fixed snapshot suffices (early targets forgotten by slot153, late targets
not yet written at slot38), so per-example confidence selection wins decisively over the strongest
pre-committed fixed control.

## Recovery/harm accounting (Section 11)

Full tables persisted for FINAL, all four fixed controls, and MAX_CONFIDENCE with exposed
denominators. Per-example correctness arrays in `T1R_WIDE_READOUTS.npz`; scored-value token map in
`T1R_SCORED_VALUE_TOKEN_MAP.json` (gold-column ↔ token id).

## Mechanism activation (Section 12)

`singlePassRuns=12`, `snapshotsCapturedInRun=960`, `snapshotsRestored=960`,
`candidateSnapshotsScored=768`, `queriesEvaluated=960`, `snapshotBoundaryChecks=5`,
`snapshotBoundaryFailures=0` (actual per-boundary captured-vs-replay hash assertions on a held-out
batch), `fastPathPrefillCalls=624`, `fastPathStepCalls=482,496`, `fallbackPathCalls=0`,
`historicalSelections=192`, `fixedSelections=192`, `finalSelections=192`. Sample records in
`boundary_check_sample`.

## End-to-end economics (Section 13 — apples-to-apples; envelope frozen before timings)

Every arm executes the **same** semantic task (context + target query + constrained scored answer),
fixing the RNN-06T defect where FINAL_fused skipped the query. 2 process starts × 40 warm iters = 80
warm samples/arm (`T1R_ECONOMICS.json`).

| arm (warm, per-query) | median | p95 |
|---|---|---|
| FINAL_FUSED_EQUIVALENT_WORK | 37.7 ms | — |
| FINAL_STEP_EQUIVALENT_WORK | 976.7 ms | — |
| RECOVERY_ENABLED_EQUIVALENT_WORK | 1010.2 ms | — |

- **Primary comparator** `RECOVERY_ENABLED − FINAL_STEP` (marginal cost of enabling recovery on the
  capture-capable step path): median **+41.3 ms**, **p95 +192.7 ms ≤ 250 ms envelope**.
- Component breakdown: trajectory+capture 889 ms/q (shared with FINAL_STEP), restore+readout
  12.2 ms/q, GPU→CPU snapshot copy 8.4 ms/q, selection 0.007 ms/q — the true recovery machinery is
  ~20 ms/q; the 889 ms is the step path you run anyway.
- Descriptive `RECOVERY_ENABLED − FINAL_FUSED` = 972 ms/q — the orthogonal cost of choosing a
  capture-capable path at all (the fused prefill kernel cannot expose mid-sequence states), reported
  descriptively, **not** the recovery premium.
- Quality gate: wide MAX_CONF−FINAL Δ 0.542 ≥ 0.05 and net_recovery 104 > 0. Cost gate: p95 192.7 ≤
  250. → `END_TO_END_RECOVERY_UTILITY_T1R = QUALIFIED`. VRAM peak: fused 4.02 GB, step 4.40 GB,
  recovery 7.73 GB; snapshot bytes (K+1)×batch = 4.16 GB CPU-serializable.

## Cross-process numerical note (negative evidence)

In-process same-path replay is BIT_EXACT (T0R test H, 0 boundary failures). *Across* separate process
starts, bf16 kernel-autotuning selects marginally different configs, flipping the argmax of ~1–3
borderline examples out of 192 (e.g. wide FINAL 0.266↔0.271, adaptive Δ +0.323↔+0.313 across two
runs). All qualification margins (recovery Δ 0.542, adaptive Δ ~0.31) are an order of magnitude larger
than this ~0.5–1.5% noise, so every verdict is stable. This is disclosed as a boundary condition, not
a defect; no run was screened or selected — both process starts yielded identical verdicts and this
document reports the committed re-run with complete mechanism counters.

## Authority

Prospective confirmation on the requalified fixed-batch lifecycle contract. Conditional on this exact
frozen checkpoint. Nothing pushed.
