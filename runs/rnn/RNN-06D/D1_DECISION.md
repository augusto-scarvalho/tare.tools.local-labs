# RNN-06D1 — DECISION

## Verdict

**`RECOVERY_UTILITY = QUALIFIED_PARAMETER_FREE`** (reason OK). Best method: **MAX_CONFIDENCE**.

Executed because `RECOVERY_CEILING = QUALIFIED` (D0, OB−FINAL = 0.776). Subject
`AntonV/mamba2-1.3b-hf` @ `703e19a4`, transformers-native naive bf16 `torch_forward`,
`chunk_size=32`, `is_fast_path_available=False`. Executed-source PROVEN: runner blob `2a7c47e0…` ==
committed, dirty ∅. Same frozen qualification set (`7bbf9b75…`) + K=4 schedule (`355e5105…`) as D0.
No trained reader, no DART, no Memory Caching. No method used the gold answer, the actual
target-write position, or the oracle-best identity.

## Independent re-capture is bit-reproducible (mechanism genuinely exercised)

D1 re-captured the pool + FINAL from scratch: `snapshotsCreated=960, snapshotsRestored=960,
candidateSnapshotsScored=768, historicalSelections=960, ensembleCalls=384, snapshotBoundaryChecks=960,
snapshotBoundaryFailures=0, queriesEvaluated=960`; boundary self-check 8/8. Cross-check vs
`D0_READOUTS.npz`: `pool_logits` and `final_logits` max-abs-diff **0.0** ⇒ **bit-reproducible**. The
methods actually restore and score historical states.

## Results (N=192; FINAL acc 0.130; ORACLE_BEST 0.906; target-agnostic K=4 pool)

| method | acc | Δ vs FINAL | Δ CI95 | recovered | harmed | net | oracle gap | robust |
|---|---:|---:|---|---:|---:|---:|---:|:--:|
| **MAX_CONFIDENCE** | **0.833** | **+0.703** | [0.641, 0.766] | 135 | **0** | 135 | 0.073 | 3/3 |
| MAX_TOP1_TOP2_MARGIN | 0.818 | +0.688 | [0.625, 0.750] | 132 | 0 | 132 | 0.088 | 3/3 |
| CONFIDENCE_X_RECENCY | 0.802 | +0.672 | [0.609, 0.734] | 129 | 0 | 129 | 0.104 | 3/3 |
| MIN_ENTROPY | 0.786 | +0.656 | [0.594, 0.719] | 126 | 0 | 126 | 0.120 | 3/3 |
| LOGIT_ENSEMBLE | 0.510 | +0.380 | [0.307, 0.453] | 76 | 3 | 73 | 0.396 | 3/3 |
| FINAL_PLUS_HISTORICAL | 0.474 | +0.344 | [0.276, 0.411] | 69 | 3 | 66 | 0.432 | 3/3 |
| RECENCY | 0.286 | +0.156 | [0.094, 0.214] | 35 | 5 | 30 | 0.620 | 3/3 |
| MATCHED_NO_HISTORY (control) | 0.130 | 0.000 | — | — | — | — | — | — |

## Controls that make the result interpretable

- **MATCHED_NO_HISTORY = FINAL (0.130 exactly).** Ensembling K FINAL readouts (same compute as
  LOGIT_ENSEMBLE, no history) gives no gain ⇒ the benefit is from *history*, not compute.
- **RECENCY is the worst method (0.286).** Always selecting the most-recent snapshot (most
  post-target load) barely helps ⇒ the benefit is from *selecting the right* snapshot, not merely
  from consulting a historical one.
- **LOGIT_ENSEMBLE / FINAL_PLUS_HISTORICAL underperform selection (0.47–0.51).** Averaging over the
  whole pool dilutes the good snapshot with pre-target/late-load garbage ⇒ discrimination beats
  blending here.
- **Confidence is the discriminator.** MAX_CONFIDENCE selection histogram = [slot38:115, slot76:75,
  slot115:1, slot153:1]: it routes to slot 38 when that is the fresh post-target snapshot (t≤38) and
  to slot 76 when slot 38 is a pre-target distractor (t>38), consistent with the D0 finding that
  post-target snapshots are more confident (0.222 vs 0.163). Selection regret vs ORACLE_BEST = 0.081.

## Preregistered utility gate — QUALIFIED_PARAMETER_FREE

Best arm MAX_CONFIDENCE: accuracy_delta 0.703 ≥ UTILITY_SESOI 0.05 ✓; net_recovery_count 135 > 0 ✓;
Δ CI lower bound 0.641 > 0 ✓; robust (Δ ≥ 0) 3/3 ≥ 2 ✓; recovery (135) ≫ harm (0) ✓. **COST_OK:**
intrinsic per-query restore+readout ≈ 36.4 ms ≤ 100 ms ✓; intrinsic memory K×52,002,816 =
208,011,264 bytes ≤ bound ✓. ⇒ `QUALIFIED_PARAMETER_FREE`.

## Economics (per batch of 2; warm) and the honest cost caveat

restore 0.4 ms · restore+readout 36 ms/query · GPU→host transfer 31 ms/batch · peak VRAM 8.2 GB ·
state 52,002,816 B/seq × K=4 = 208,011,264 B. **Capture cost caveat:** producing the pool costs K
extra prefills (~1.7 s/batch at FINAL length) on this **naive backend** — a substrate artifact, not a
property of the method: a fast-path / incremental capture folds capture into the single forward pass.
The intrinsic recovery mechanism (restore + score + select over an already-captured pool) is cheap.

## Scientific interpretation

On this frozen Mamba-2 substrate a **target-agnostic, parameter-free** confidence selector over a
fixed K=4 historical snapshot schedule recovers the large majority of the model's forgetting failures
(0.130 → 0.833, capturing 0.703 of the 0.776 oracle ceiling; 135/167 FINAL-wrong recovered, zero
harm), without a trained reader, without the gold answer, and without the target-write position. The
demonstrated historical-state *presence* (RNN-06C) is therefore not merely present but **exploitable**
by a trivial mechanism. Scope: this exact checkpoint/backend/config and this synthetic MQAR
forgetting construction; transportability to the official Mamba kernel and to non-synthetic tasks is
untested.

## Consequence (train policy — outcome QUALIFIED_PARAMETER_FREE)

One next recommendation (NOT executed): **OPEN an official-Mamba transportability replication in a NEW
session** — reproduce the parameter-free recovery utility on the fast-path (`mamba_ssm`) kernel and,
if it holds, on a non-synthetic long-context retrieval task, under independent audit and fresh
pre-registration. Do NOT implement it here; no reader/DART/Memory Caching/StateX/GDN/Qwen; nothing
pushed.
