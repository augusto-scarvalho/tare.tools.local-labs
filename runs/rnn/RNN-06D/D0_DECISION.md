# RNN-06D0 — DECISION

## Verdict

**`RECOVERY_CEILING = QUALIFIED`** (reason OK). `RNN-06D1 = OPEN`.

Subject `AntonV/mamba2-1.3b-hf` @ `703e19a4`, transformers-native naive bf16 `torch_forward`,
`chunk_size=32`, `is_fast_path_available=False`. Executed-source PROVEN: runner blob `96d125cb…` ==
committed, dirty ∅; lib dirty ∅; HEAD `eac2a1e` at run. `qualificationSetSha256` /
`snapshotScheduleSha256` re-verified in-runner (match). N=192 (3 strata × 64), chance 1/256. Runtime
665 s, peak VRAM 8.4 GB. State = 52,002,816 bytes/seq (remeasured), × K=4 = 208,011,264 bytes.

## Machinery validity (the state paths actually executed)

- **Boundary self-check (before outcomes): 16/16 pass** (8 examples × {slot 38, slot 191}) —
  re-prefilling each prefix reproduced the recorded conv/ssm state hashes. `failures = 0`.
- **Counters:** `snapshotsCreated=960, snapshotsRestored=960, poolReadouts=768, finalReadouts=192,
  snapshotBoundaryChecks=960, snapshotBoundaryFailures=0, queriesEvaluated=960`. All required paths
  ran; every snapshot carries temporal identity (`prefixTokenSha256`, `convStateSha256`,
  `ssmStateSha256`, `combinedStateSha256`, `cachePosition == len(prefix)`).

## Arms (fixed 770-token sequences; target-agnostic K=4 schedule [38,76,115,153])

| arm | accuracy | note |
|---|---:|---|
| FINAL (full state) | **0.130** (25/192) | degraded forgetting regime (≤ 0.75) |
| FIXED_HISTORICAL_POOL per-slot | 38:0.490 · 76:0.771 · 115:0.458 · 153:0.286 | substrate for D1 |
| ORACLE_TARGET_PROXIMAL (diagnostic) | **0.901** (173/192) | first pool snapshot ≥ t; post-target frac 1.0 |
| ORACLE_BEST_GOLD (upper bound) | **0.906** (174/192) | any pool snapshot correct vs gold |

Pool per-slot pattern confirms the construction: slot 76 (post-target for the whole band, moderate
load) is best (0.771); slot 38 is a **genuine pre-target distractor** for t∈(38,64] (drags it to
0.490); slots 115/153 carry progressively more post-target load (0.458 → 0.286). FINAL (191, all
post-target load) collapses to 0.130.

## Recovery ceiling (paired, gold-aware)

- **ORACLE_BEST − FINAL = 0.776**, stratified-bootstrap 95% CI **[0.719, 0.833]** (lb > 0.05).
- **ORACLE_PROXIMAL − FINAL = 0.771**. Robust across **3/3** strata (per-stratum OB−FINAL all
  ≥ CEILING_SESOI).
- **Recoverable substrate:** among 167 FINAL-wrong examples, **149 (0.89)** are recoverable by some
  pool snapshot (≥ RECOV_FRAC_MIN 0.30 and ≫ RECOV_N_MIN 20). Ample substrate for D1.
- **Coverage:** 88.3% of (example, snapshot) pairs are post-target; 174/192 examples have a correct
  post-target snapshot; mean top-1 prob post-target 0.222 vs pre-target 0.163 (a modest confidence
  separation — the signal D1 parameter-free methods must exploit).

## Preregistered gate — QUALIFIED

All satisfied: boundary_ok ✓, counters_ok ✓, final_degraded (0.130 ≤ 0.75) ✓, proximal_competent
(0.901 ≥ 0.75) ✓, effect_ge_sesoi (0.776 ≥ 0.15) ✓, ci_lb_gt (0.719 > 0.05) ✓, robust (3/3 ≥ 2) ✓,
recoverable (0.89 ≥ 0.30 and 149 ≥ 20) ✓ ⇒ `QUALIFIED`.

## Scientific interpretation (ceiling, not a mechanism)

The target-agnostic historical snapshot pool demonstrably CONTAINS the target information that FINAL
has lost: an oracle that may consult the pool recovers 89% of FINAL's failures, and even the
target-position-proximal snapshot alone reaches 0.901 vs FINAL 0.130. This justifies building and
testing a recovery mechanism (D1). It is a **ceiling / diagnostic** result: ORACLE_BEST and
ORACLE_PROXIMAL both use privileged information (gold / target position) and are NOT deployable. The
open question D1 answers is whether a *target-agnostic, parameter-free* method can capture a useful
fraction of this ceiling. No recovery mechanism, reader, or Memory Caching was built here.

## Cost profile (per batch of 2; warm)

capture (re-prefill FINAL, 768 tok) 1.708 s · GPU→host transfer 0.031 s · restore 0.0004 s ·
restore+readout 0.073 s. Intrinsic per-query restore+readout ≈ 36 ms. The re-prefill capture is a
**naive-backend artifact** (no multi-token mid-sequence forward on this substrate); a fast-path /
incremental capture folds capture into the single forward pass. Reported transparently and carried to
D1's economics.

## Consequence (train policy)

`RECOVERY_CEILING = QUALIFIED` ⇒ proceed to **RNN-06D1** (Target-Agnostic Parameter-Free Recovery
Utility) under a separate pre-registration. No RNN-06B3/06C artifact modified; no reader trained; no
Memory Caching; no GDN/Qwen; nothing pushed.
