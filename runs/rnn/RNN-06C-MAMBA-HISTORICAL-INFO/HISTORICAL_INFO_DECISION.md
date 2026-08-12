# RNN-06C-MAMBA-HISTORICAL-INFO — DECISION

## Verdict

**`HISTORICAL_STATE_INFORMATION = QUALIFIED`** (reason OK).

Executed because the upstream gate opened: RNN-06B3 `STATE_LOAD_FORGETTING_PERTURBATION =
QUALIFIED`. Subject `AntonV/mamba2-1.3b-hf` @ `703e19a4`, transformers-native naive bf16
`torch_forward`, `chunk_size=32`. Executed-source PROVEN (runner blob `3fdf8fd` == committed,
dirty = ∅; HEAD `0a7da96`; `is_fast_path_available=False`). `historicalInfoSetSha256` re-verified;
disjoint from all six prior sets. N=192, S=3 strata, chance 1/256. Runtime 435 s, peak VRAM
8.3 GB. Scoped to this exact checkpoint/backend/config. **No recovery, no reader, no Memory
Caching.**

> This run was first interrupted by a Windows Update auto-restart (2026-08-12 01:29, Event 1074
> "Service pack (Planned)" — NOT a power blackout; see `ops/` note) and re-executed
> deterministically to completion. Results are from the completed run.

## Machinery validity (the state paths actually executed)

- **Boundary self-check (before outcomes): 8/8 pass** — re-prefilling each recorded prefix
  reproduced the recorded conv/ssm state hashes, and H equals the post-target state of N and L
  (branch-from-same-H). `failures = 0`.
- **Mechanism counters:** `snapshotsCreated=576, snapshotsHashed=576, snapshotsRestored=576,
  historicalDirectReadouts=192, neutralAgedReadouts=192, highLoadReadouts=192,
  branchPairsCompleted=192, snapshotBoundaryChecks=576, snapshotBoundaryFailures=0,
  queriesEvaluated=576`. All required paths ran; no zero counts.
- Each snapshot carries full temporal identity (`prefixTokenSha256`, `convStateSha256`,
  `ssmStateSha256`, `combinedStateSha256`, `cachePosition == len(prefix)`, role, model/backend
  identity). HARD boundary invariant held (`snapshotBoundaryFailures=0`).

## Three state conditions (from the SAME target-slot-0 prefix; identical query, model, backend)

| condition | continuation | query position | accuracy (k/n) | 95% CI |
|---|---|---:|---:|---|
| **H** historical-direct | none (right after target write) | 4 | **0.849 (163/192)** | [0.791, 0.893] |
| **N** same-aged neutral | 764 sentinel tokens (low load) | 768 | **1.000 (192/192)** | [0.980, 1.000] |
| **L** high-load final | 764 tokens at U=HIGH=152 (order-stable) | 768 | **0.547 (105/192)** | [0.476, 0.616] |

N and L are matched on continuation length, final query position, query tokens, model/backend,
and initial state H — they differ ONLY in body content (neutral sentinel vs unique high load).

## Primary causal endpoint & transitions (§18)

- **`neutral_minus_load` (N − L) = 0.453**, stratified-bootstrap 95% CI **[0.385, 0.526]** —
  far above SESOI 0.15 and the trivial region; robust in **3/3** strata (0.500 / 0.453 / 0.406).
- `historical_minus_load` (H − L) = 0.302.
- **Paired transitions:** `N_correct→L_wrong = 87`, `N_wrong→L_correct = 0`;
  `H_correct→L_wrong = 75`, `H_wrong→L_correct = 17`. The high-load continuation moves 87
  examples from correct (neutral) to wrong (loaded) and **zero** in the reverse direction —
  overwhelmingly directional.
- `L reproduces the B3 degradation`: L = 0.547 vs B3 U=152 DS = 0.568 (within the ±0.10 band).

## Preregistered gate (§19) — QUALIFIED

All satisfied: (a) H competent (0.849 ≥ 0.75) — target info accessible near the write; (b) N
retains materially more than L (N−L 0.453 ≥ SESOI 0.15) with CI lower bound 0.385 > 0.05; (c) L
reproduces the B3 degradation; (d) robust across 3/3 strata; (e) all snapshot temporal-identity
checks pass; (f) all mechanism counters > 0. ⇒ `QUALIFIED`.

## Scientific interpretation (presence, not recovery)

From an IDENTICAL earlier recurrent state (H), aging the sequence with neutral low-information
content preserves the target essentially perfectly (N = 1.000), whereas aging with the qualified
unique-load perturbation destroys behavioral access to the target (L = 0.547). Because N and L
are matched on elapsed length, final position, gap, query, and initial state, the loss in L is
attributable to the **load-induced state transformation**, not to elapsed sequence/position.
Therefore target information that becomes behaviorally UNAVAILABLE after the load perturbation
(L) **was demonstrably present earlier and remains functionally accessible from the neutral-aged
state** (H competent; N near-perfect) — `HISTORICAL_STATE_INFORMATION = QUALIFIED`.

This is INFORMATION PRESENCE only. It does NOT demonstrate a *recovery* mechanism, a trained
reader, or utility — those are explicitly out of scope and were not built.

## Consequence (train stop/pivot policy §23, outcome C)

B3 QUALIFIED + 06C QUALIFIED ⇒ STOP. One next recommendation (NOT executed): **OPEN RNN-06D
HISTORICAL-STATE RECOVERY/UTILITY TRAIN** — to test whether the demonstrated presence can be
*exploited* (a recovery/read contract), under independent audit and fresh preregistration. Do NOT
implement RNN-06D, recovery, Memory Caching, GDN repair, or Qwen here.
