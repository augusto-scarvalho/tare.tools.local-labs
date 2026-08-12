# RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD — PRE-REGISTRATION

**Written and committed BEFORE any B2 qualification outcome.** Construction was frozen by the
bounded exploratory calibration (`B2_CALIBRATION_DECISION.md`). Mints exactly one
`FIXED_LENGTH_STATE_LOAD_REGION ∈ {QUALIFIED | BLOCKED}`.

## 1. Causal question

With **total token length and target→query gap held EXACTLY constant**, does increasing the
number of UNIQUE bindings cause graded retrieval degradation? This isolates general
unique-binding / recurrent-state load from sequence length — the factor RNN-06B left
`NOT_DISAMBIGUATED` (its LC arm removed same-space competition but still grew length AND unique
bindings together).

## 2. Exact frozen subject & backend (identical to the qualified substrate)

`AntonV/mamba2-1.3b-hf` @ `703e19a43f397c70315244a3424d79456b54fb34`; transformers-native
`Mamba2ForCausalLM`; transformers 4.48.3; torch 2.6.0+cu124; bf16; no quantization; no
mamba_ssm/causal_conv1d (`is_fast_path_available=False`); **`chunk_size = 32`**. Frozen across
the entire sweep; no per-condition change; no quantization axis.

## 3. Frozen fixed-length construction (from calibration; see `rnn_06b2_lib.py`)

- **M = 128** association slots ⇒ every prompt is exactly **4·128 + 2 = 514 tokens**, for
  EVERY dose and arm. Target at **slot 0**; query `target_key =` at the end;
  **target→query gap = 127 slots**, constant across all doses.
- Dose **U** = number of unique active bindings including target (1..128). U−1 load bindings
  are placed at the first U−1 of a per-example permutation of slots [1..127]; the remaining
  128−U slots are **REPEAT1 sentinel** (one fixed sentinel pair repeated — low information,
  disjoint pool).
- **Arms:**
  - **DS (PRIMARY)** — load bindings from a DISJOINT filler key/value space ⇒ general
    unique-binding/state load WITHOUT same-scored-space competition.
  - **SS (secondary)** — load bindings from the SCORED space ⇒ general load + same-space
    competition. Everything else identical (same slots, sentinel count, length, gap, target).
- Pools: 5 mutually-disjoint single-token pools (scored keys/vals, filler keys/vals, sentinel);
  constrained scoring over the 256-token scored value vocabulary (chance = 1/256).

## 4. Stress grid (frozen) — `b2StressGridSha256`

- **Doses U = [1, 24, 48, 64, 80, 96, 112, 128]** (low-load anchor + interior resolution
  across the graded region located in calibration).
- **Arms:** DS, SS. **Strata:** S = 3 seed strata × n = 64 ⇒ **N = 192 per (dose, arm)**.
- **Non-gating length diagnostic (descriptive only):** DS at fixed low load U=2, M ∈
  {32,64,96,128} (lengths 130/258/386/514) — shows length-alone effect at low unique load.

## 5. Primary endpoint & channels (predeclared)

- **PRIMARY: DS constrained retrieval accuracy vs unique-load dose U** (isolates general state
  load). Report separately, never collapsed: **constrained retrieval**, **unconstrained
  exact**, **format adherence**.
- **Secondary mechanistic contrast:** SS vs DS (dose×condition). **We do NOT require SS to
  beat DS.** If SS ≈ DS, that REINFORCES general state load over same-space competition.

## 6. Equivalence thresholds & full-curve statistics (fixed BEFORE outcomes)

`τ_hi = 0.75` (competence / upper band), `τ_lo = 0.45` (material loss / lower band). A dose is
mid-band iff `τ_lo < acc < τ_hi`.

Full-curve statistics (preferred over near-floor endpoints):
- **delta-AURC (primary full-curve stat):** `mean_over_doses( max(0, (acc[U1] − acc[U]) / acc[U1]) )`
  — the average fractional accuracy deficit relative to the U=1 competence level. Requires
  **≥ 0.15** (a materially declining curve, not near-flat).
- **Paired curve contrast (DS vs SS):** reported with a dose×condition summary; descriptive.
- **Cluster bootstrap** (2000×, over examples) CIs on the pooled DS curve and on delta-AURC.

## 7. B2 graded-region gate — `FIXED_LENGTH_STATE_LOAD_REGION`

`QUALIFIED` requires ALL (on the pooled **DS** curve unless noted):
1. **Competence** — DS acc at U=1 `≥ τ_hi (0.75)`.
2. **Material loss** — min DS acc `≤ τ_lo (0.45)`.
3. **Interior resolution** — `≥ 2` mid-band doses.
4. **Bounded monotonicity** — DS acc non-increasing within tolerance `0.05`; `≤ 1` violation,
   none `> 0.10`.
5. **Full-curve effect** — `delta-AURC ≥ 0.15`.
6. **Robustness** — (1) competence and (3) ≥2 mid-band hold in `≥ 2 / 3` seed strata.
7. **EXACT fixed length** — all DS doses have identical token length (514); asserted, else
   BLOCKED `LENGTH_NOT_FIXED`.
8. **EXACT fixed target→query gap** — identical across doses (127 slots); asserted, else BLOCKED
   `GAP_NOT_FIXED`.
9. **Frozen subject/backend/source** — one checkpoint/backend/dtype/chunk_size/generator.

Failure of any ⇒ `BLOCKED` with the specific reason (`TASK_NOT_COMPETENT`, `FLAT_HIGH`,
`INSUFFICIENT_LOSS`, `IMMEDIATE_CLIFF`, `NON_MONOTONE`, `WEAK_FULL_CURVE_EFFECT`,
`NOT_ROBUST_ACROSS_STRATA`, `LENGTH_NOT_FIXED`, `GAP_NOT_FIXED`). Historical RNN-06B remains
BLOCKED under its old contract regardless. **No threshold tuning after results.**

Scientific reading: if the **DS** curve qualifies, that is positive evidence for
`GENERAL_STATE_LOAD_FORGETTING` at fixed length (the mechanism 06B left OPEN); the length
diagnostic contextualizes it.

## 8. Challenge-set identities & disjointness (frozen before outcomes)

- `ops/rnn_06b2_challenges.py` (`generator_version = rnn06b2_fixed_length_state_load_v1`,
  `master_seed = 20260815`) emits the abstract per-example spec → `B2_QUALIFICATION_SPEC.json`;
  `b2QualificationSetSha256` = SHA-256 over it. `B2_STRESS_GRID.json` freezes doses/arms/τ/N →
  `b2StressGridSha256`.
- **Disjoint** from: P0 calibration (`779fb37a…`), RNN-06B qualification (`e351a444…`), B2
  calibration (`727c5367…`) — proven at example (target/load slot-tuple) level + distinct
  master seeds. **No seed screening.**

## 9. Executed-source identity (before outcomes)

Results record runner + lib SHA-256, git blobs, HEAD, dirty, model identity, backend source
hashes, chunk_size, dtype, versions, protocol hash, `b2QualificationSetSha256`,
`b2StressGridSha256`. Assert `is_fast_path_available is False`.

## 10. Predeclared 06C dose-selection RULE (frozen NOW, before B2 outcomes)

If B2 QUALIFIES, 06C conditions are derived deterministically from the pooled **DS** curve:
- **HIGH** = the maximum-load dose (U = 128) — the branch under which forgetting is strongest;
  06C's high-load continuation branch.
- **LOW** = the largest dose with DS acc `≥ 0.80` (competent low-load reference).
- **MID** = the dose whose DS acc is closest to `(acc_LOW + acc_HIGH) / 2`.
The actual U values become known only after B2 outcomes; the RULE is frozen here.

## 11. Invariants

No GDN repair, no Qwen, no Memory Caching, no historical-state reader (that is 06C), no
RNN-06D, no serving change, no training, no seed screening, no push. No
`HISTORICAL_STATE_INFORMATION` mint here. Conclusions scoped to this exact checkpoint/backend/
config. Engineering amendments (if any, before substantive outcomes) appended + committed.
