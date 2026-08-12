# RNN-06B3-MAMBA-CONTROLLED-STATE-LOAD-PERTURBATION — PRE-REGISTRATION

**Written and committed BEFORE any B3 qualification outcome.** Construction frozen by the bounded
exploratory calibration (`B3_CALIBRATION_DECISION.md`). This is a NEW contract; it does NOT
modify or reclassify RNN-06B2. Mints `STATE_LOAD_FORGETTING_PERTURBATION ∈ {QUALIFIED | BLOCKED}`
plus descriptive `TRANSITION_SHAPE ∈ {GRADED | CLIFF | MIXED | FLAT}`.

## 1. Causal question

With total token length AND target→query gap held EXACTLY constant, AND with the temporal-order
churn eliminated (permanent ordinal↦slot↦binding, nested-identity invariant) AND the full-packing
boundary excluded (≥ MIN_SENTINEL_RESERVE sentinels, never U=M), does increasing the number of
UNIQUE active bindings cause a reproducible material retrieval-loss perturbation?

## 2. Frozen subject & backend

`AntonV/mamba2-1.3b-hf` @ `703e19a43f397c70315244a3424d79456b54fb34`; transformers-native
`Mamba2ForCausalLM`; transformers 4.48.3; torch 2.6.0+cu124; bf16; no quantization; no
mamba_ssm/causal_conv1d (`is_fast_path_available=False`); **`chunk_size=32`**. Frozen across the
sweep; no per-condition change; no quantization axis.

## 3. Frozen construction (order-stable; `rnn_06b3_lib.py`)

- **M = 192** slots ⇒ every prompt is exactly **4·192+2 = 770 tokens** for EVERY dose/arm.
  Target at **slot 0**; query `target_key =` at end; **target→query gap = 191 slots (constant)**.
- **MIN_SENTINEL_RESERVE = 16** ⇒ qualification doses satisfy `U ≤ M−16 = 176`; every dose
  retains ≥16 REPEAT1 sentinels. **No U=M cell.**
- **Order-stable load:** load ordinal `i` ↦ fixed slot `load_positions[i]` ↦ fixed DS binding `i`
  (filler space) ↦ fixed SS binding `i` (scored space). At dose U only ordinals `0..U-2` are
  active. Increasing U activates exactly one new ordinal at its fixed slot/binding; all
  already-active bindings keep identity AND position (nested-identity invariant, asserted).
- **Arms:** DS (PRIMARY, disjoint-space load). SS (secondary diagnostic, non-gating).
- Constrained scoring over the 256-token scored value vocabulary (chance 1/256).

## 4. Stress grid (frozen) — `b3StressGridSha256`

- **Doses U = [1, 24, 48, 72, 96, 128, 152, 176]** (competence anchors + interior + max
  subpacked 176). `U_low = 1`, `U_high = 176` (both DS and SS).
- **Strata:** S=3 seed strata × n=64 ⇒ **N=192 per (dose, arm)**. Same target example evaluated
  across all doses (paired/nested).

## 5. Primary endpoint, SESOI & practical-effect threshold (fixed BEFORE outcomes)

- **PRIMARY endpoint:** the **paired** DS constrained-retrieval accuracy loss from the competence
  dose `U_low=1` to the max subpacked dose `U_high=176`, i.e. `Δ = acc(U1) − acc(U176)`, computed
  paired per example.
- **SESOI = 0.20** absolute paired accuracy loss. **Justification (from 06C need, NOT from B2
  fitting):** RNN-06C's primary contrast is neutral-aged (N) vs high-load (L) from the same
  historical state; for that paired contrast to have power, the high-load condition must flip a
  substantial fraction of examples correct→wrong. A ≥0.20 paired loss guarantees ≥~20% discordant
  (correct→wrong) pairs at the population level, the minimum for a usable N−L test. **Trivial-
  effect region = [−0.05, +0.05].**
- **Effect reported with interval:** paired-loss point estimate + stratified cluster-bootstrap
  95% CI. Lack of significance is NOT equated with equivalence.

## 6. B3 causal gate — `STATE_LOAD_FORGETTING_PERTURBATION`

`QUALIFIED` requires ALL (primary arm DS unless noted), prospectively:
1. **Low-load competence** — `acc(U_low=1) ≥ 0.75`.
2. **Material paired degradation** at ≥1 higher subpacked load — primary: `Δ = acc(U1)−acc(U176)
   ≥ SESOI (0.20)`.
3. **Fixed length** — all DS doses identical token length (770); asserted (else BLOCKED
   `LENGTH_NOT_FIXED`).
4. **Fixed gap** — target→query gap constant (191 slots); asserted (else BLOCKED `GAP_NOT_FIXED`).
5. **Nested binding identity invariant PASS** — for every example and adjacent dose pair
   (`nestedBindingIdentityFailures == 0`); else BLOCKED `NESTED_IDENTITY_FAIL`.
6. **Positive sentinel reserve** at every dose (`min sentinel_slots ≥ 16`); else BLOCKED
   `PACKING_BOUNDARY`.
7. **Frozen substrate/source** — one checkpoint/backend/dtype/chunk_size/generator.
8. **Robustness across strata** — paired loss `Δ_s ≥ SESOI` in `≥ 2/3` seed strata.
9. **Paired uncertainty excludes the trivial region** — paired-loss 95% CI lower bound `> 0.05`.

Failure of any ⇒ `BLOCKED` with the specific reason. Smooth gradedness is NOT required.

### TRANSITION_SHAPE (descriptive; categories defined BEFORE outcomes)

Let `acc_low = acc(U1)`, `min_acc = min_U acc(U)`, `total_loss = acc_low − min_acc`; interior
doses = # doses with `min_acc+0.10 < acc(U) < acc_low−0.10`; `max_step` = largest single
adjacent-dose accuracy drop.
- **FLAT** — `total_loss < SESOI`.
- **CLIFF** — `total_loss ≥ SESOI` AND `max_step ≥ 0.60·total_loss` AND `interior < 2`.
- **GRADED** — `total_loss ≥ SESOI` AND `interior ≥ 2` AND `max_step < 0.60·total_loss`.
- **MIXED** — `total_loss ≥ SESOI` but neither CLIFF nor GRADED.
A reproducible subpacked CLIFF still QUALIFIES if the §6 controls pass; shape is descriptive only.

## 7. Curve statistic cleanup (§8)

Two descriptive statistics reported (neither is the primary gate):
- **`MEAN_RELATIVE_RETENTION_DEFICIT`** = `mean_U max(0, (acc(U1)−acc(U))/acc(U1))` — the honest
  name for the quantity RNN-B2 mislabeled "delta-AURC" (history NOT rewritten).
- **`DEFICIT_AURC_NORMALIZED`** = normalized trapezoidal integral of `deficit(U)=1−acc(U)/acc(U1)`
  over the dose axis `U∈[1,176]`, divided by `(176−1)` (trapezoidal rule over the actual grid);
  ∈ [0,1]. Normalization + integration defined here, before outcomes.

## 8. Statistical hierarchy (§9)

- **Inference population:** conditional on THIS exact frozen checkpoint/backend. Example strata
  are NOT independent training seeds; thousands of observations are NOT thousands of model
  replications.
- **Primary analysis:** paired example-level (same target example across doses). Discordant-pair
  counts reported (U1-correct/U176-wrong and inverse).
- **Uncertainty:** stratified (by seed stratum) cluster bootstrap over examples, 2000×.
- Strata preserved as a robustness/generalization check (§6.8), not as independent replications.

## 9. Construction-activation evidence (§10) — persisted counters/invariants

`examplesEvaluated, cellsEvaluated, uniqueBindingsMaterializedByDose, sentinelSlotsByDose,
nestedBindingIdentityChecks, nestedBindingIdentityFailures, fixedLengthChecks, fixedGapChecks,
DSCells, SSCells`. The machine artifact must prove the intended construction actually fired.

## 10. Executed-source identity (§21) — before outcomes

Runner + lib SHA-256 + git blobs + HEAD + dirty; protocol hash; challenge-set hash;
model/revision; backend source hashes; chunk_size; dtype; versions;
`b3QualificationSetSha256`; `b3StressGridSha256`. Assert `is_fast_path_available is False`.

## 11. Challenge identities & disjointness (§4)

`ops/rnn_06b3_challenges.py` (`generator_version = rnn06b3_order_stable_state_load_v1`,
`master_seed = 20260817`) → `B3_QUALIFICATION_SPEC.json` (`b3QualificationSetSha256`);
`B3_STRESS_GRID.json` (`b3StressGridSha256`). Disjoint from P0 calib (`779fb37a`), RNN-06B qual
(`e351a444`), B2 qual (`a92870a9`), B2 calib (`727c5367`), B3 calib (`342f0961`) — distinct
seeds/generator + example-level (0 overlap vs B3 calib) + distinct SHAs. **No seed screening.**

## 12. Frozen B3 → 06C dose-selection RULE (frozen NOW, before B3 outcomes; §14)

If B3 QUALIFIES, 06C load levels derive deterministically from the DS curve + paired losses:
- **HIGH** = the qualified grid dose with the **maximum paired loss** vs U1 (downstream-justified:
  maximizes 06C's N−L power). 06C's L (high-load) branch.
- **LOW** = the highest grid dose with `acc ≥ 0.75` (most-loaded still-competent anchor).
- **MID** = the smallest grid dose with paired loss `≥ SESOI` (transition point).
Actual U values become known only after B3; the RULE is frozen here.

## 13. B3 dependency gate

BLOCKED ⇒ `RNN-06C = BLOCKED_BY_06B3`; no historical-state readout; package & STOP. **No
RNN-06B4, no further tuning iteration.** QUALIFIED ⇒ persist qualified dose region + identities,
proceed to 06C.

## 14. Invariants

No GDN/Qwen/Memory Caching/recovery/reader/RNN-06D; no seed screening; no threshold change after
outcomes; frozen model; nothing pushed. Conclusions scoped to this exact checkpoint/backend/config.
