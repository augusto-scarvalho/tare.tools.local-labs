# RNN-06B-MAMBA-BASE — DECISION

## Verdict

**`FIXED_BACKBONE_GRADED_REGION = BLOCKED`** — sole reason **`CONFOUNDED_WITH_LENGTH`**.

Executed because the upstream gate opened (`CONTINUATION_LIFECYCLE = QUALIFIED`, RNN-06A2,
re-qualified at the operative `chunk_size=32`). This is a fully-executed 06B with a BLOCKED
*graded-region* verdict — it is NOT `BLOCKED_BY_06A2`. Subject: `AntonV/mamba2-1.3b-hf`
@ `703e19a4`, transformers-native naive bf16 `torch_forward`, **pinned `chunk_size=32`**
(train AMENDMENT 1). Scoped to this exact checkpoint/backend/config; not generalized to
"Mamba-2".

## Executed-source identity (PROVEN)

Runner git blob `ebe6bb965ed4b01fb1aea58d7010fa946f18ab8d` == committed; dirty = ∅; HEAD
`fd8f863`; `is_fast_path_available=False`; `effective_chunk_size=[32,32]`;
`qualificationSetSha256=e351a444…` and `stressGridSha256=d29e442f…` re-verified;
example-level disjoint from P0 calibration (0 overlap). N=192 per (dose,condition), S=3 seed
strata. Runtime 756 s, peak VRAM 17.9 GB.

## Primary curve (MP constrained retrieval) + matched control (LC), with bootstrap CIs

| dose P | MP acc | LC acc | LC−MP | MP boot-95% | seq_len |
|---:|---:|---:|---:|---|---:|
| 8   | 0.828 | 0.880 | +0.052 | [0.771, 0.880] | 34 |
| 16  | 0.682 | 0.703 | +0.021 | [0.615, 0.750] | 66 |
| 24  | 0.620 | 0.589 | −0.031 | [0.552, 0.688] | 98 |
| 32  | 0.510 | 0.479 | −0.031 | [0.438, 0.578] | 130 |
| 48  | 0.375 | 0.380 | +0.005 | [0.302, 0.443] | 194 |
| 64  | 0.297 | 0.271 | −0.026 | [0.229, 0.365] | 258 |
| 96  | 0.208 | 0.203 | −0.005 | [0.151, 0.266] | 386 |
| 128 | 0.130 | 0.130 | +0.000 | [0.083, 0.177] | 514 |

Chance = 1/256 ≈ 0.0039. Raw denominators in `BASE_CURVES.csv` / `BASE_RESULTS.json`.

## Preregistered gate evaluation (PRE_REGISTRATION §7)

| Criterion | Threshold | Observed | Pass |
|---|---|---|:--:|
| 1 Competence | MP@P8 ≥ 0.75 | 0.828 | ✅ |
| 2 Material loss | min MP ≤ 0.45 | 0.130 @ P128 | ✅ |
| 3 Interior resolution | ≥ 2 mid-band doses | 3 → [16, 24, 32] | ✅ |
| 4 Monotonicity | ≤ 1 viol, none > 0.10 | 0 violations | ✅ |
| 5 Robustness | graded in ≥ 2 / 3 strata | 3 / 3 | ✅ |
| **6 Confound-controlled** | **mean(LC−MP)@{96,128} ≥ 0.15** | **−0.0026** | ❌ |

Five of six criteria pass — the curve *is* competent, graded, monotone, and robust. The gate
is BLOCKED solely by the **confound control**.

## What the matched control shows (the scientific finding)

MP (all P associations in the scored key/value space) and LC (only the ONE probed association
in the scored space; the other P−1 are structurally-identical filler pairs from disjoint
pools, at identical sequence length, probe position, write→query gap, and total binding count)
**degrade in lockstep**: `mean(LC−MP)` over the whole ladder ≈ 0, and at the two highest doses
it is −0.0026. Removing same-space competition (LC) does **not** rescue accuracy.

Therefore the graded degradation is **not** attributable to same-space associative retrieval
interference; it is explained by the factors LC holds equal to MP — **generic sequence-length
/ recurrent-state-saturation load** (more tokens to integrate, longer write→query gap, more
total bindings to store), independent of whether the distractors compete in the probed space.
This is precisely the confound RNN-06B was designed to detect (PRE_REGISTRATION §4/§15). A
region whose loss is length/capacity-driven rather than interference-driven is **not** a clean
BASE retrieval-loss region suitable for testing historical-state information, so it must not be
minted `QUALIFIED`.

**Contrast with P0 (calibration, quarantined).** Exploratory P0 saw the same-shaped MP curve
(at `chunk_size=32`) and labelled it `P0_GRADED_BAND = PLAUSIBLE`. P0 had **no** matched
control and conflated #associations, length, and gap. 06B's control falsifies the memory
interpretation of that band. P0 remains exploratory; its verdict is unchanged.

## Secondary channels (reported, not primary)

Unconstrained-exact and format-adherence are far lower than constrained retrieval at every
dose (e.g. P8: constrained 0.828 vs unconstrained 0.490, format 0.505), confirming the base LM
frequently does not emit the value token as the *global* argmax under the `k=v` convention.
This validates choosing **constrained retrieval** as the primary endpoint (§16): it isolates
retained answer *preference* from response-format failure. Both secondary channels also
decline with dose (full table in results).

## State-economics carry-forward (PRE_REGISTRATION §11)

`state_bytes_per_sequence = 52,002,816` (≈ 49.59 MiB bf16), carried from RNN-06A/06A2. A rough
upper-bound derived estimate for a hypothetical 06C snapshotting every qualification example
across a region is recorded in `BASE_RESULTS.json → state_economics` for the next audit. **No
snapshot/reader/recovery machinery was built.** Because 06B is BLOCKED, this estimate is
informational only.

## Consequence

Per train §22: BLOCKED ⇒ **STOP scientific progression; do NOT execute historical-state
experiments; do NOT start RNN-06C**. No qualified pressure region is recorded (the confound
control was not satisfied). No `FIXED_BACKBONE_GRADED_REGION = QUALIFIED` is minted anywhere in
this train. Frozen-model invariant held across the entire sweep (one checkpoint/backend/
dtype/chunk_size/task-generator; no per-condition training or model change; no quantization).
