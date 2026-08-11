# RNN-06B-MAMBA-BASE — PRE-REGISTRATION

**Written and committed BEFORE any outcome-bearing execution.** Executes ONLY because the
upstream gate opened: `CONTINUATION_LIFECYCLE = QUALIFIED` (RNN-06A2). Mints exactly one
`FIXED_BACKBONE_GRADED_REGION ∈ {QUALIFIED | BLOCKED}`. This is the ONLY stage in the train
permitted to mint that verdict. No RNN-06C is started; no historical-state test is run.

## 1. Scientific objective

On the exact frozen qualified Mamba subject, does an **independent deterministic
confirmatory** challenge show a **stable graded retrieval-loss region** suitable for later
testing of historical-state information — i.e. a region where constrained associative recall
degrades gradually with memory pressure, is competent at low pressure, and whose degradation
is attributable to associative interference rather than generic sequence-length/position
decay?

## 2. Exact frozen subject & backend (identical to 06A2; frozen across the entire sweep)

`AntonV/mamba2-1.3b-hf` @ `703e19a43f397c70315244a3424d79456b54fb34`; transformers-native
`Mamba2ForCausalLM`; transformers 4.48.3; torch 2.6.0+cu124; no mamba_ssm/causal_conv1d
(`is_fast_path_available=False`); bf16; no quantization; **pinned `chunk_size=256`** (native
default; the value 06A/06A2 qualified). Same checkpoint, backend, source, dtype, chunk_size,
task generator across ALL conditions. **No per-condition training or model change. No
quantization axis.**

> Note on P0: exploratory RNN-06-P0 ran at a `chunk_size=32` override. Its band
> (P4 0.953 → P128 0.234) is therefore treated as **calibration on a different numerical
> path** — informing the pressure *range* only. 06B re-establishes competence and the band at
> the pinned `chunk_size=256`. P0's examples remain **quarantined** as calibration data.

## 3. Task — single-token MQAR associative recall (exact, low-variance scoring)

Each key and value is exactly ONE token id (pools built per tokenizer), so every prompt at a
given dose has identical length and the answer sits at a known position. A prompt is
`P` blocks of `⟨key⟩=⟨value⟩\n` followed by the query `⟨probe_key⟩=`; the model scores the
next token. **Nested-monotonic** across doses (EXT2/P0 pattern): dose `P` uses the first `P`
pairs; the probed pair lives at a fixed per-example index inside the shared prefix, so its
write position is held constant while trailing pairs (and the write→query gap) grow.

## 4. The matched control — separating memory pressure from length/position (train §15)

Two conditions at **identical sequence length, identical probe position, identical write→query
gap**, differing ONLY in whether the non-probe pairs compete in the scored key/value space:

- **MP (memory-pressure):** all `P` pairs drawn from the **shared scored** key/value pools
  (distinct keys, distinct values). The probe competes against `P−1` same-space associations.
- **LC (length/interference control):** the probed pair (1) drawn from the **shared scored**
  pools and placed at the SAME index; the other `P−1` pairs drawn from **disjoint filler**
  key/value pools (filler keys never equal the probe key; filler values are OUTSIDE the scored
  value vocabulary). Same length/position/gap; only ONE scored-space association; `P−1` filler
  bindings occupy state and length but do not compete in the scored space.

**What each control falsifies.** If MP degrades with pressure while LC stays high at matched
length ⇒ degradation requires same-space associative interference ⇒ genuine memory
retrieval-loss (the confound "it's just longer sequences / farther-back answers / raw
write-capacity" is falsified, since LC has the SAME length, position, gap, and total binding
count). If LC degrades ≈ as much as MP ⇒ the loss is explained by generic
sequence-length/state-saturation independent of same-space competition ⇒ the region is
**confounded** and NOT a clean memory region. Constrained scoring is over the shared scored
value vocabulary in BOTH conditions (identical chance level).

## 5. Three outcome channels, reported independently (train §16)

- **Constrained retrieval acc (PRIMARY endpoint):** argmax over the scored value vocabulary at
  the answer position == gold value. Justification: it probes retained answer *preference*
  independent of response-format failure — a base LM under an unusual `k=v` convention may not
  emit the value as global argmax, yet still rank the correct value first among plausible
  values; that is the memory signal 06C needs.
- **Unconstrained exact acc (secondary):** global argmax == gold.
- **Format adherence (secondary):** global argmax ∈ scored value vocabulary.

All three reported per dose per condition with raw denominators.

## 6. Competence requirement (train §17)

Predeclared **`τ_hi = 0.75`**. Competence = MP constrained acc at the lowest dose (`P=8`)
`≥ τ_hi`. If the model is not competent at low pressure on the independent qualification set:
`FIXED_BACKBONE_GRADED_REGION = BLOCKED`, reason `TASK_NOT_COMPETENT` (NOT reinterpreted as
forgetting).

## 7. Graded-region definition (predeclared; train §18) — thresholds NOT inherited blindly

`τ_hi = 0.75` (upper band edge / competence), **`τ_lo = 0.45`** (lower band edge / material
loss). A dose is **mid-band** iff `τ_lo < acc < τ_hi`. `QUALIFIED` requires ALL of, on the
pooled MP curve unless noted:
1. **Competence** — MP acc at `P=8` `≥ τ_hi`.
2. **Material high-pressure loss** — min MP acc across the grid `≤ τ_lo`.
3. **Interior resolution** — `≥ 2` mid-band doses (rules out immediate cliff / flat).
4. **Monotonicity** — MP acc non-increasing within tolerance `0.05`; at most 1 violation and
   none exceeding `0.10` (rules out noisy/non-monotone).
5. **Robustness across strata** — properties (1)+(3) hold in `≥ 2 of 3` seed strata
   (overlapping graded band), and per-cell 95% CIs reported.
6. **Confound-controlled** — at the two highest doses `{96,128}`, `mean(acc_LC − acc_MP) ≥
   0.15` (LC retains materially more ⇒ loss is same-space interference, not generic length).

Failure of any ⇒ `BLOCKED` with the specific reason (`TASK_NOT_COMPETENT`, `FLAT_HIGH`,
`IMMEDIATE_CLIFF`, `NON_MONOTONE`, `NOT_ROBUST_ACROSS_STRATA`, `CONFOUNDED_WITH_LENGTH`).
Distinguished shapes: flat-high (fails 2/3), flat-low/not-competent (fails 1), immediate cliff
(fails 3), noisy (fails 4), confounded (fails 6). **No threshold tuning after results.**

## 8. Statistical design (train §19)

- **Unit of analysis:** one example (a single probed prompt). Nested ladder ⇒ an example is
  **paired across doses** (same probe pair, growing distractors) and **paired across
  conditions** (same example under MP and LC).
- **N:** `S = 3` seed strata × `n_per_stratum = 64` ⇒ **`N = 192` per (dose, condition)**
  (3× P0's exploratory n=64).
- **Stress grid (doses):** `[8, 16, 24, 32, 48, 64, 96, 128]` — finer resolution spanning
  P0's P16–P128 transition (chosen from P0 calibration; frozen as `stressGridSha256` before
  outcomes).
- **CIs:** Wilson 95% per cell; cluster bootstrap over examples for the pooled MP curve.
- **Primary curve summary:** MP constrained acc vs dose with CIs.
- **Monotonicity diagnostic:** count of dose-to-dose violations beyond tolerance.
- **Raw denominators exposed** for every cell (`n_correct / n`). No seed-averaged percentages
  without counts.

## 9. Challenge-set identities & disjointness (train §13, §21) — frozen before outcomes

- `ops/rnn_06b_challenges.py` (`generator_version = rnn06b_mqar_matched_control_v1`,
  `master_seed = 20260813`) emits an ABSTRACT model-independent spec (per-example key/value
  slot permutations, probe index, stratum id, filler slot permutations) →
  `QUALIFICATION_SPEC.json`; `qualificationSetSha256` = SHA-256 over that spec.
- `STRESS_GRID.json` freezes the dose ladder, conditions, τ's, N, strata → `stressGridSha256`.
- **Disjointness:** `qualificationSetSha256 ≠ calibrationSetSha256` (P0 = `779fb37a…`) AND
  example-level disjointness — no abstract example (key_slots, val_slots, probe_index) equals
  any P0 calibration example (verified against `RNN-06-P0/calibration_examples.json`).
  Recorded as a contamination/disjointness proof. **No seed screening.**
- Tokenizer/model identity, per-dose materialized prompt SHAs, and scored/filler pool SHAs are
  recorded by the runner at execution.

## 10. Executed-source identity (train §8, §21) — before outcomes

Results record runner SHA-256, runner git blob, git HEAD, dirty indicator, model identity,
backend source hashes, protocol hash, `qualificationSetSha256`, `stressGridSha256`. Assert
`is_fast_path_available is False`. No outcome-bearing run with unidentified source bytes.

## 11. State-economics carry-forward (train §23)

Carry `state_bytes_per_sequence = 52,002,816` (bf16). Include a short derived estimate of the
future 06C snapshot cost over the qualified region (if any) so the next audit can judge
snapshot-cadence practicality. NO snapshot/reader/recovery machinery is built here.

## 12. Invariants

No GDN repair, no Qwen, no Memory Caching, no historical-state reader, no RNN-06C/06D, no
serving change, no training, no seed screening, no push. Conclusions scoped to this exact
checkpoint/backend/config (not generalized to "Mamba-2"). If not competent, failure is NOT
reinterpreted as forgetting. Preregistration is not edited after seeing results.
