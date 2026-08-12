# RNN-06D0 — PRE-REGISTRATION (Recovery Ceiling & Snapshot Schedule)

Frozen BEFORE the D0 qualification run. Thresholds/SESOI were frozen earlier in `TRAIN_PROTOCOL.md`
and are NOT changed here. Calibration (exploratory) chose only the configuration
(`D0_CALIBRATION_DECISION.md`, AMENDMENT 1 v2, K=4). Qualification set is fresh + disjoint. No seed
screening. No threshold tuned after outcomes.

## Frozen subject (verified live)

`AntonV/mamba2-1.3b-hf` @ `703e19a43f397c70315244a3424d79456b54fb34`; transformers-native
`Mamba2ForCausalLM`; transformers 4.48.3; torch 2.6.0+cu124; bf16; no quantization; no
mamba_ssm/causal_conv1d (`is_fast_path_available=False`, naive `torch_forward`); `chunk_size=32`.
State = `Mamba2Cache{conv,ssm}` = 52,002,816 bytes/seq (remeasured in the runner).

## Frozen construction (v2, anti-oracle) and schedule

- `GENERATOR_VERSION = rnn06d_anti_oracle_random_target_v2_sentinel_pre_load_post`, M=192, band
  [T_MIN,T_MAX]=[8,64]. Target-write slot t randomized in the band; slots [0,t-1] = REPEAT1 sentinel;
  slot t = scored-space target; slots [t+1,M-1] = unique DS filler load. Fixed length 4·192+2 = 770.
  Constrained argmax over 256 scored values; chance 1/256; gold = target value.
- **Snapshot schedule (target-agnostic, frozen):** K=4, slots [38, 76, 115, 153]
  (`floor(M·(k+1)/(K+1))`), all strictly before FINAL (slot 191). Independent of t and gold. Slot 38
  is a **pre-target distractor** for t∈(38,64]; {76,115,153} straddle/follow the target.
- Snapshot@slot s = state from an independent prefill of prefix [0:4(s+1)] (naive backend has no
  multi-token mid-sequence forward). Boundary self-check re-prefills and matches conv/ssm hashes.

## Frozen challenge identities

- `qualificationSetSha256 = 7bbf9b753ed769adf93573f50b15d91a0e89bf70c14c2951adb5fc022f29d324`
  (seed 20260902; N=192 = 3 strata × 64). Example-level disjoint (overlap 0) from p0_calib,
  rnn06b_qual, b2_calib, b2_qual, b3_calib, b3_qual, c06, and d0_calib.
- `snapshotScheduleSha256 = 355e51056793ba46375f71e6a1f6d6f9f72e564678487641fda715f4b2e1c541`.
- `calibrationSetSha256 = 0bf7d2613c4054f602e01662c66eb97260adce59806b10ca33a6d5b75601a936` (seed
  20260901; distinct from qualification).

## Arms

- **FINAL** — full-sequence recurrent state (baseline; degraded regime).
- **FIXED_HISTORICAL_POOL** — all K=4 target-agnostic snapshots (substrate for D1).
- **ORACLE_TARGET_PROXIMAL** — first pool snapshot at/after t (diagnostic; uses t).
- **ORACLE_BEST_GOLD** — any pool snapshot correct vs gold (upper bound; uses gold).

## Frozen thresholds (from TRAIN_PROTOCOL; restated, unchanged)

`CEILING_SESOI=0.15`, `FINAL_ACC_MAX=0.75`, `TAU_PROX=0.75`, CI lower bound > 0.05, robustness ≥ 2/3
strata at CEILING_SESOI, recoverable substrate `RECOV_FRAC_MIN=0.30` and `RECOV_N_MIN=20`.

## Gate — mint exactly `RECOVERY_CEILING ∈ {QUALIFIED | TOO_SMALL | NOT_TESTABLE}`

1. boundary/identity/counter checks fail ⇒ `NOT_TESTABLE`;
2. else FINAL not degraded (acc > 0.75) or ORACLE_PROXIMAL not competent (< 0.75) ⇒ `NOT_TESTABLE`;
3. else {ORACLE_BEST−FINAL ≥ 0.15, CI_lb > 0.05, robust ≥ 2/3, recoverable ≥ 0.30 & ≥ 20} ⇒
   `QUALIFIED`;
4. else ⇒ `TOO_SMALL`.

If not QUALIFIED: persist negative evidence, set `RNN-06D1 = BLOCKED_BY_D0`, do not run D1, package,
STOP; one next recommendation = PARK historical-snapshot recovery on this substrate, open the
current-state-memory line. If QUALIFIED: proceed to D1 (separate prereg).

## Executed-source discipline

Runner git blob == committed, dirty ∅, recorded before outcomes; `is_fast_path_available=False`
asserted; spec/schedule SHAs re-verified in-runner. Prior gates (P0/06A/06A2/06B/06B2/06B3/06C) not
modified.
