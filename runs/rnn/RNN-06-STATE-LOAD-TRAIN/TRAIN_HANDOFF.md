# HANDOFF — RNN-06 Fixed-Length State Load + Historical Information Train

**Author:** Claude (Opus 4.8) via Claude Code. **Date:** 2026-08-11. **Branch:** `master` (no
upstream). **Pushed:** NO. Two backlog items, one hard dependency gate.

## Dependency status (at a glance)

```
BACKLOG 1  RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD : EXECUTED
                 FIXED_LENGTH_STATE_LOAD_REGION = BLOCKED (IMMEDIATE_CLIFF + NOT_ROBUST_ACROSS_STRATA)
                       │
                       └── gate did NOT open ──►
BACKLOG 2  RNN-06C-MAMBA-HISTORICAL-INFO         : BLOCKED_BY_06B2 (not executed)
                 HISTORICAL_STATE_INFORMATION = NOT_MINTED
```

## HEAD boundary

- **before-train HEAD:** `79e0dc5c3c2832555b6a9a8f9794f7805d5f06dd` (previous train evidence
  commit, atop `1ddf64a` 06B decision).
- **after-train HEAD:** `42e449e7c7d691310069428abeae07be49f3631a`.
- Tree clean of tracked changes. Nothing pushed. No amend/rebase of outcome history.

## All commits (append-only; 79e0dc5..HEAD)

| commit | boundary |
|---|---|
| `38db83f` | train protocol + B2 fixed-length lib + calibration runner (no gate) |
| `f94b65e` | B2 calibration grid extended to M∈{48,64,96,128} (exploratory) |
| `6b0bfdf` | B2 calibration surface + decision → freeze M=128/REPEAT1 (exploratory) |
| `33c4c32` | B2 pre-registration + frozen qualification spec + stress grid + 06C dose rule (no outcomes) |
| `eff78c1` | B2 fixed-length state-load runner (no outcomes) |
| `598115a` | B2 results → FIXED_LENGTH_STATE_LOAD_REGION = BLOCKED |
| `d3766ac` | B2 decision → BLOCKED |
| `42e449e` | RNN-06C = BLOCKED_BY_06B2 (not executed) |
| *(this handoff commit)* | train evidence + handoff + bundle builder |

## Exact identities (frozen subject; both items)

`AntonV/mamba2-1.3b-hf` @ `703e19a43f397c70315244a3424d79456b54fb34`; transformers-native
`Mamba2ForCausalLM`; transformers 4.48.3; torch 2.6.0+cu124; bf16; no quantization; no
mamba_ssm/causal_conv1d (`is_fast_path_available=False`); **`chunk_size=32`**;
`modeling_mamba2.py` sha256 `83685d78…`. Executed-source PROVEN: B2 runner blob `07a3e0b7` ==
committed, dirty = ∅; lib blob `095403ab`, dirty = ∅; HEAD at run `eff78c1`.

Prior gate artifacts verified immutable: P0 `d35db764`, 06A `d10527b6`, 06A2-cs32 `4c2dd568`,
06B `22d355bd`.

## Calibration vs qualification boundary (B2)

- **Calibration (EXPLORATORY, no gate):** `b2CalibrationSetSha256 =
  727c53678f8393882a743a49f3919bcd734c97d6a891590110e0ad4d7d09a2b2`, N=48/cell, DS arm, swept
  M∈{48,64,96,128}×sentinel∈{REPEAT1,CYCLE4}×coarse U. Used ONLY to choose the frozen
  construction: **M=128 (514 tokens), REPEAT1 sentinel, target@slot0**. No examples cherry-picked.
- **Qualification (frozen, preregistered):** `b2QualificationSetSha256 =
  a92870a99babdb93b3e232549e2db26f8dec68c520b50cbc13dff56bf583ea10`; `b2StressGridSha256 =
  7afd7275b02c6a09cf50b9b08fa7c41463f633b3f8086b2b9a615fb808f5de46`; N=192/(dose,arm), S=3
  strata. Disjoint from P0 calibration (`779fb37a`), RNN-06B qualification (`e351a444`), B2
  calibration (`727c5367`) — distinct seeds/generator + example-level (0 overlap vs B2 calib) +
  distinct SHAs. No seed screening.

## BACKLOG 1 — RNN-06B2 — full curves & raw denominators (chance 1/256)

Primary = DS constrained retrieval at FIXED 514-token length, target→query gap fixed.

| U | DS con (k/n) | DS boot-95% | SS con | DS−SS | DS unc | DS fmt |
|---:|---|---|---:|---:|---:|---:|
| 1   | 0.958 (184/192) | [0.927,0.984] | 0.958 | +0.000 | 0.682 | 0.698 |
| 24  | 0.953 (183/192) | [0.922,0.979] | 0.922 | +0.031 | 0.708 | 0.745 |
| 48  | 0.906 (174/192) | [0.865,0.943] | 0.870 | +0.036 | 0.557 | 0.604 |
| 64  | 0.859 (165/192) | [0.807,0.906] | 0.828 | +0.031 | 0.458 | 0.510 |
| 80  | 0.812 (156/192) | [0.755,0.865] | 0.797 | +0.016 | 0.396 | 0.432 |
| 96  | 0.771 (148/192) | [0.714,0.828] | 0.755 | +0.016 | 0.281 | 0.302 |
| 112 | 0.714 (137/192) | [0.651,0.776] | 0.703 | +0.010 | 0.266 | 0.281 |
| 128 | 0.432 (83/192)  | [0.359,0.505] | 0.411 | +0.021 | 0.068 | 0.078 |

**Length diagnostic (non-gating, U=2 fixed low load):** M=32/64/96/128 → 0.948 / 0.943 / 0.932 /
0.938 (130 → 514 tokens). **Length-alone barely degrades.**

**Gate:** 7/9 criteria pass (competent 0.958; material loss 0.432≤0.45; monotone 0 violations;
delta-AURC 0.164 [CI 0.127–0.203] ≥0.15; fixed length 514 all doses; fixed gap; frozen). BLOCKED
by **criterion 3 (`IMMEDIATE_CLIFF` — only 1 mid-band dose; transition is a −0.28 cliff at
fully-packed U=128=M)** and **criterion 6 (`NOT_ROBUST_ACROSS_STRATA` — graded in 1/3 strata)**.

## Controls & negative/positive evidence

- **Real directional effect:** DS declines monotonically with unique-binding load at FIXED
  length (0.958→0.432; delta-AURC 0.164). Positive evidence for general state-load forgetting.
- **NOT a stable graded region:** shallow plateau then a cliff at full packing; 1 interior
  mid-band dose; not robust. Thresholds NOT tuned post-hoc → BLOCKED.
- **Length disambiguated:** length-only diagnostic flat (~0.94 across 130–514 tokens) ⇒
  degradation is unique-load-driven, not token-count.
- **DS ≈ SS** at every dose ⇒ same-space competition adds little; general state load dominates.

## Status labels carried (precise; historical labels NOT rewritten)

- `GENERAL_STATE_LOAD_FORGETTING = DIRECTIONAL_SUPPORT_WITHOUT_QUALIFIED_REGION`.
- `LENGTH_VS_STATE_LOAD = DISAMBIGUATED_TOWARD_UNIQUE_LOAD` (was NOT_DISAMBIGUATED).
- `SAME_SPACE_ASSOCIATIVE_INTERFERENCE = NOT_SUPPORTED` (reinforced; DS≈SS).
- RNN-06B `CONFOUNDED_WITH_LENGTH` machine label unchanged (historical). RNN-06A NOT_QUALIFIED,
  RNN-06A2 QUALIFIED (cs=32), `GDN_COMPATIBILITY_GAP=OPEN`, `QWEN_GDN_TRANSPLANT_GATE=DEFER`
  unchanged.

## BACKLOG 2 — RNN-06C — BLOCKED_BY_06B2 (not executed)

Dependency gate did not open ⇒ no `historicalInfoSetSha256`, no state-readout runs, no snapshots
written, no reader trained, no recovery built, `HISTORICAL_STATE_INFORMATION = NOT_MINTED`.
Explicit markers: `RNN-06C-MAMBA-HISTORICAL-INFO/BLOCKED_BY_06B2.md` + `BLOCKED.json`. The frozen
06C dose-selection rule (HIGH=U128, LOW=U80, MID=U112) is recorded in `B2_RESULTS.json →
c06_dose_selection` with `applicable=False`.

## Deviations

- Calibration grid was extended once (M∈{48,64}→{48,64,96,128}) during the EXPLORATORY
  calibration stage (before any gate/qualification) to locate a feasible material-loss region —
  a permitted calibration action, not a threshold change.
- No protocol amendment was needed for B2/06C outcomes. Bundle invariant fixed (see below).

## Bundle invariant (fix carried from prior train §22)

`ops/rnn_06_state_load_bundle.py` enforces: **archive payload == manifest payload == SHA256SUMS
payload**, excluding only `TRAIN_MANIFEST.json` and `SHA256SUMS.txt` by explicit rule, and
**raises if any unmanifested payload is present** (all payload, incl. 06C `BLOCKED.json`/
`BLOCKED_BY_06B2.md`, are real on-disk manifested files — no writestr-only payloads).

## Resource usage

B2 qualification: runtime 2069 s, peak VRAM 9.8 GB (cs=32, 514-token cells, batch 2). Calibration
(two exploratory passes) within budget. State bytes/seq = 52,002,816 (carried; no 06C snapshots).

## Confirmations

`NO_HISTORICAL_ARTIFACT_REWRITTEN = TRUE` · `NO_HISTORICAL_COMMIT_REWRITTEN = TRUE` ·
`FAILED_GATE_NOT_REINTERPRETED = TRUE` (BLOCKED preserved; thresholds not tuned) ·
`NO_SEED_SCREENING = TRUE` · `FROZEN_MODEL_INVARIANT_HELD = TRUE` · `NO_GDN = TRUE` ·
`NO_QWEN = TRUE` · `NO_MEMORY_CACHING = TRUE` · `NO_RECOVERY_BUILT = TRUE` ·
`NO_READER_TRAINED = TRUE` · `NO_RNN_06C_EXECUTED = TRUE` · `NO_RNN_06D = TRUE` ·
`NOTHING_PUSHED = TRUE`.

## Exactly one next recommendation (NOT executed)

**OPEN `RNN-06B3-MAMBA-STATE-LOAD-SUBPACKING` in a NEW session** — an independently preregistered
experiment that (a) uses **fixed length with U capped BELOW full packing (always ≥ a few
sentinels)** to test whether the state-load decline is smoothly graded once the U=M packing
boundary is excluded, and (b) adds finer interior doses around the transition with a mid-band
window justified from the calibration curve, to determine whether a *stable, robust* graded
state-load region exists away from the packing artifact — before any historical-information
(06C) work is reconsidered. Do NOT repair GDN, run Qwen, build recovery/Memory Caching, or start
RNN-06C/06D there.

**STOP after this train. Do NOT implement RNN-06D / Memory Caching / recovery / GDN repair / Qwen.**
