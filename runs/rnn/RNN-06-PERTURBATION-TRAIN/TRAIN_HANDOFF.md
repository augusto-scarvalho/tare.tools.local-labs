# HANDOFF — RNN-06 Controlled State-Load Perturbation + Historical Information Train

**Author:** Claude (Opus 4.8) via Claude Code. **Date:** 2026-08-12. **Branch:** `master` (no
upstream). **Pushed:** NO. Two backlog items, one hard dependency gate. **Outcome C: both
QUALIFIED.**

## Dependency status (at a glance)

```
BACKLOG 1  RNN-06B3-MAMBA-CONTROLLED-STATE-LOAD-PERTURBATION : EXECUTED
                 STATE_LOAD_FORGETTING_PERTURBATION = QUALIFIED   TRANSITION_SHAPE = GRADED
                       │  gate OPENED
                       ▼
BACKLOG 2  RNN-06C-MAMBA-HISTORICAL-INFO                     : EXECUTED
                 HISTORICAL_STATE_INFORMATION = QUALIFIED
```

## HEAD boundary

- **before-train HEAD:** `41ecfb785744d510d9514fa4467b16cf86d53e52` (prev state-load train).
- **after-train HEAD:** `a3bf5c5c8e91eb42ed2e66293bcfa38f67dffa97`.
- Tree clean of tracked changes. Nothing pushed. No amend/rebase of outcome history.

## All commits (append-only; 41ecfb78..HEAD)

`7cda781` train protocol + B3 order-stable lib + calibration → `7a1054b` B3 calibration surface +
decision (freeze M=192/res16) → `1f8c491` B3 prereg + qual spec + grid + 06C dose rule →
`b42b976` B3 runner → `7f813be` B3 trapz fix (pre-outcome engineering fix) → `ec8dd88` B3 results
(QUALIFIED) → `c3dbf20` B3 decision → `bf4dbdb` 06C prereg + held-out spec → `0a7da96` 06C runner
→ `b7a284c` 06C results+decision (QUALIFIED) → `a3bf5c5` ops WU revert script.

## Exact identities (frozen subject; both items)

`AntonV/mamba2-1.3b-hf` @ `703e19a43f397c70315244a3424d79456b54fb34`; transformers-native
`Mamba2ForCausalLM`; transformers 4.48.3; torch 2.6.0+cu124; bf16; no quantization; no
mamba_ssm/causal_conv1d (`is_fast_path_available=False`); **`chunk_size=32`**; `modeling_mamba2.py`
sha256 `83685d78…`. Executed-source PROVEN: B3 runner blob `95e6b7aa` == committed (dirty ∅);
06C runner blob `3fdf8fd` == committed (dirty ∅). Prior gates immutable: P0 `d35db764`, 06A2-cs32
`4c2dd568`, 06B2 `e1ca4261`.

## Audit-confound fixes (Item 1) — both eliminated, verified

- **Temporal-order churn** (B2 assigned load bindings by scanning active slots): fixed via a
  permanent ordinal↦slot↦binding map. `nestedBindingIdentityCheck = PASS` — 384 checks (192×2
  arms), **0 failures**: as U rises, already-active bindings keep identity AND position.
- **Full-packing boundary** (B2's loss only at U=M): excluded via `MIN_SENTINEL_RESERVE = 16`;
  **no U=M cell** (min reserve 16 at U=176).

## Calibration vs qualification boundary (Item 1)

- **Calibration (EXPLORATORY):** `b3CalibrationSetSha256 = 342f0961…`, N=48, candidate family
  M∈{128,192}×reserve∈{16,32} (no grid extension used). Chose **M=192, reserve=16, REPEAT1,
  target@slot0**. `nested_identity_all_pass=True`.
- **Qualification (frozen):** `b3QualificationSetSha256 = 9c878e2c…`, `b3StressGridSha256 =
  845bd641…`, N=192/(dose,arm), S=3 strata. Disjoint from P0/06B/B2-calib/B2-qual/B3-calib
  (distinct seeds + generator + example-level 0 overlap vs B3-calib + distinct SHAs). No seed
  screening.

## BACKLOG 1 — RNN-06B3 — DS curve (order-stable, subpacked, fixed 770 tok, fixed gap)

| U | sentinels | DS (k/n) | 95% CI | SS |
|---:|---:|---|---|---:|
| 1   | 191 | 0.990 (190/192) | [0.963,0.997] | 0.990 |
| 24  | 168 | 0.917 (176/192) | [0.869,0.948] | 0.917 |
| 48  | 144 | 0.865 (166/192) | [0.809,0.906] | 0.844 |
| 72  | 120 | 0.771 (148/192) | [0.706,0.825] | 0.755 |
| 96  |  96 | 0.755 (145/192) | [0.690,0.811] | 0.693 |
| 128 |  64 | 0.651 (125/192) | [0.581,0.715] | 0.562 |
| 152 |  40 | 0.568 (109/192) | [0.497,0.636] | 0.542 |
| 176 |  16 | 0.573 (110/192) | [0.502,0.641] | 0.557 |

**Gate: all 9 pass.** competence 0.990; **paired loss U1→U176 = 0.417, CI [0.349,0.490]**;
robust 3/3; **discordant 81 correct→wrong vs 1 wrong→correct**; nested-identity 0/384 fail;
reserve 16; fixed length/gap. `MEAN_RELATIVE_RETENTION_DEFICIT=0.231`, `DEFICIT_AURC_NORMALIZED=
0.238`. **`STATE_LOAD_FORGETTING_PERTURBATION = QUALIFIED`, `TRANSITION_SHAPE = GRADED`** (3
interior doses, max step 0.104 < 0.60·total). DS≈SS (general load dominates). Frozen B3→06C
doses: HIGH=U152 (0.568), LOW=U96 (0.755), MID=U72 (0.771).

## BACKLOG 2 — RNN-06C — three state conditions (same H prefix; identical query/model/backend)

| condition | body | query pos | acc (k/n) | 95% CI |
|---|---|---:|---:|---|
| H historical-direct | none | 4 | 0.849 (163/192) | [0.791,0.893] |
| N same-aged neutral | 764 sentinel | 768 | **1.000 (192/192)** | [0.980,1.000] |
| L high-load (U=152) | 764 order-stable load | 768 | 0.547 (105/192) | [0.476,0.616] |

**PRIMARY `neutral_minus_load` (N−L) = 0.453, CI [0.385,0.526]**, robust 3/3 (0.500/0.453/0.406).
`historical_minus_load` = 0.302. Transitions: **N_correct→L_wrong = 87, N_wrong→L_correct = 0**;
H_correct→L_wrong = 75, H_wrong→L_correct = 17. L reproduces B3 U=152 (0.547 vs 0.568, ±0.10).

**Machinery validity:** boundary self-check 8/8 (state hashes reproduced; branch-from-same-H);
counters `snapshotsCreated=576, snapshotsRestored=576, H/N/L readouts=192 each,
branchPairsCompleted=192, snapshotBoundaryFailures=0, queriesEvaluated=576`. Every snapshot
carries temporal identity (prefix/conv/ssm hashes, `cachePosition==len(prefix)`).

**Gate: all pass ⇒ `HISTORICAL_STATE_INFORMATION = QUALIFIED`.** Interpretation: from an identical
earlier state, neutral aging preserves the target (N=1.000) but the qualified load destroys
behavioral access (L=0.547) at matched length/position/gap — so the info was present earlier and
remains accessible from the neutral-aged state. **Presence only — no recovery, no reader, no
Memory Caching.**

## Status labels advanced (precise; historical artifacts NOT rewritten)

- `UNIQUE_LOAD_EFFECT`: → **CONFIRMED (order-stable, subpacked)**.
- `FULL_PACKING_BOUNDARY`: → **EXCLUDED**.
- `GENERAL_RECURRENT_STATE_SATURATION`: → **QUALIFIED**.
- `HISTORICAL_STATE_INFORMATION`: → **QUALIFIED (presence)**.
- RNN-06B2 `BLOCKED` and its `CONFOUNDED_WITH_LENGTH` label unchanged; RNN-06A2 QUALIFIED (cs=32),
  `GDN_COMPATIBILITY_GAP=OPEN`, `QWEN_GDN_TRANSPLANT_GATE=DEFER` unchanged. Prior-train 06C
  `BLOCKED_BY_06B2` markers preserved + explicitly superseded (`SUPERSEDED_NOTE.md`).

## Deviations & operational notes

- **B3 trapz fix** (`7f813be`): numpy 2.5.2 removed `np.trapz`; the FIRST B3 run computed all 16
  GPU cells then crashed in post-GPU analysis before writing results. Version-proof trapezoidal
  fix committed BEFORE the results-bearing re-run (deterministic — curves identical to the logged
  first run). Per §22 implementer autonomy (fix runner bugs); no threshold/model change.
- **06C interruption**: the first 06C run was killed by a **Windows Update auto-restart**
  (2026-08-12 01:29, Event 1074 "Service pack (Planned)" — verified via Windows Event Log; NOT a
  power blackout; no Event 41/6008 that day). Re-executed deterministically to completion. Applied
  `HKLM\…\WindowsUpdate\AU\NoAutoRebootWithLoggedOnUsers=1` to defer future auto-reboots during
  runs; revert script `ops/revert_wu_noautoreboot.ps1` (backup: AU key was absent before).
- No qualification threshold was changed after outcomes; no seed screening; no example regeneration.

## Resource usage

B3: runtime 2776 s, peak VRAM 8.8 GB (cs=32, 770-tok cells, batch 2). 06C: runtime 435 s, peak
VRAM 8.3 GB (streaming snapshots, batch 2). State bytes/seq = 52,002,816 (no permanent snapshot
explosion; streamed capture→hash→restore→release; boundary self-check on 8-example audit sample).

## Bundle invariant

`ops/rnn_06_perturbation_bundle.py` enforces: **archive payload == manifest payload == SHA256SUMS
payload**, excluding only `TRAIN_MANIFEST.json` / `SHA256SUMS.txt` by explicit rule; **raises on
any unmanifested payload** (verified, incl. negative test).

## Confirmations

`NO_HISTORICAL_ARTIFACT_REWRITTEN=TRUE` · `NO_HISTORICAL_COMMIT_REWRITTEN=TRUE` ·
`THRESHOLDS_NOT_TUNED_AFTER_OUTCOMES=TRUE` · `NO_SEED_SCREENING=TRUE` ·
`FROZEN_MODEL_INVARIANT_HELD=TRUE` · `NESTED_IDENTITY_PASS=TRUE (0/384)` ·
`SNAPSHOT_BOUNDARY_FAILURES=0` · `NO_RECOVERY_BUILT=TRUE` · `NO_READER_TRAINED=TRUE` ·
`NO_MEMORY_CACHING=TRUE` · `NO_GDN=TRUE` · `NO_QWEN=TRUE` · `NO_RNN_06D=TRUE` · `NOTHING_PUSHED=TRUE`.

## Exactly one next recommendation (NOT executed) — stop/pivot outcome C

**OPEN `RNN-06D-MAMBA-HISTORICAL-STATE-RECOVERY-UTILITY` in a NEW session** — test whether the
demonstrated historical-state *presence* can be *exploited* (a recovery/read contract with
downstream utility), under independent audit and a fresh preregistration, on fresh held-out data.
Do NOT implement recovery, a reader, Memory Caching, GDN repair, or Qwen in this train.

**STOP. Do NOT start RNN-06D / recovery / GDN repair / Qwen here.**
