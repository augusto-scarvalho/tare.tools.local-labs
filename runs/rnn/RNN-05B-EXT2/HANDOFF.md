# RNN-05B-EXT2 — HANDOFF

**Fixed-Backbone Retention Dose–Response — the FINAL planned synthetic H3 experiment.**

## Headline verdict

| field | value |
|---|---|
| `FIXED_BACKBONE_GRADED_REGION` | **BLOCKED** |
| `H3_TESTABILITY` | **BLOCKED_FIXED_BACKBONE** |
| `H3` | **BLOCKED_FIXED_BACKBONE** (Case A) |
| `QWEN_GDN_TRANSPLANT_GATE` | **DEFER** |
| `SYNTHETIC_DENSE_MC` | **PARK** |
| next | **STOP — no MC was run, no EXT3** |

**One-sentence result:** three GDN (and three DN, one LA) backbones, each trained ONCE to competence across the
full inference-distractor-density ladder and then frozen, retain **all 12 associations with ≥0.98 recall at every
dose up to 0.64 density (≈250 gap distractors)** — there is **no graded-forgetting regime** for these fixed
representations over the tested inference-pressure range, so historical-state recovery cannot be tested and the
synthetic dense Memory-Caching line is parked.

## Git state
- **Start HEAD:** `df8317b5acc7c7d982663153f70bacf34892fdbf` (master).
- **Pre-registration commit:** `5abeab4b7f55db966e5148261f715fcc524bd168` — written+committed BEFORE any
  outcome-bearing run; recorded as `sourceGitHead` inside `rnn05bext2_results.json`/`_outcomes.json`.
- **End HEAD:** see `git_evidence.txt` / the results commit printed at the end. **Nothing pushed.** RNN-05B/EXT
  evidence untouched (immutable).
- Weights (`artifacts/*.pt`, ~368 KB each) are **gitignored** (`.gitignore`) — bundled in the ZIP, not committed.

## CURRENT / RESEARCH / PROPOSED
- **CURRENT (this packet):** synthetic toy LA/DN/GDN substrate (`rnn_delta_substrate`, UNCHANGED), memory-bound
  nested MQAR, fixed-backbone inference-stress. No Qwen weights, no serving, no TPTT, no FLA, no new kernels.
- **RESEARCH pointers (recorded, NOT executed):** DART, StateX, Sparse Delta Memory, Gated-DeltaNet-2, FG2-GDN,
  SGLang int8 recurrent checkpoint pool, ReplaySSM. If EXT2 blocks (it did), these are candidates for a *separate*
  RNN-06 comparative research packet — NOT mixed into EXT2.
- **PROPOSED next:** none in the synthetic line (Case A ⇒ no EXT3). The single recommendation is at the bottom.

## Pre-registration (written before any outcome-bearing run)
`PRE_REGISTRATION.md` + `machine_config.json`, both derived from ONE frozen `Ext2Config`. Key locks:
- **Backbone reuse decision (§2):** RNN-05B backbones were NOT saved to disk AND were trained on a different
  (capacity) MQAR distribution ⇒ exact reuse INVALID+IMPOSSIBLE ⇒ `BACKBONE_REUSE = RETRAIN_ONCE_THEN_FREEZE`
  (train each seed once, save, SHA-256, freeze; identical weights for every stress point).
- **Architecture UNCHANGED:** `MQARDeltaModel` d_model=128, d_k=d_v=64, conv_k=4, seg=64.
- **Task:** memory-bound MQAR, **fixed seq_len=512**, num_pairs=12 (far below the ~40 capacity cliff),
  num_queries=8, num_keys=128, num_vals=64, writes in the early 25% of the body, queries at the end.
- **Stress axis (§5):** `postwrite_gap_distractor_density_nested` — distractor keys fill the post-write retention
  gap in a fixed ascending order; the dose ladder is **nested** (higher dose = same base example + more gap
  distractors). Ladder = `[0.0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64]` (n_distractors
  `[0,31,62,94,125,156,187,218,250]`).
- **One recipe:** each seed trained ONCE, single-state, on a MIXTURE over the ladder (domain-randomized dose per
  step), steps=2500, lr=3e-3, batch=96, pool_train=4096; then frozen. Seeds (ALL count; no screening): GDN
  {42,43,44}, DN {42,43,44}, LA {42}.
- **SESOI (§16):** PRIMARY on DELTA_AURC = **0.05**, justified by MC's ~7× live-state storage + read-latency cost
  at seq_len 512/seg 64. The old **3% margin = `OPERATOR_HEURISTIC` only** (direction labels), not authority.
- **Decision policy (§21):** Cases A/B/C predeclared.

## challengeGridSha256 (§4)
`66ff24765d17c4fa95dcfcbaf4a7b374c66aa1ba52e507cd047b486ad512a9e5` — recorded **identically** in
PRE_REGISTRATION.md, machine_config.json, BASE_QUALIFICATION.json, results.meta, outcomes.
Process-stable self-check (fresh subprocess reconstructs the exact config): **PASS**.

## Exact frozen-backbone identities (backboneSha256, first 16 hex)
| substrate | seed | backbone_sha256 | file_sha256 | final loss | train wall_s |
|---|---|---|---|---|---|
| gdn | 42 | `15c96113f887132f` | `50d6a001d159` | 0.0007 | 99.8 |
| gdn | 43 | `df7387c234890b1f` | `d2ce641a19c0` | 0.0014 | 101.0 |
| gdn | 44 | `1b5d886ea8960a81` | `352f50ddad0b` | 0.0019 | 99.8 |
| dn | 42 | `28e8829e661fad0f` | `c6880dc67fca` | 0.0010 | 94.8 |
| dn | 43 | `5b073d0a26056f6d` | `7ecedb2a082c` | 0.0005 | 94.5 |
| dn | 44 | `2a57bbd8e7937985` | `cb75da0238aa` | 0.0007 | 95.2 |
| la | 42 | `ec147cb38adc646b` | `f5213ff4e057` | 0.0007 | 32.1 |

exampleSetSha256: train `741b48d29aaedbf2`, dev `49da3c06f1b365eb`, holdout `6d54ce009431c6e6`.

## BASE qualification artifact + proof it occurred BEFORE any MC
`BASE_QUALIFICATION.json` contains challengeGridSha256, configSha256, sourceGitHead, exampleSetSha256, stressAxis,
dose_ladder, backboneSha256[], **per-seed BASE retention curves**, gdn_graded_region, verdict. The MC entrypoint
**loads + verifies** it (grid/config/backbone hashes) and returns on BLOCK **before** any MC code — see
`source_excerpts.md §1` (`rnn_05b_ext2.py:1058-1076`). Executed proof: `rnn05bext2_results.json` has
`mc={}`, `snapshot_identity={}`, `ablation={}` — **MC/reader/ablation never ran** (cleaner than RNN-05B-EXT, which
had the gate-ordering defect and post-hoc exploratory MC). Gate verify logged `ok=True (verified)`.

## Per-seed BASE retention curves (holdout, BASE = single final state; dose → accuracy)
```
dose:      0.0    0.08   0.16   0.24   0.32   0.40   0.48   0.56   0.64
gdn s42  0.9968 0.9966 0.9968 0.9968 0.9971 0.9963 0.9968 0.9971 0.9961
gdn s43  0.9978 0.9941 0.9941 0.9941 0.9937 0.9937 0.9934 0.9929 0.9924
gdn s44  0.9878 0.9841 0.9866 0.9851 0.9863 0.9863 0.9841 0.9839 0.9856
dn  s42  0.9971 0.9814 0.9819 0.9817 0.9817 0.9812 0.9817 0.9817 0.9819
dn  s43  0.9976 0.9971 0.9976 0.9976 0.9976 0.9971 0.9971 0.9971 0.9968
dn  s44  0.9990 0.9990 0.9990 0.9990 0.9988 0.9985 0.9980 0.9980 0.9980
la  s42  0.9998 0.9995 0.9995 0.9993 0.9995 0.9993 0.9993 0.9990 0.9993
```
Every curve is **flat-high**: competent at low dose (max ≥0.75 ✓) but **never degrades** (min 0.984–0.996, never
≤ grade_lo=0.45), **zero doses in the mid band [0.40,0.80]**. `AURC_RETENTION ≈ 0.99` for all curves (full CSV in
`rnn05bext2_curves.csv`).

## Graded-region gate (§7) — the load-bearing decision
Rule: qualify iff every GDN seed has max BASE ≥0.75, min BASE ≤0.45, ≥2 doses with BASE∈(0.40,0.80), AND the mid
doses OVERLAP across all GDN seeds. Result per GDN seed: `competent=True, degrades=False, resolved=False,
seed_graded=False`; `ALL_SEEDS_GRADED=False`, `COMMON_REGION_NONEMPTY=False` ⇒ **FIXED_BACKBONE_GRADED_REGION =
BLOCKED**. A single cell in an arbitrary band would NOT have qualified anyway — none exists.

## AURC / D50 / transition width; recovery/harm; path counters; ablation; stats; efficiency
**Not computed** — all are downstream of the graded-region gate, which BLOCKED. This is by design (§6/§7): with no
graded region there is no BASE-wrong population to recover and no dose-response transition to fit. Reporting these
would be meaningless. The machinery for every one of them is implemented and pipeline-validated (smoke run:
mutation=0 on all cells, snapshot-position identity PASS, ablation random control = 4 excluding proximal {0,1} and
irrelevant 6, hierarchical bootstrap CI, efficiency Pareto with prewarm/warm) — see `source_excerpts.md` and
`ops/rnn_05b_ext2.py`.

## Substrate sanity (cheap reconfirm; already qualified in RNN-05B)
LA/DN/GDN all: REFERENCE_PARITY=PASS, FULL_MODULE_LIFECYCLE=NUMERICALLY_EQUIVALENT, REQUEST_ISOLATION=PASS.
Eager sequential scan remains the correctness reference (§20); no new kernels introduced.

## SESOI + decision logic
SESOI(DELTA_AURC)=0.05 (primary). Decision policy **Case A** applies: *no graded fixed-backbone region* ⇒
`H3_TESTABILITY=BLOCKED_FIXED_BACKBONE`, `QWEN_GDN_TRANSPLANT_GATE=DEFER`, `SYNTHETIC_DENSE_MC=PARK`, STOP, **no
EXT3**. (Cases B/C were unreachable — they require a qualified region.)

## Performance methodology & storage/compute Pareto
Efficiency accounting is implemented (prewarm→warm steady-state, live matrix/conv bytes, historical-snapshot
bytes, peak VRAM, per-stage latency, DELTA_AURC_PER_KiB/ms) but not exercised — MC never ran. Run cost:
**10.4 GPU-minutes total** (7 backbones × ~1.6 min + BASE sweep), well under the <1 GPU-hr target / 2 h ceiling.
Peak VRAM <1 GB. BASE qualification was cheap because backbones are frozen (design did NOT drift toward
per-stress-point retraining — §23 satisfied).

## Failures and negative evidence (honest scope)
- **This is a NEGATIVE result and a recipe-dependent one.** The pre-registered recipe trains each backbone on a
  MIXTURE spanning the *same* dose ladder used for the inference sweep, so every dose (incl. 0.64) is
  in-distribution — the model learned to **retain through the interference**, yielding flat-high robustness and no
  graded region. This does **not** prove that *no* fixed GDN can ever exhibit graded forgetting: it shows that a
  representation trained to competence **across** the tested pressure range is robust to that range. A different
  (now **not pursued**, per the Case-A / no-EXT3 decision) design — narrower training pressure, or inference
  pressure pushed beyond the training range, or a stronger axis (much longer retention gap, larger pair count near
  the capacity cliff) — is a **distinct untested regime**, labeled here `FIXED_BACKBONE_GRADED_FORGETTING_UNDER_
  OTHER_RECIPES = NOT_TESTED`.
- Contrast with RNN-05B-EXT: EXT trained a *fresh* backbone per condition and saw a sharp seed cliff
  (`TRAIN_PER_CONDITION_STABILITY=FAILED`). EXT2 removes that confound (one frozen backbone) and finds the opposite
  failure mode for testability — **robustness**, not instability. Both routes leave H3 untestable in the synthetic
  toy: EXT via unstable base, EXT2 via a non-degrading fixed base.
- No pre-registration deviation, no seed screening, no post-hoc grid/recipe change after seeing BASE curves
  (that would be exactly the fishing the RNN-05B-EXT audit prohibited).

## Reproduction commands (WSL Ubuntu-24.04, `~/tptt-venv/bin/python`, torch 2.6.0+cu124, RTX 3090)
```
cd /mnt/c/projects/local-model-lifecycle/ops
# pre-registration + machine config + challengeGridSha256 (no GPU)
~/tptt-venv/bin/python rnn_05b_ext2.py --preregister ../runs/rnn/RNN-05B-EXT2
# nested-generator self-qualification (no GPU)
~/tptt-venv/bin/python rnn_05b_ext2.py --selftest ../runs/rnn/RNN-05B-EXT2/generator_selftest.json
# full run (GPU; ~10 min) — trains/freezes/hashes backbones, BASE qualification, gate
~/tptt-venv/bin/python rnn_05b_ext2.py --run --outdir ../runs/rnn/RNN-05B-EXT2 \
      --artifacts ../runs/rnn/RNN-05B-EXT2/artifacts
# end-to-end pipeline smoke (forces qualification to exercise the MC/ablation/stats paths)
~/tptt-venv/bin/python rnn_05b_ext2.py --run --outdir /tmp/ext2_smoke --artifacts /tmp/ext2_smoke_art --smoke
```

## Artifact hashes
challengeGridSha256 `66ff24765d17c4fa95dcfcbaf4a7b374c66aa1ba52e507cd047b486ad512a9e5`; backbone/file SHAs in the
identity table above; ZIP SHA-256 printed in the final delivery block.

## Exactly one next recommendation
**PARK the synthetic dense Memory-Caching / historical-state-recovery line and DEFER the Qwen-GDN transplant gate.**
Two independent, pre-registered synthetic attempts (RNN-05B-EXT unstable-base; RNN-05B-EXT2 non-degrading
fixed-base) have failed to produce a *testable* graded-forgetting regime in the toy GDN/DN substrate. Do **not**
build EXT3. If the research question is still worth pursuing, open a **separate RNN-06 packet** that tests H3 on a
**real** recurrent LM (a genuinely capacity-limited setting where the final state provably loses information),
optionally comparing the recorded RESEARCH pointers — but that is new scope requiring its own pre-registration, not
a continuation of EXT.
