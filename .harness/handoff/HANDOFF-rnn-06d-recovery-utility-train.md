# HANDOFF — RNN-06D Historical-State Recovery Ceiling + Parameter-Free Utility Train

**Author:** Claude (Opus 4.8) via Claude Code. **Date:** 2026-08-12. **Branch:** `master` (no
upstream). **Pushed:** NO. Two backlog items, one hard dependency gate. **Both QUALIFIED.**

## Dependency status (at a glance)

```
BACKLOG 1  RNN-06D0 Recovery Ceiling & Snapshot Schedule : EXECUTED
                 RECOVERY_CEILING = QUALIFIED  (OB-FINAL = 0.776)
                       │  gate OPENED
                       ▼
BACKLOG 2  RNN-06D1 Target-Agnostic Parameter-Free Utility : EXECUTED
                 RECOVERY_UTILITY = QUALIFIED_PARAMETER_FREE  (best = MAX_CONFIDENCE)
```

## HEAD boundary

- **START HEAD (before train):** `139045f0f1f8e65a1f7b4045e163ea22d859576f`.
- **FINAL HEAD (this train):** `57f61d4` (see git_evidence.txt for the exact rev at packaging).
- Tree clean of tracked changes. Nothing pushed. No amend/rebase of outcome history.

## Commits (append-only; 139045f..HEAD)

`d75369c` train protocol + anti-oracle lib → `75e5940` calibration + **AMENDMENT 1** (v2
construction) + freeze K=4 → `62ebee0` D0 prereg + frozen qual identities + schedule → `eac2a1e` D0
ceiling runner → `d70f486` D0 results (QUALIFIED) → `de1a012` D0 decision → `7e9a97e` D1 prereg →
`8c7b21b` D1 runner → `f7f385b` D1 results (QUALIFIED_PARAMETER_FREE) → `57f61d4` D1 decision →
(evidence/handoff/bundle commit follows).

## Frozen subject + identities (both items)

`AntonV/mamba2-1.3b-hf` @ `703e19a43f397c70315244a3424d79456b54fb34`; transformers-native
`Mamba2ForCausalLM`; transformers 4.48.3; torch 2.6.0+cu124; bf16; no quantization; no
mamba_ssm/causal_conv1d (`is_fast_path_available=False`, naive `torch_forward`); `chunk_size=32`;
state 52,002,816 B/seq (remeasured). Executed-source PROVEN (dirty ∅): D0 ceiling blob `96d125cb…`,
D1 recovery blob `2a7c47e0…`, lib `ba93fff3…`. `calibrationSetSha256=0bf7d261…`,
`qualificationSetSha256=7bbf9b75…` (N=192, disjoint from all 8 prior sets),
`snapshotScheduleSha256=355e5105…` (K=4, slots [38,76,115,153]).

## Anti-oracle construction (v2; AMENDMENT 1)

Target-write slot t randomized in [8,64]; slots [0,t-1] = REPEAT1 sentinel (target cleanly encoded);
slot t = scored target; slots [t+1,191] = unique DS filler load (subsequent forgetting). Fixed 770
tokens. **v1 (all-load) was corrected during calibration BEFORE any outcome**: it saturated the
pre-target state so even the ORACLE_PROXIMAL snapshot could not retrieve the target (0.15–0.40 <
TAU_PROX 0.75) — the ceiling was not meaningfully testable. v2 isolates the quantity recovery must
exploit. Append-only, thresholds unchanged, qualification set fresh + disjoint. K=4 frozen for
anti-oracle validity (smallest adequate K whose schedule has a **pre-target distractor** snapshot,
slot 38, for t∈(38,64]); ceiling identical to K=2.

## BACKLOG 1 — RNN-06D0 — RECOVERY_CEILING = QUALIFIED

N=192, chance 1/256. FINAL (full state) **0.130** (degraded); ORACLE_TARGET_PROXIMAL **0.901**
(diagnostic, uses t); ORACLE_BEST_GOLD **0.906** (upper bound, uses gold). Pool per-slot acc
{38:0.490, 76:0.771, 115:0.458, 153:0.286} — slot 38 is a genuine pre-target distractor, slot 76 best,
graded post-target forgetting through 153. **ORACLE_BEST−FINAL = 0.776, CI [0.719, 0.833]**; robust
3/3; **recoverable 149/167 (0.89)** FINAL-wrong examples. Machinery: boundary self-check 16/16, 960
snapshots created/restored, 0 boundary failures, all counters > 0. All 8 gate criteria pass ⇒
QUALIFIED. It is a **ceiling** (oracle arms use privileged info; not deployable).

## BACKLOG 2 — RNN-06D1 — RECOVERY_UTILITY = QUALIFIED_PARAMETER_FREE

Independent re-capture, **bit-reproducible** vs D0 (`pool_logits`/`final_logits` max-abs-diff 0.0);
960 restored, 0 boundary failures. Target-agnostic, parameter-free methods (no gold, no target
position, no oracle identity):

| method | acc | Δ vs FINAL | recovered | harmed | oracle gap |
|---|---:|---:|---:|---:|---:|
| **MAX_CONFIDENCE** | **0.833** | **+0.703** [0.641,0.766] | 135 | 0 | 0.073 |
| MAX_TOP1_TOP2_MARGIN | 0.818 | +0.688 | 132 | 0 | 0.088 |
| CONFIDENCE_X_RECENCY | 0.802 | +0.672 | 129 | 0 | 0.104 |
| MIN_ENTROPY | 0.786 | +0.656 | 126 | 0 | 0.120 |
| LOGIT_ENSEMBLE | 0.510 | +0.380 | 76 | 3 | 0.396 |
| FINAL_PLUS_HISTORICAL | 0.474 | +0.344 | 69 | 3 | 0.432 |
| RECENCY | 0.286 | +0.156 | 35 | 5 | 0.620 |
| MATCHED_NO_HISTORY (control) | 0.130 | 0.000 | — | — | — |

Best = MAX_CONFIDENCE: Δ 0.703 (robust 3/3, CI_lb 0.641), 135 recovered / **0 harmed**, oracle gap
only 0.073, selection regret 0.081, cost_ok (intrinsic 36 ms/query, 208 MB for K=4). Controls make
it interpretable: **MATCHED_NO_HISTORY = FINAL exactly** (compute alone is worthless) and **RECENCY
is worst** (touching history is not enough — you must *select*). ⇒ QUALIFIED_PARAMETER_FREE.

## Economics / honest caveat

Intrinsic mechanism (restore + score + select over an already-captured pool) is cheap: restore
0.4 ms, restore+readout 36 ms/query, K×52 MB = 208 MB. Producing the pool costs K extra prefills
(~1.7 s/batch) **only because this naive backend has no multi-token mid-sequence forward** — a
substrate artifact; a fast-path/incremental capture folds capture into the single forward pass.
Reported transparently; excluded from the intrinsic-cost gate by pre-registration.

## Scientific arc

RNN-06C proved historical-state information *presence*. RNN-06D shows that presence is **exploitable**:
the target-agnostic historical pool contains the lost target (D0 ceiling 0.906 vs FINAL 0.130), and a
trivial parameter-free confidence selector captures 0.703 of the 0.776 ceiling with zero harm (D1).
No trained reader, no DART, no Memory Caching, no oracle info in any deployable arm.

## Confirmations

`NO_HISTORICAL_ARTIFACT_REWRITTEN=TRUE` · `NO_HISTORICAL_COMMIT_REWRITTEN=TRUE` ·
`THRESHOLDS_NOT_TUNED_AFTER_OUTCOMES=TRUE` · `NO_SEED_SCREENING=TRUE` · `FROZEN_MODEL_INVARIANT_HELD=TRUE` ·
`ANTI_ORACLE_NO_GOLD_OR_TARGET_POS_IN_DEPLOYABLE_ARMS=TRUE` · `RECAPTURE_BIT_REPRODUCIBLE=TRUE` ·
`SNAPSHOT_BOUNDARY_FAILURES=0` · `NO_READER_TRAINED=TRUE` · `NO_DART=TRUE` · `NO_MEMORY_CACHING=TRUE` ·
`NO_STATEX=TRUE` · `NO_GDN=TRUE` · `NO_QWEN=TRUE` · `NO_SERVING_CHANGE=TRUE` · `NO_WU_POLICY_CHANGE=TRUE` ·
`NOTHING_PUSHED=TRUE`.

## Exactly one next recommendation (NOT executed) — outcome QUALIFIED_PARAMETER_FREE

**OPEN an official-Mamba transportability replication in a NEW session** — reproduce the
parameter-free recovery utility on the fast-path (`mamba_ssm`) kernel and, if it holds, on a
non-synthetic long-context retrieval task, under independent audit and fresh pre-registration. Do NOT
implement it here; no reader / DART / Memory Caching / StateX / GDN repair / Qwen; nothing pushed.

**STOP. Do NOT start the next train here.**
