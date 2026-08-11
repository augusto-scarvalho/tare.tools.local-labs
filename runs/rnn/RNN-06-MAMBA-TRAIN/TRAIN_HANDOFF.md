# HANDOFF — RNN-06 Mamba Qualification Implementation Train

**Author:** Claude (Opus 4.8) via Claude Code. **Date:** 2026-08-11. **Branch:** `master`
(no upstream). **Pushed:** NO. Two backlog items, one hard dependency gate.

## Dependency status (at a glance)

```
RNN-06A2-MAMBA-CONTINUATION : CONTINUATION_LIFECYCLE   = QUALIFIED  (gate OPENED)
        │
        └─► RNN-06B-MAMBA-BASE : EXECUTED
                                 FIXED_BACKBONE_GRADED_REGION = BLOCKED (CONFOUNDED_WITH_LENGTH)
```

- **BACKLOG 1 — RNN-06A2:** status **QUALIFIED** (operative substrate cs=32; also QUALIFIED at
  cs=256). Gate opened ⇒ 06B was permitted to execute.
- **BACKLOG 2 — RNN-06B:** **EXECUTED** (not BLOCKED_BY_06A2). Own graded-region gate verdict
  = **BLOCKED**, sole reason `CONFOUNDED_WITH_LENGTH`. No historical-state test, no RNN-06C.

## HEAD boundary

- **before-train HEAD:** `0560c3da12491b2ac3b5fb69213d565cf747c1cb` (RNN-06A closure).
- **after-train HEAD:** `1ddf64a55a50046e66d25ceb5f5391569d3d1b71`.
- Tree clean of tracked changes. Nothing pushed. No amend/rebase of any historical commit.

## All commits (append-only; 0560c3d..HEAD)

| commit | boundary |
|---|---|
| `353de85` | train protocol + 06A2 pre-registration (no outcomes) |
| `5abc29b` | 06A2 lifecycle runner (identity-freezing; no outcomes) |
| `364147e` | 06A2 results → CONTINUATION_LIFECYCLE = QUALIFIED (cs=256) |
| `0404b5e` | 06A2 decision → QUALIFIED |
| `402cb82` | 06B pre-registration + qualification spec + stress grid (no outcomes) |
| `d6618b8` | 06B graded-region runner (no outcomes) |
| `bd701a4` | AMENDMENT 1: pin chunk_size=32 (memory); parameterize runners (no outcomes) |
| `fd8f863` | 06A2 re-qualified at cs=32 → QUALIFIED (operative substrate) |
| `6ccb54b` | 06B results → FIXED_BACKBONE_GRADED_REGION = BLOCKED (CONFOUNDED_WITH_LENGTH) |
| `1ddf64a` | 06B decision → BLOCKED |
| *(this handoff commit)* | train evidence + handoff + bundle builder |

## Exact source / model / backend identities

- Subject: `AntonV/mamba2-1.3b-hf` @ `703e19a43f397c70315244a3424d79456b54fb34`.
- Backend: transformers-native `Mamba2ForCausalLM`, transformers 4.48.3, torch 2.6.0+cu124,
  bf16, no mamba_ssm/causal_conv1d (`is_fast_path_available=False`), no quantization.
  `modeling_mamba2.py` sha256 `83685d78…`.
- **Pinned `chunk_size = 32`** (operative; AMENDMENT 1). Originally 256; changed BEFORE any
  substantive 06B outcome because the naive `torch_forward`'s `G_intermediate` is
  O(chunk_size²) → ~45 GiB/seq at MQAR lengths on 24 GiB (90 GiB OOM observed). Not a backend
  change (SSD tiling knob). 06A2 re-qualified at cs=32 to keep the gate valid.
- Executed-source PROVEN (runner blob == committed, dirty = ∅):
  06A2 runner `2bed403d` (HEAD `fd8f863` at cs=32 run) · 06B runner `ebe6bb96` (HEAD `fd8f863`).

## Pre-registration / challenge hashes

- `lifecycleQualificationSetSha256 = 72fa7f4962bd42633bbc093f81579255977af93aadcd0e506f5c97c5b8b6b8e4`
  (06A2; disjoint from RNN-06A exact sequences).
- `qualificationSetSha256 = e351a4449796cdf71fa04b3f77fd9038f6950cb2672d23968bb22a5e031cf0ee`
  (06B; example-level disjoint from P0 `calibrationSetSha256 = 779fb37a…`, 0 overlap / 200).
- `stressGridSha256 = d29e442f03e2cae116081a09712b354b320495ec3c046f270be9fa8e36e36de1`.

## RNN-06A2 — CURRENT × PROPOSED × RESULT

- **Question:** does the frozen subject expose an operational continuation/checkpoint contract
  (derived from 06B/06C needs) with continuation equivalence under the SAME algorithm and the
  generation frontier carried inside the snapshot?
- **Result:** QUALIFIED. 12/12 gate checks BIT_EXACT at cs=256 **and** at cs=32. A determinism,
  B/C/D greedy checkpoint-restore (incl. destroy→reload, frontier-from-snapshot-only), E branch
  + no contamination, F parent immutability, G neighbor-invariance (logits+state+continuation),
  H reset, I round-trip — all BIT_EXACT. Diagnostic P-alone-vs-in-batch `NOT_EQUIVALENT`
  (max_abs 1.0) but **non-gating** — batch-shape artifact not masqueraded as leakage. Fixes
  historical 06A's cross-algorithm (Claim B) and frontier (R5) defects by construction, not
  relaxation. Does NOT reclassify 06A (permanently NOT_QUALIFIED).

## RNN-06B — CURRENT × PROPOSED × RESULT (raw denominators exposed)

Primary endpoint = MP constrained retrieval accuracy. N=192 per (dose,condition), S=3 strata,
chance 1/256. Matched control LC = same length/probe-position/gap/binding-count, only ONE
scored-space association.

| P | MP corr/n | MP acc | LC acc | LC−MP | MP boot-95% |
|---:|---|---:|---:|---:|---|
| 8 | 159/192 | 0.828 | 0.880 | +0.052 | [0.771,0.880] |
| 16 | 131/192 | 0.682 | 0.703 | +0.021 | [0.615,0.750] |
| 24 | 119/192 | 0.620 | 0.589 | −0.031 | [0.552,0.688] |
| 32 | 98/192 | 0.510 | 0.479 | −0.031 | [0.438,0.578] |
| 48 | 72/192 | 0.375 | 0.380 | +0.005 | [0.302,0.443] |
| 64 | 57/192 | 0.297 | 0.271 | −0.026 | [0.229,0.365] |
| 96 | 40/192 | 0.208 | 0.203 | −0.005 | [0.151,0.266] |
| 128 | 25/192 | 0.130 | 0.130 | +0.000 | [0.083,0.177] |

**Gate (PRE_REGISTRATION §7):** competence ✅ (0.828≥0.75), material loss ✅ (0.130≤0.45),
≥2 mid-band ✅ (3: [16,24,32]), monotone ✅ (0 violations), robust ✅ (3/3 strata),
**confound-controlled ❌** — mean(LC−MP)@{96,128} = **−0.0026** (< +0.15). 5/6 pass; BLOCKED
solely by the confound control.

**PASS/FAIL summary:** 06A2 = QUALIFIED (all BIT_EXACT). 06B = BLOCKED (`CONFOUNDED_WITH_LENGTH`).

## Negative evidence (the finding)

The MQAR graded curve is real (competent, monotone, robust) but the matched LC control degrades
in lockstep with MP, so removing same-space competition does not rescue accuracy. The loss is
**generic sequence-length / recurrent-state-saturation degradation, not same-space associative
retrieval interference** — the exact confound RNN-06B was built to detect. P0 (no control)
called this same-shaped band `PLAUSIBLE`; the control falsifies its memory interpretation. P0
remains exploratory and unchanged.

## Deviations

- **AMENDMENT 1** (chunk_size 256→32) — the only protocol deviation; committed BEFORE any
  substantive 06B outcome (`bd701a4`); 06A2 re-qualified at cs=32; all thresholds/endpoints/
  gates unchanged. Documented in `RNN-06-MAMBA-TRAIN/AMENDMENT_1_chunksize.md`.
- 06A2 test J (stochastic frontier) recorded `NOT_APPLICABLE_BY_CONTRACT` (deterministic greedy
  contract) — scope deliberately not broadened.

## Authority / effect status

- `CONTINUATION_LIFECYCLE = QUALIFIED` (06A2) — new experiment; does NOT alter historical
  RNN-06A (`RNN-06A-MAMBA_STRICT_CONTRACT = NOT_QUALIFIED` permanent).
- `FIXED_BACKBONE_GRADED_REGION = BLOCKED` — the train's only mint of this gate; no QUALIFIED
  region minted anywhere. `GDN_COMPATIBILITY_GAP = OPEN`, `QWEN_GDN_TRANSPLANT_GATE = DEFER`
  (untouched). Historical 06A + P0 blobs verified immutable (`git_evidence.txt`).
- `state_bytes_per_sequence = 52,002,816` carried forward; 06C snapshot-cost estimate is
  informational only (06B BLOCKED).

## Confirmations

`NO_HISTORICAL_ARTIFACT_REWRITTEN = TRUE` · `NO_HISTORICAL_COMMIT_REWRITTEN = TRUE` ·
`FAILED_GATE_NOT_REINTERPRETED = TRUE` (BLOCKED preserved, not massaged to QUALIFIED) ·
`NO_SEED_SCREENING = TRUE` · `FROZEN_MODEL_INVARIANT_HELD = TRUE` · `NO_GDN = TRUE` ·
`NO_QWEN = TRUE` · `NO_MEMORY_CACHING = TRUE` · `NO_HISTORICAL_STATE_TEST = TRUE` ·
`NO_RNN_06C = TRUE` · `NO_SERVING_CHANGE = TRUE` · `NO_TRAINING = TRUE` · `NOTHING_PUSHED = TRUE`.

## Exactly one next recommendation (NOT executed)

**OPEN `RNN-06B2-MAMBA-BASE-LENGTH-DECONFOUNDED` in a NEW session** — a new, independently
preregistered confirmatory experiment that seeks a graded region whose degradation is
attributable to associative interference *after* removing the generic length/state-saturation
factor (e.g. hold sequence length AND write→query gap fixed while varying only the number of
same-space competitors via in-context substitution, or add a positive length-only control arm),
justified from RNN-06C requirements before observing outcomes. Do NOT repair GDN, run Qwen, test
historical state, implement Memory Caching, change the backend, or start RNN-06C in that work.

**STOP after this train. Do NOT start RNN-06C. Do NOT test historical-state information.**
