# RNN-06 — Controlled State-Load Perturbation + Historical Information Train — PROTOCOL

**Author:** Claude (Opus 4.8) via Claude Code. **Started:** 2026-08-11. Written BEFORE any B3
or 06C outcome-bearing execution. Two backlog items, one hard dependency gate.

## Backlog items

- **Item 1 — `RNN-06B3-MAMBA-CONTROLLED-STATE-LOAD-PERTURBATION`**: does the frozen Mamba
  subject show a reproducible retrieval-loss perturbation from increasing UNIQUE recurrent-state
  load, AFTER eliminating (a) the full-packing boundary (U=M) and (b) the dose-dependent
  temporal-order churn found in audit? Mints `STATE_LOAD_FORGETTING_PERTURBATION ∈ {QUALIFIED |
  BLOCKED}` + descriptive `TRANSITION_SHAPE ∈ {GRADED | CLIFF | MIXED | FLAT}`. Smooth
  gradedness is descriptive, NOT the primary requirement.
- **Item 2 — `RNN-06C-MAMBA-HISTORICAL-INFO`** (CONDITIONAL on Item 1 QUALIFIED): is target
  information that becomes behaviorally unavailable after the qualified load perturbation still
  functionally accessible from an EARLIER recurrent state? INFORMATION PRESENCE only. Mints
  `HISTORICAL_STATE_INFORMATION ∈ {QUALIFIED | NOT_DETECTED | BLOCKED}`. **No recovery, no
  reader, no Memory Caching.**

## Hard dependency gate

```
RNN-06B3
   ├── QUALIFIED ─────────────→ RNN-06C may execute
   └── BLOCKED   ─────────────→ RNN-06C = BLOCKED_BY_06B3 (no readout); package & STOP. NO RNN-06B4.
```

## CURRENT (reconstructed from live Git at train start)

- HEAD `41ecfb785744d510d9514fa4467b16cf86d53e52` (prev state-load train evidence commit;
  expected prev endpoint `42e449e` is its ancestor), branch `master`, no upstream, tree clean.
- Prior gate artifacts verified immutable: P0 `d35db764`, 06A `d10527b6`, 06A2-cs32 `4c2dd568`,
  06B `22d355bd`, 06B2 `e1ca4261` / decision `5fa24408`.

## Preserved historical results (carried forward UNCHANGED; never rewritten)

- `RNN-06A-MAMBA_STRICT_CONTRACT = NOT_QUALIFIED`; `RNN-06A2 CONTINUATION_LIFECYCLE = QUALIFIED`
  (cs=32); `RNN-06B ORIGINAL CONTRACT = BLOCKED` (`CONFOUNDED_WITH_LENGTH` label historical).
- `RNN-06B2 FIXED_LENGTH_STATE_LOAD_REGION = BLOCKED` (reasons `IMMEDIATE_CLIFF`,
  `NOT_ROBUST_ACROSS_STRATA`); `RNN-06C = BLOCKED_BY_06B2` (not executed).

## Audit interpretation carried into this train (precise; historical artifacts NOT rewritten)

- `RNN-06B2 ORIGINAL GRADED CONTRACT = BLOCKED`.
- `RAW LENGTH-ONLY EXPLANATION = NOT_SUPPORTED` (length-only diagnostic was flat ~0.94).
- `SAME_SPACE_ASSOCIATIVE_INTERFERENCE = NOT_SUPPORTED` (DS ≈ SS).
- `UNIQUE_LOAD_EFFECT = DIRECTIONAL_SUPPORT_WITH_ORDER_CHURN_CONFOUND` — B2's `materialize`
  assigned load bindings by scanning active physical slots ascending, so as U grew the *ordinal→
  position→binding* mapping of already-active bindings shifted (temporal-order churn). B3 fixes
  this with a permanent ordinal↦slot↦binding map + nested-identity invariants.
- `FULL_PACKING_BOUNDARY = OPEN_CONFOUND` — B2's material loss appeared only at U=M (zero
  sentinels). B3 forbids U=M and enforces `MIN_SENTINEL_RESERVE`.
- `GENERAL_RECURRENT_STATE_SATURATION = NOT_YET_QUALIFIED`.

## Exact frozen substrate (both items; identical to the qualified subject)

`AntonV/mamba2-1.3b-hf` @ `703e19a43f397c70315244a3424d79456b54fb34`; transformers-native
`Mamba2ForCausalLM`; transformers 4.48.3; torch 2.6.0+cu124; bf16; no quantization; no
mamba_ssm/causal_conv1d (`is_fast_path_available=False`); **`chunk_size=32`**. No backend
change. No GDN repair. No Qwen.

## Global invariants

No seed/example screening · no threshold change after outcomes · no regeneration of
qualification examples after results · no crossing a failed gate · no recovery mechanism · no
reader training · no Memory Caching · no GDN repair · no Qwen · **no RNN-06B4** · no RNN-06D
here · nothing pushed · frozen model across all conditions. Bundle invariant enforced
(archive == manifest == SHA256SUMS payload; fail on unmanifested).

## Stop / pivot policy (§23)

- **B3 BLOCKED** ⇒ 06C not run; one next recommendation: PARK Mamba/MQAR H3 recipe, open a
  comparative memory-mechanism line. NO RNN-06B4, no further tuning iteration.
- **B3 QUALIFIED, 06C NOT_DETECTED/BLOCKED** ⇒ STOP; one next recommendation on parking
  historical-state recovery vs comparative line.
- **B3 QUALIFIED, 06C QUALIFIED** ⇒ STOP; one next recommendation: open RNN-06D recovery/utility
  train (NOT executed here).
