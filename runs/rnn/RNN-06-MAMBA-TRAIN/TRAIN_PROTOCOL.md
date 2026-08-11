# RNN-06 — Mamba Qualification Implementation Train — TRAIN PROTOCOL

**Author:** Claude (Opus 4.8) via Claude Code. **Started:** 2026-08-11.
**Nature:** bounded two-item implementation train with a HARD dependency gate. This
document is written BEFORE any 06A2 or 06B outcome-bearing execution.

## Backlog items

- **Item 1 — `RNN-06A2-MAMBA-CONTINUATION`**: define and independently qualify the
  operational continuation/checkpoint contract actually required by downstream
  RNN-06B/06C, WITHOUT reclassifying or weakening historical RNN-06A-MAMBA. Mints
  exactly one `CONTINUATION_LIFECYCLE ∈ {QUALIFIED | NOT_QUALIFIED | NOT_TESTABLE}`.
- **Item 2 — `RNN-06B-MAMBA-BASE`**: IFF `CONTINUATION_LIFECYCLE = QUALIFIED`, run a
  preregistered confirmatory experiment on independent deterministic examples to decide
  whether a stable graded BASE retrieval-loss region exists on the exact frozen Mamba
  subject. Mints exactly one `FIXED_BACKBONE_GRADED_REGION ∈ {QUALIFIED | BLOCKED}`.

## Hard dependency gate

```
RNN-06A2
   ├── QUALIFIED ─────────────→ RNN-06B may execute
   └── NOT_QUALIFIED/NOT_TESTABLE → RNN-06B = BLOCKED_BY_06A2 (no outcome-bearing 06B)
```

A failed upstream gate BLOCKS dependent work. Failed evidence is never reinterpreted to
finish the backlog. RNN-06C and RNN-06D are OUT OF SCOPE for this train.

## CURRENT (reconstructed from live Git at train start)

- HEAD `0560c3da12491b2ac3b5fb69213d565cf747c1cb` (== historical RNN-06A closure), branch
  `master`, **no upstream**, tree clean of tracked changes (only untracked bundles/helpers).
- Historical evidence verified immutable (git blob SHAs): `LIFECYCLE_RESULTS.json`
  `d10527b6`, `PRE_REGISTRATION.md` `50c8a41d`, `LIFECYCLE_DECISION.md` `ff2dd32d`,
  `LIFECYCLE_MATRIX.csv` `26693353`, `MODEL_IDENTITY.json` `2058561a`,
  `machine_config.json` `98d0b6ac`; P0 `P0_RESULTS_MAMBA2.json` `d35db764`;
  `AUDIT_RECONCILIATION.md` `602d6170`.
- Historical results carried forward UNCHANGED: `RNN-06A-MAMBA_STRICT_CONTRACT =
  NOT_QUALIFIED`; `CLAIM_B_ORIGINAL_GATE = FAIL`;
  `E_ORIGINAL_ALONE_VS_BATCH_CRITERION = NOT_MET_AS_BIT_EXACT`;
  `E_NEIGHBOR_INVARIANCE = PASS_BIT_EXACT`;
  `RECURRENT_CACHE_CHECKPOINT_RESTORE_ON_SEGMENTED_PATH = BIT_EXACT`;
  `COMPLETE_AUTOREGRESSIVE_GENERATION_CHECKPOINT = NOT_PROVEN_BY_06A`;
  `state_bytes_per_sequence = 52,002,816`; `GDN_COMPATIBILITY_GAP = OPEN`;
  `QWEN_GDN_TRANSPLANT_GATE = DEFER`.

History is NOT modified to match expectations — CURRENT is resolved from live evidence and
matches the historical expectation exactly.

## Exact frozen subject (both items)

- Model `AntonV/mamba2-1.3b-hf` @ revision `703e19a43f397c70315244a3424d79456b54fb34`.
- Backend: transformers-native `Mamba2ForCausalLM`, transformers `4.48.3`, torch
  `2.6.0+cu124`, **no** `mamba_ssm`, **no** `causal_conv1d` (`is_fast_path_available=False`
  naive `torch_forward`), **bf16**, **no quantization**.
- **Pinned `chunk_size = 256`** (the model config's native default; the value RNN-06A
  lifecycle-qualified and for which `CHUNK_SIZE_IS_PART_OF_EXECUTION_IDENTITY = TRUE`). This
  single value is held constant across BOTH items. (Note: exploratory P0 used a
  `chunk_size=32` override; P0's band is therefore treated as approximate *pressure-range*
  calibration only, on a different numerical path — see 06B prereg §confound/§calibration.)
- If CURRENT made this exact substrate impossible to reproduce, the train would STOP and
  classify the substrate problem. It did NOT: substrate reproduced (RTX 3090, WSL2,
  `~/rnn06_env`, versions above verified live).

## Global train invariants (both items)

No GDN repair · no Qwen · no Memory Caching · no historical-state reader/probe · no RNN-06C ·
no RNN-06D · no serving changes · no training/fine-tuning of the Mamba backbone · no seed
screening · no push. Model remains frozen. Exact subject explicitly scoped — conclusions are
NOT generalized to "Mamba-2" beyond this exact checkpoint/backend/config.

## Git & evidence structure

Distinct commit boundaries (append-only; never amend outcome history):
train protocol/prereg → 06A2 impl → 06A2 results → 06A2 decision → [if QUALIFIED: 06B prereg
→ 06B impl → 06B results → 06B decision] → train evidence/handoff. Nothing pushed.

## Stop condition

After both items resolve under their gates and the final train bundle is attached: STOP. No
RNN-06C, no historical-state test, no recovery impl, no GDN repair, no Qwen, no push.
