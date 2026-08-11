# RNN-06 — Fixed-Length State Load + Historical Information Train — TRAIN PROTOCOL

**Author:** Claude (Opus 4.8) via Claude Code. **Started:** 2026-08-11. Written BEFORE any B2
or 06C outcome-bearing execution. Two backlog items, one hard dependency gate.

## Backlog items

- **Item 1 — `RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD`**: does the frozen qualified Mamba
  subject show stable graded degradation as UNIQUE associative / recurrent-state load
  increases while **total sequence length and target→query gap are held constant**? Mints
  `FIXED_LENGTH_STATE_LOAD_REGION ∈ {QUALIFIED | BLOCKED}`.
- **Item 2 — `RNN-06C-MAMBA-HISTORICAL-INFO`** (CONDITIONAL on Item 1 QUALIFIED): is target
  information that is no longer behaviorally available from the FINAL recurrent state still
  functionally readable from an EARLIER recurrent state? Mints `HISTORICAL_STATE_INFORMATION ∈
  {QUALIFIED | NOT_DETECTED | BLOCKED}`. **No recovery mechanism is built; no reader trained.**

## Hard dependency gate

```
RNN-06B2
   ├── QUALIFIED ─────────────→ RNN-06C may execute
   └── BLOCKED   ─────────────→ RNN-06C = BLOCKED_BY_06B2 (no outcome-bearing 06C); package & STOP
```

## CURRENT (reconstructed from live Git at train start)

- HEAD `79e0dc5c3c2832555b6a9a8f9794f7805d5f06dd` (previous train's evidence commit, directly
  atop the expected `1ddf64a` 06B decision), branch `master`, **no upstream**, tree clean of
  tracked changes.
- Prior gate artifacts verified immutable (git blobs): P0 `d35db764`; 06A `LIFECYCLE_RESULTS`
  `d10527b6` / decision `ff2dd32d`; 06A2 `CONTINUATION_RESULTS_cs32` `4c2dd568`; 06B
  `BASE_RESULTS` `22d355bd` / decision `da9f723d`.

## Preserved prior results (carried forward UNCHANGED)

- `RNN-06A-MAMBA_STRICT_CONTRACT = NOT_QUALIFIED`.
- `RNN-06A2 CONTINUATION_LIFECYCLE = QUALIFIED` on the exact pinned **cs=32** substrate.
- `RNN-06B ORIGINAL CONTRACT = BLOCKED`; machine label `CONFOUNDED_WITH_LENGTH` is historical
  and is NOT rewritten.
- `SAME_SPACE_ASSOCIATIVE_INTERFERENCE = NOT_SUPPORTED`.
- `GENERAL_STATE_LOAD_FORGETTING = OPEN`.
- `LENGTH_VS_STATE_LOAD = NOT_DISAMBIGUATED`.
- `GDN_COMPATIBILITY_GAP = OPEN`; `QWEN_GDN_TRANSPLANT_GATE = DEFER`.

## §21 audit correction carried into this train

RNN-06B's persisted machine label `CONFOUNDED_WITH_LENGTH` remains historical and is not
rewritten. The **precise scientific interpretation** used going forward: RNN-06B showed
**same-space competition is NOT SUPPORTED** as the driver (LC ≈ MP), BUT its LC arm still
contained the SAME total number of structurally valid `key=value` bindings — so it removed
same-space competition but did **NOT** disambiguate **sequence length** from **general
unique-binding / state load**. We do **not** claim RNN-06B falsified memory broadly. B2 is
built precisely to disambiguate length from general unique-binding/state load.

## Global subject (both items; frozen, identical to the qualified substrate)

`AntonV/mamba2-1.3b-hf` @ `703e19a43f397c70315244a3424d79456b54fb34`; transformers-native
`Mamba2ForCausalLM`; transformers 4.48.3; torch 2.6.0+cu124; bf16; no quantization; no
mamba_ssm; no causal_conv1d (`is_fast_path_available=False`); **`chunk_size = 32`**. Backend
NOT changed. No GDN repair. No Qwen.

## Global invariants

No seed screening · no threshold change after outcomes · no silent regeneration of
qualification examples · no advance past a failed gate · no recovery mechanism · no Memory
Caching · no GDN repair · no Qwen · no RNN-06D · nothing pushed · frozen model across all
conditions. Engineering amendments (if needed before substantive outcomes) are appended +
committed, original protocol preserved.

## Bundle invariant (fix carried from §22)

The train ZIP must satisfy: **archive payload files == manifest payload files == SHA256SUMS
payload files**, excluding ONLY `TRAIN_MANIFEST.json` and `SHA256SUMS.txt` by explicit rule.
Bundle construction MUST fail if any unmanifested payload is present (the previous train's
`BLOCKED.json` was unmanifested — fixed here).

## Stop condition

After both items resolve under their gates and the bundle is attached: STOP. No RNN-06D, no
Memory Caching, no GDN repair, no Qwen, no push.
