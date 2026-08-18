# HANDOFF — RNN-06A-MAMBA (Frozen-Backbone Lifecycle Qualification)

**Packet:** RNN-06A-MAMBA. Follows CLOSED RNN-06-P0. **Author:** Claude (Opus 4.8) via Claude Code. **Date:** 2026-08-11.
**Verdict:** `FROZEN_BACKBONE_LIFECYCLE = NOT_QUALIFIED` (conservative; sole blocker = claim B; lifecycle machinery BIT_EXACT).

## HEAD / commit boundary
- **before HEAD (start):** `1b0cd2c297578c4a1f4a681c5fc2f0ddce66b170` (P0 closure).
- **after HEAD (this handoff written against):** `1ff0f56295b7c03566e395ba5575803ac924c733`.
- **branch:** `master` · **upstream:** none · **pushed:** NO.
- **06A append-only chain (no amend/rebase):**
  - `2fd8b4e` research — pre-registration + protocol (no outcomes)
  - `8a02147` tool — lifecycle runner (identity-freezing; no outcomes)
  - `7cc91f3` results — lifecycle qualification → NOT_QUALIFIED
  - `ddef963` decision — FROZEN_BACKBONE_LIFECYCLE = NOT_QUALIFIED
  - `1ff0f56` evidence — git evidence
- Working tree: only pre-existing untracked helpers (RNN-04/05*/08*/P0 bundles) + this handoff + the 06A bundle ZIP (untracked deliverables). No tracked changes pending.

## Exact subject / backend / library identity
- Model: `AntonV/mamba2-1.3b-hf` @ `703e19a43f397c70315244a3424d79456b54fb34` (HF cache refs/main == snapshots). Class `Mamba2ForCausalLM`, bf16, cuda:0.
- Backend: transformers **4.48.3** naive `Mamba2Mixer.torch_forward` (`is_fast_path_available=False`); torch **2.6.0+cu124**; CUDA 12.4; RTX 3090; **no** `mamba_ssm`, **no** `causal_conv1d`; no quantization.

## Executed-source identity (repairs P0 `PER_CANDIDATE_EXECUTED_SOURCE_IDENTITY = NOT_PROVEN`)
Self-recorded by the runner INTO `LIFECYCLE_RESULTS.json.executed_source_identity` BEFORE outcomes:
- runner `git_head` = `8a02147` (commit the runner ran from); **runner git blob `c64c2494…` == committed blob** (verified); **dirty = ∅** (clean).
- runner on-disk sha256 = `362120138c9ae890a210a17a9f340dce6dac17fb5bb5710468d66661d9327df6`.
- `modeling_mamba2.py` sha256 = `83685d785c0df6578fefca8d5a2ed382d70e651ffc83c00b8f40b64807a04fdb`.
- `configuration_mamba2.py` sha256 = `f9a3c2f5694b74b4a23ed7e23a6b2e20b2cc8a20532f65ab05091828fc9c4a1c`.
- config hash sha256 = `41284e51df350a07f67f131666cabefd2b8a5ed9bafbf846c12dcdb31833c887`.
This is a cryptographic binding of the exact bytes that executed. `06A_EXECUTED_SOURCE_IDENTITY = PROVEN`.

## Full state inventory (measured; full-module, not partial)
Complete sequence-owned recurrent state = 2 components, all 48 layers:
| component | shape | dtype | bytes/seq |
|---|---|---|---|
| conv_states | [48, B, 4352, 4] | bf16 | 1 671 168 |
| ssm_states | [48, B, 64, 64, 128] | bf16 | 50 331 648 |
| **total** | | | **52 002 816 (≈49.59 MiB)** |
Byte-accounting exact vs contract; no `seqlen_offset`/position inside the cache (caller-managed `cache_position`, only `[0]` read: `==0`⇒prefill, `>0`⇒single-token decode).

## Preregistered tolerances (unchanged post-hoc)
`BIT_EXACT` = torch.equal. `NUMERICALLY_EQUIVALENT` = argmax 100% & max_abs ≤ 2e-2 & mean_abs ≤ 2e-3. `BOUNDED_DIFFERENCE` = argmax 100% & max_abs ≤ 5e-1. `NOT_EQUIVALENT` = argmax<100% or max_abs>5e-1. `NOT_TESTABLE` = state not exposed / cannot (de)serialize / OOM.

## Lifecycle matrix (from LIFECYCLE_MATRIX.csv) + PASS/FAIL
Gate load-bearing checks: **8 PASS / 1 FAIL** ⇒ NOT_QUALIFIED.
- A determinism: logits `BIT_EXACT`, state exact — **PASS**
- **B full-vs-segmented: a2 `NOT_EQUIVALENT` (max_abs 0.625), a5/a8 `BOUNDED_DIFFERENCE` (0.375); argmax 100% ALL — FAIL (gate)**
- C serialize→destroy→reload→restore→continue: `BIT_EXACT`; reload weights match — **PASS**
- D branch1/branch2 vs independent `BIT_EXACT`; no-contamination `BIT_EXACT`; parent-unchanged true — **PASS**
- E neighbor-invariance logits `BIT_EXACT` + per-row state exact — **PASS** (alone-vs-batch `BOUNDED_DIFFERENCE` = benign batch-GEMM, not load-bearing)
- F reset==fresh exact; prefill-after-reset `BIT_EXACT` — **PASS**
- G round-trip state exact; byte-accounting ok (2 comps, 52 002 816 B) — **PASS**
- weights immutable (M1 fingerprint before==after), training off — **PASS**
- full-module (conv AND ssm, 48 layers) — **PASS**

## Source excerpts (file/function/line — full text in source_excerpts.md)
`transformers/models/mamba2/modeling_mamba2.py` (sha256 `83685d78…`):
- L133–207 `Mamba2Cache`: state = `conv_states` (L173), `ssm_states` (L181); in-place slice updates (L191–203); `reset()` zeroes both (L205–207). No position field.
- L465–568 `Mamba2Mixer.torch_forward`: **decode branch L478–567 is single-token** (L516 `dt=dt[:,0,:]`; L567 emits one token); **prefill branch L568+** = chunked naive-ssd; `cache_position[0]` is the prefill/decode selector (L478/L510/L615).
- L663–664 `forward`: `is_fast_path_available and cuda` ⇒ kernels; here False ⇒ `torch_forward`.
- L254 `self.chunk_size = config.chunk_size` (settable per mixer — chunk-size sub-experiment).

## Runner patch excerpt (the identity-freeze that repairs P0)
`ops/rnn_06a_mamba_lifecycle.py` (blob `c64c2494…`), `main()`:
```
identity = { "runner_source_sha256": sha256_file(runner_path),
             "runner_git_blob": git("hash-object", runner_path),
             "git_head": git("rev-parse","HEAD"),
             "modeling_mamba2_sha256": sha256_file(modeling_src), ... }
results["executed_source_identity"] = identity   # written BEFORE any outcome
```
snap/restore path (why C is BIT_EXACT): `snapshot()` `.clone().cpu()` both tensors;
`restore()` builds a fresh `Mamba2Cache` and `copy_()` bf16 bytes in (lossless).

## Command/output excerpts
- Run: `MSYS_NO_PATHCONV=1 wsl -d Ubuntu-24.04 -- bash -lc '~/rnn06_env/bin/python ops/rnn_06a_mamba_lifecycle.py'`
- stdout (`stdout_lifecycle.log`): `is_fast_path_available: False` … gate: A T,B F,C T,D T,E T,F T,G T,imm T,full T … `FROZEN_BACKBONE_LIFECYCLE = NOT_QUALIFIED` … runtime 30.4s, peak_vram 10.23 GB.
- Latencies (s): serialize 0.79, load 0.76, restore 0.006, continue 0.21, reload_model 1.05. Snapshot on disk 52 004 396 B (sha `f0e4d586…`).

## Negative / caveat evidence (preserved, not deleted)
- **B blocker**: prefill-vs-decode (and even prefill-vs-prefill at the seam) differ ≈1% in bf16 logits; `prefix_position_identity` max_abs == compare max_abs ⇒ divergence dominated by prefill-padding seam, NOT decode; **argmax 100% everywhere** (token decisions never diverge).
- **`CHUNK_SIZE_IS_PART_OF_EXECUTION_IDENTITY = TRUE`**: cs8 vs cs256 on 20-token prefix = `BOUNDED_DIFFERENCE` (max_abs 0.375, argmax 100%). Lifecycle claims run at native cs=256.
- E alone-vs-batch `BOUNDED_DIFFERENCE` (batched-GEMM reduction order; not leakage — neighbor-invariance is BIT_EXACT).
- Isolation tested equal-length/no-padding only (padding out of scope). All bf16.

## Authority / effect status
- `06A_EXECUTED_SOURCE_IDENTITY = PROVEN`. `FROZEN_BACKBONE_LIFECYCLE = NOT_QUALIFIED` (checkpoint/restore/branch/isolation/reset/round-trip machinery BIT_EXACT; blocked only by base-model bf16 cross-path numerics in claim B).
- Not minted: `FIXED_BACKBONE_GRADED_REGION` (none), RNN-06B `qualificationSetSha256` (none). P0 `P0_GRADED_BAND = PLAUSIBLE_EXPLORATORY` unchanged. `GDN_COMPATIBILITY_GAP = OPEN`. `QWEN_GDN_TRANSPLANT_GATE = DEFER`.
- P0 result blobs verified immutable (`d35db764…` / `54f1a161…`). RNN-04/05*/EXT/EXT2 untouched.

## Confirmations
- `NO_TRAINING = TRUE` (eval, no optimizer, no `.backward()`; weights immutable M1 before==after; reload fingerprint matches). `NO_WEIGHT_MUTATION = TRUE`.
- `NO_HISTORICAL_STATE_TEST = TRUE` · `NO_MEMORY_CACHING = TRUE` · `NO_GDN_REPAIR = TRUE` · `NO_QWEN = TRUE` · `NO_SERVING_CHANGE = TRUE` · `NO_06B = TRUE`.
- `NOTHING_PUSHED = TRUE` (no upstream). No threshold changed after seeing results.

## Exactly one next recommendation (NOT executed)
**Open `RNN-06A-EXT` in a NEW session**: re-qualify claim B **alone** under a token-decision-preserving criterion (argmax) and/or an fp32 logit read-out on the same pinned bf16 state I/O backend, to decide whether the frozen backbone qualifies for a *continuation* contract (already BIT_EXACT for checkpoint via C) vs a *value-identical full-sequence-reproduction* contract. Do NOT repair GDN, run 06B, test historical state, or change backend.

**STOP after this handoff/bundle. Do NOT start RNN-06B.**
