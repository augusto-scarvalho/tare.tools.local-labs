# RNN-06A2-MAMBA-CONTINUATION — PRE-REGISTRATION

**Written and committed BEFORE any outcome-bearing execution.** Defines the continuation
contract's equivalence classes, isolation criterion, tests A–J, challenge-set identity,
source-identity requirements, and the single gate. Thresholds are defined here, before
results, and are NOT inherited blindly from RNN-06A (justified below).

Companion: `CONTINUATION_CONTRACT.md` (derivation of the contract from RNN-06B/06C needs).
This experiment does NOT reclassify historical RNN-06A (permanently `NOT_QUALIFIED`).

## 1. Exact subject & backend (frozen; identical to train protocol)

`AntonV/mamba2-1.3b-hf` @ `703e19a43f397c70315244a3424d79456b54fb34`; transformers-native
`Mamba2ForCausalLM`; transformers 4.48.3; torch 2.6.0+cu124; **no** mamba_ssm / causal_conv1d
(`is_fast_path_available=False`); **bf16**; **no quantization**; **pinned `chunk_size=256`**
(native config default; the value 06A qualified). Backend is NOT changed for performance.

## 2. Contract objects (see CONTINUATION_CONTRACT.md §3)

- **Recurrent Cache State** = `conv_states ⊕ ssm_states ⊕ cache_position`.
- **Generation Frontier** (deterministic/greedy contract) = `{frontier_token :=
  argmax(last-step logits), next cache_position}`; the last-step logit vector is also
  serialized for verification. NO sampler config, NO RNG state (deterministic greedy — §5).
- **Execution Identity** = model+revision+backend+dtype+config(incl. chunk_size)+source
  hashes+runtime versions+runner blob/HEAD/dirty, recorded INTO results before outcomes.

## 3. Continuation algorithm (identical on both sides)

`greedy_continue(cache, frontier_token, start_cp, N)`: repeat N times — feed the current
token as a single-token input with `cache_position=[cp]` (cp>0 ⇒ decode branch), read
last-step logits, next token = argmax(logits), cp += 1; return the stacked per-step logits
`[N, V]` and the token sequence `[N]`. Reference and restored paths call the SAME function.
Teacher-forced variant `forced_continue(cache, frontier_token, tokens, start_cp)` feeds the
given `tokens` instead of argmax (used only for branch test E).

**Frontier sufficiency (fixes 06A R5 gap):** in tests B/C/D the restored path obtains
`frontier_token` ONLY from the serialized snapshot — never from a tensor held outside the
snapshot. Restoring from the snapshot alone must reproduce the reference.

## 4. Predeclared equivalence classes (thresholds fixed before results)

Comparisons are on float32-upcast logit tensors and on raw state tensors.
- **BIT_EXACT** — `torch.equal` (exact bitwise) on the compared tensor.
- **TOKEN_IDENTICAL** — argmax/token sequences identical (secondary channel).
- **NUMERICALLY_EQUIVALENT** — argmax 100% AND max_abs ≤ 2e-2 AND mean_abs ≤ 2e-3.
- **BOUNDED_DIFFERENCE** — argmax 100% AND max_abs ≤ 5e-1.
- **NOT_EQUIVALENT** — otherwise.
- **NOT_TESTABLE** — the substrate cannot run the test.

**Load-bearing bar = BIT_EXACT.** Justification (not inherited blindly): the restored path
resumes from a bitwise-identical recurrent state + frontier and runs an *identical*
algorithm, so exact reproduction is both achievable and necessary to certify the checkpoint
machinery; token-identity or bounded value-equivalence is NOT accepted for load-bearing
continuation/branch/isolation/round-trip tests. The `NUMERICALLY_EQUIVALENT`/
`BOUNDED_DIFFERENCE` classes exist ONLY for the **diagnostic** P-alone-vs-in-batch channel
(test G), which is explicitly non-gating. No post-hoc relaxation.

## 5. Deterministic/greedy scope

Downstream 06B/06C are deterministic greedy. Test **J (stochastic frontier / RNG-state)** is
therefore recorded `NOT_APPLICABLE_BY_CONTRACT` and NOT executed; scope is not broadened.

## 6. Isolation criterion (train §6)

**Primary = neighbor invariance** (load-bearing, BIT_EXACT): for fixed P row and fixed batch
shape, `[P,Q1]` vs `[P,Q2]` must leave P's last-token logits, P's per-row recurrent-state
slices, and P's restored greedy continuation all BIT_EXACT. **Diagnostic (non-gating) =**
P-alone vs P-in-batch (batch-shape GEMM difference expected; recorded, never fails the gate).

## 7. Challenge set (held-out; committed before outcomes)

`ops/rnn_06a2_challenges.py` (`generator_version = rnn06a2_continuation_challenges_v1`,
`master_seed = 20260812`) deterministically emits `CONTINUATION_CHALLENGES.json`:
token-id sequences drawn from vocab range `[106, 50100)` (avoids special/low ids),
`prefix_len = 16`, `boundaries = [4, 8, 12]`, `continuation_len = 6`. Contents: determinism
seqs (A), checkpoint seqs (B/C/D), a branch triple (E: prefix + two forced streams), and an
isolation triple (G: P, Q1, Q2). **`lifecycleQualificationSetSha256`** = SHA-256 over
{generator_version, master_seed, vocab range, prefix_len, boundaries, continuation_len, all
token sequences, boundary/split points, branch continuations}. **Disjoint from RNN-06A's
lifecycle sequences** (06A used fixed lists `[11,42,7,128,...]`, `P/Q/Q2` of length 8;
06A2 uses length-16 sequences from a distinct seeded generator) — disjointness recorded in
results. **No seed screening.**

## 8. Executed-source identity (before every outcome-bearing run)

Machine-readable results MUST contain, written before outcomes: runner SHA-256, runner git
blob, git HEAD, dirty-state indicator, model identity, backend source hashes
(`modeling_mamba2.py`, config), protocol/challenge-set hashes. No outcome-bearing run with
unidentified source bytes (assert `is_fast_path_available is False`).

## 9. Tests (each evaluated separately; no averaging across claims)

- **A. Fresh determinism** — prefill each determinism seq twice: logits BIT_EXACT AND state
  BIT_EXACT.
- **B. Checkpoint + frontier restore + continuation (greedy, in-memory)** — prefill to
  boundary t=8; snapshot {state, frontier_token, cp}; restore into a NEW cache from the
  snapshot ALONE; `greedy_continue` N=6; compare per-step logits + tokens to the uninterrupted
  `greedy_continue`. BIT_EXACT (all steps) required.
- **C. Serialize → destroy runtime → reload → restore → continue (greedy)** — snapshot to
  disk; `del model`; reload model (weights fingerprint must match); `torch.load`; restore;
  `greedy_continue` N=6 from snapshot ALONE; compare to the in-memory uninterrupted reference
  captured before destroy. BIT_EXACT required; reload weights match required.
- **D. Multiple checkpoint boundaries** — for one checkpoint seq, checkpoint at t ∈ {4,8,12};
  each restored greedy continuation BIT_EXACT vs its uninterrupted reference.
- **E. Branch replay** — from one snapshot S_t (t=8): restore→`forced_continue`(B1)→r1;
  fresh restore of S→`forced_continue`(B2)→r2. Independent refs via fresh prefill+forced. Gate:
  r1 BIT_EXACT ref1, r2 BIT_EXACT ref2, and no cross-branch contamination (run B1 then fresh
  restore + B2 == r2 BIT_EXACT).
- **F. Parent snapshot immutability** — snapshot hash unchanged before/after all E ops.
- **G. Neighbor request isolation** — primary neighbor invariance (P logits + P state slices +
  P restored greedy continuation) BIT_EXACT across `[P,Q1]` vs `[P,Q2]`; diagnostic
  P-alone-vs-in-batch recorded (non-gating).
- **H. Reset / fresh** — `cache.reset()` state == fresh cache state (exact, all-zeros); prefill
  after reset == fresh prefill (BIT_EXACT).
- **I. Serialization round-trip** — `torch.save`→`torch.load` of the state → exact; metadata
  (shape/dtype) preserved; byte accounting == 52,002,816.
- **J. Stochastic frontier** — `NOT_APPLICABLE_BY_CONTRACT` (not executed).
- **Weight immutability** — fingerprint before == after all ops; training mode off.
- **Full-module state** — 2 components (conv+ssm), byte accounting 52,002,816 (no
  partial-state false green).

## 10. Gate — mint exactly one `CONTINUATION_LIFECYCLE`

`QUALIFIED` iff ALL load-bearing checks pass at BIT_EXACT: A, B, C (incl. reload-weights),
D, E, F, G-neighbor-invariance, H, I, weight-immutability, full-module-state. Test J is
excluded (NOT_APPLICABLE_BY_CONTRACT); diagnostic G-alone-vs-batch is excluded.
- Any load-bearing BIT_EXACT failure → `NOT_QUALIFIED` (no relaxation; failures not averaged
  away — if isolation fails, the lifecycle fails).
- Substrate cannot run the contract → `NOT_TESTABLE`.

If `QUALIFIED` → RNN-06B may execute. If `NOT_QUALIFIED`/`NOT_TESTABLE` → `RNN-06B =
BLOCKED_BY_06A2`; persist negative evidence; NO outcome-bearing 06B; proceed only to
packaging/handoff.

## 11. Invariants

No GDN repair, no Qwen, no Memory Caching, no historical-state reader, no RNN-06C/06D, no
serving change, no training, no seed screening, no push. No `FIXED_BACKBONE_GRADED_REGION`
mint here (that is 06B's exclusive gate). Weights frozen; conclusions scoped to this exact
checkpoint/backend/config.
