# RNN-06A-MAMBA — Lifecycle Decision

**Verdict:** `FROZEN_BACKBONE_LIFECYCLE = NOT_QUALIFIED`
**Subject:** `AntonV/mamba2-1.3b-hf` @ `703e19a43f397c70315244a3424d79456b54fb34`
**Backend:** transformers 4.48.3 naive `Mamba2Mixer.torch_forward` (torch 2.6.0+cu124, bf16, no `mamba_ssm`/`causal_conv1d`; `is_fast_path_available=False`).
**Date:** 2026-08-11. **Decision commit chain:** protocol `2fd8b4e` → runner `8a02147` → results `7cc91f3` → this decision.
**Executed-source identity (self-recorded, P0 repair):** runner git blob `c64c2494…` (== committed, dirty=∅), `modeling_mamba2.py` sha256 `83685d78…04fdb`, config hash `41284e51…`. See `LIFECYCLE_RESULTS.json.executed_source_identity`.

---

## 1. Headline (read this first)

The verdict is a **conservative NOT_QUALIFIED driven by exactly one predeclared gate item
— claim B** (full-sequence-prefill vs segmented-prefill+decode). **It is NOT a
checkpoint/restore defect.** The lifecycle *machinery* — serialize → destroy runtime →
reload → restore → continue, branch, request-isolation, reset, full-module round-trip, and
determinism — is **BIT_EXACT** on this pinned backend. The blocker is a **bf16 cross-path
numerical property of the frozen base model**, with **100 % token-argmax agreement on every
tested split**. No threshold was altered after seeing results.

## 2. Gate matrix (predeclared PRE_REGISTRATION §7)

| Claim | Load-bearing result | Class / bool |
|---|---|---|
| A determinism (rerun logits + state) | PASS | `BIT_EXACT` / state exact |
| **B full vs segmented continuation** | **FAIL (gate blocker)** | a=2 `NOT_EQUIVALENT` (max_abs 0.625 > 0.5); a=5,8 `BOUNDED_DIFFERENCE`; **argmax 100 % all splits** |
| C serialize→destroy→reload→restore→continue | PASS | `BIT_EXACT`; reload weights match |
| D branch (vs independent; no contamination; parent unchanged) | PASS | `BIT_EXACT` × 3 + parent-unchanged |
| E request isolation (neighbor-invariance logits + per-row state) | PASS | `BIT_EXACT` + state exact |
| F reset == fresh; prefill-after-reset == prefill-fresh | PASS | state exact + `BIT_EXACT` |
| G full-module round-trip (2 components, byte accounting) | PASS | state exact + 52 002 816 B/seq |
| weights immutable (M1 fingerprint before==after) | PASS | true; training off |
| full-module (not partial) | PASS | conv **and** ssm, all 48 layers |

`QUALIFIED` required **all** load-bearing checks to hold. B did not → **NOT_QUALIFIED**.
Per the predeclared rule, failures are not averaged away.

## 3. What the lifecycle machinery proved (BIT_EXACT)

- **Full-module state is completely exposed and serializable.** The entire sequence-owned
  recurrent state is `{conv_states [48,B,4352,4], ssm_states [48,B,64,64,128]}` = **52 002 816
  bytes/sequence** (bf16), byte-accounting exact, component count 2, all 48 layers. No hidden
  per-sequence state; no position counter inside the cache (position is caller-supplied
  `cache_position`). This closes the RNN-05A partial-state gap for this backend: conv **and**
  ssm both round-trip.
- **Checkpoint/restore across a true runtime destroy is exact (C).** After `del model`,
  `del cache`, `empty_cache()`, reloading the model from the pinned revision, and restoring
  the serialized state from disk, the continuation logits are **bit-identical** to an
  in-memory continuation that never destroyed anything. Reloaded weights fingerprint-match
  the original.
- **Branch semantics are clean (D).** Two continuations from one saved state each equal their
  independent fresh-prefill reference bit-for-bit; running branch B1 does not perturb branch
  B2; the stored parent snapshot is byte-unchanged after branching.
- **Request isolation holds (E).** A sequence's row is **bit-identical** regardless of which
  neighbor shares its batch (neighbor-invariance logits `BIT_EXACT`, per-row state slices
  exact) — no cross-sequence leakage. (The weaker alone-vs-in-batch comparison is
  `BOUNDED_DIFFERENCE`, a benign batched-GEMM reduction-order effect, not leakage; it is not
  the load-bearing isolation criterion.)
- **Reset returns to the exact fresh-state contract (F)** (all zeros), and a prefill on a
  reset cache equals a prefill on a fresh cache bit-for-bit.
- **Round-trip (G)** preserves component count, names, shapes, dtypes, devices, byte counts,
  and exact values.
- **Determinism (A)** is bit-exact rerun-to-rerun; weights are immutable across all work
  (`training=False`, no optimizer, no `.backward()`).

## 4. The blocker, precisely (claim B)

Claim B compares a **single full-sequence prefill** (chunked naive-ssd path) against
**prefill(prefix) + token-by-token recurrent decode**. Measured (see `LIFECYCLE_MATRIX.csv`):

| split |A| | class | max_abs | mean_abs | rel_err | argmax |
|---|---|---|---|---|---|---|
| 2 | `NOT_EQUIVALENT` | 0.625 | 0.079 | 0.9 % | **100 %** |
| 5 | `BOUNDED_DIFFERENCE` | 0.375 | 0.046 | 1.4 % | **100 %** |
| 8 | `BOUNDED_DIFFERENCE` | 0.375 | 0.054 | 1.4 % | **100 %** |

Two facts make this a base-model numerics property, not a lifecycle defect:

1. **Token decisions never diverge** — argmax agreement is 100 % on all splits. The only
   difference is in logit *values* (≈1 % relative, on logits of magnitude ~26–66) in bf16.
2. **The divergence is dominated by the prefill-padding seam, not by decode.** The recorded
   `prefix_position_identity` (prefill(A) last logit vs full-sequence logit at the same
   causal position, **no decode involved**) has max_abs **equal to** the full comparison's
   max_abs (0.625 / 0.375 / 0.3125). i.e. `prefill(len 2)` and `prefill(len 12)` already
   differ by that amount at the shared position, purely because the naive path pads to the
   256-chunk with different pad sizes and the fp reductions regroup. The recurrent decode
   steps add essentially nothing beyond this base-model prefill-vs-prefill wobble.

So B measures **"can a single full-sequence prefill be reproduced value-for-value by
prefill+decode in bf16?"** — answer: **token-identical, not logit-identical**, and the gap is
a prefill/padding artifact. It does **not** measure checkpoint fidelity; that is claim C,
which is `BIT_EXACT`.

My predeclared `BOUNDED_DIFFERENCE` upper bound (max_abs ≤ 0.5) was set before data and one
split (a=2) landed at 0.625, tripping `NOT_EQUIVALENT` by the absolute-error edge despite
full argmax agreement. **This threshold was not relaxed post-hoc.** The conservative verdict
stands as computed.

## 5. Chunk-size identity (PRE_REGISTRATION §9)

`CHUNK_SIZE_IS_PART_OF_EXECUTION_IDENTITY = TRUE`. On a 20-token prefix, `chunk_size=8` vs
`chunk_size=256` gives `BOUNDED_DIFFERENCE` (max_abs 0.375, argmax 100 %). Chunk size changes
the numerics (not the token decisions) → it is part of the pinned execution identity. 06A's
lifecycle claims were run at the model-native `chunk_size=256`; no silent switch. (P0 used
`chunk_size=32` for feasibility of a long sweep; that is a *different numerical configuration*
and its equivalence claims do not transfer, consistent with the backend-change rule.)

## 6. Scope & caveats (preserved)

- Scoped to **this repo+revision+backend+versions+dtype+source-identity** only. Not
  generalizable to "Mamba-2", to the original `state-spaces` weights, or to a
  `mamba_ssm`/`causal_conv1d` kernel backend (a kernel backend would require independent
  re-qualification; equivalence claims do not transfer).
- `AntonV/mamba2-1.3b-hf` is an **unofficial HF-format conversion**; qualification is of this
  artifact as-is.
- Request isolation tested at **equal sequence length, no padding** (padding/attention-mask
  semantics were declared out of scope for 06A).
- All numerics are **bf16**; a token-match alone was never accepted as state equivalence —
  state tensors were compared directly (and are exact for C/D/E/F/G).
- This is a **lifecycle/semantics** verdict. It is **not** a memory-quality result. The P0
  observation stays `P0_GRADED_BAND = PLAUSIBLE_EXPLORATORY`; no `FIXED_BACKBONE_GRADED_REGION`
  is minted; no RNN-06B `qualificationSetSha256`; no historical-state test; no Memory Caching.

## 7. Boundaries unchanged

`GDN_COMPATIBILITY_GAP = OPEN` (not touched). `QWEN_GDN_TRANSPLANT_GATE = DEFER`. No Qwen, no
serving-infra change, no push. RNN-04/05A/05B/EXT/EXT2 and P0 evidence untouched (P0 result
blobs verified immutable at CURRENT reconstruction).

## 8. Exactly one next recommendation (NOT executed)

**Open a scoped `RNN-06A-EXT` (new session) that re-qualifies claim B alone under a
token-decision-preserving equivalence criterion and/or fp32 logit read-out**, holding the
same pinned bf16 backend for state I/O. Rationale: the checkpoint/restore/branch/isolation
machinery is already `BIT_EXACT` (C–G), and the only gate failure is a bf16 prefill-padding
numerical edge with 100 % token agreement; a criterion that scores continuation by token
decisions (argmax) — or reads logits in fp32 to remove the padding-reduction wobble — would
determine whether the frozen backbone qualifies for a *continuation* contract as opposed to a
*value-identical full-sequence-reproduction* contract. **Do not** repair GDN, run RNN-06B,
test historical state, or change backend as part of that. Alternatively, if a value-identical
full-sequence contract is required, the backend is `NOT_QUALIFIED` as-is and a kernel backend
would need its own qualification.
