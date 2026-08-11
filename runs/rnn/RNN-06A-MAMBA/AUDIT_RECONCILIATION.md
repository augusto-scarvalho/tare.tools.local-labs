# RNN-06A-MAMBA — Audit Reconciliation (append-only closure)

**Nature:** interpretation reconciliation against the FROZEN pre-registration. **No** GPU
rerun, **no** measured-result change, **no** threshold change, **no** edit to
PRE_REGISTRATION / LIFECYCLE_RESULTS / LIFECYCLE_MATRIX / LIFECYCLE_DECISION. The historical
verdict is permanent: **`FROZEN_BACKBONE_LIFECYCLE = NOT_QUALIFIED`**.
**Date:** 2026-08-11. **Author:** Claude (Opus 4.8) via Claude Code.
**Authority:** this file is the authoritative interpretation and **supersedes** the
"sole blocker = Claim B" phrasing used in `LIFECYCLE_DECISION.md` and the prior session
summary (see §1). Historical artifacts are left byte-unchanged (append-only convention).

**Packet audit file:** `AUDIT_RECONCILIATION_RNN-06A-MAMBA_2026-08-11.md` was **not present
in the session uploads** (searched repo + temp/uploads). Requirements transcribed verbatim
from the request message. `PACKET_AUDIT_FILE = NOT_PRESENT_IN_UPLOADS` (precedent-consistent
with the RNN-06-P0 closure).

**Live-Git integrity at closure start:** HEAD `2b19d068`, branch `master`, no upstream, tree
clean of tracked changes. Historical blobs intact — `LIFECYCLE_RESULTS.json` `d10527b6`,
`PRE_REGISTRATION.md` `50c8a41d`, `LIFECYCLE_DECISION.md` `ff2dd32d`; P0 `P0_RESULTS_MAMBA2.json`
`d35db764` (immutable). RNN-04/05*/EXT/EXT2 untouched.

---

## 1. Claim E preregistration drift (the material correction)

The committed `PRE_REGISTRATION.md` defines Claim E's PRIMARY isolation test as **alone-vs-
in-batch** and predeclares it BIT_EXACT and load-bearing:
- §5 E: "Run each ALONE (batch=1) and TOGETHER (batch=2, rows [P;Q]). Verify P-alone logits
  == P-in-batch and Q-alone == Q-in-batch (no cross-row leakage). Also verify per-row state
  slices match."
- §6: "E: expect BIT_EXACT".
- §7 gate: "C, D, E, F, G each BIT_EXACT".

The executed results contain:
- `P_alone_vs_in_batch = BOUNDED_DIFFERENCE`, max_abs = 0.5 (argmax 100%).
- `Q_alone_vs_in_batch = BOUNDED_DIFFERENCE`, max_abs = 0.5 (argmax 100%).

The runner ADDITIONALLY introduced a **neighbor-invariance** test — P's row in batch `[P,Q]`
vs in batch `[P,Q2]` — found P-row logits **BIT_EXACT** and P-row recurrent state **BIT_EXACT**,
and used THAT (not the preregistered alone-vs-batch) as `E_ok`. Neighbor-invariance is a
better *causal* leakage test (it isolates the neighbor's data as the only varying input), but
**it was not the preregistered primary criterion.**

Append-only record (do not delete or discredit the neighbor-invariance result):
- `E_ORIGINAL_ALONE_VS_BATCH_CRITERION = NOT_MET_AS_BIT_EXACT` (it was BOUNDED_DIFFERENCE, a
  benign batched-GEMM reduction-order effect, not leakage).
- `E_NEIGHBOR_INVARIANCE = PASS_BIT_EXACT` (logits and per-row state).
- `E_PREREGISTERED_GATE_CONFORMANCE = DEVIATION` (the load-bearing E check was substituted for
  a non-preregistered, arguably stronger one).

**Consequence for wording:** the claim that the verdict's **"sole blocker = Claim B" is
incorrect** and is hereby corrected. Under the *literal* preregistered gate, **both** Claim B
**and** Claim E (alone-vs-batch BIT_EXACT) fail the strict BIT_EXACT bar. The overall verdict
is therefore *over-determined* NOT_QUALIFIED — it does **not** hinge on B alone. No measured
number changes; only the interpretation is corrected. Corrected phrasing to use going forward:
> "NOT_QUALIFIED under the strict preregistered gate, failed by Claim B (NOT_EQUIVALENT) and by
> Claim E's preregistered alone-vs-batch criterion (not met as BIT_EXACT); Claim E's
> neighbor-invariance leakage test passed BIT_EXACT as a non-preregistered substitute."

## 2. Claim B preserved exactly as failed

No threshold relaxed. Preserved permanently:
- `B split a2 = NOT_EQUIVALENT` because `max_abs = 0.625 > preregistered 0.5`.
- `B split a5 / a8 = BOUNDED_DIFFERENCE` (max_abs 0.375).
- Argmax = 100% on all splits is **secondary** evidence; it does **not** make the historical
  preregistered Claim B pass.
- `CLAIM_B_ORIGINAL_GATE = FAIL`.

## 3. Root-cause claim narrowed

Supported: `B_DIVERGENCE_EXISTS_BEFORE_DECODE_OR_RESTORE = SUPPORTED` — the recorded
`prefix_position_identity` already differs (max_abs 0.625/0.375/0.3125) between a short-prefix
prefill and the same causal position inside the longer prefill, i.e. before any decode/restore.

But the *exact* mechanism is NOT causally proven:
- `PREFILL_PADDING_OR_REDUCTION_ORDER_MECHANISM = PLAUSIBLE / NOT_CAUSALLY_PROVEN`.
- 06A did **not** run the dedicated ablation needed to distinguish chunk-padding vs
  tensor-contraction ordering vs chunk-implementation vs another length-dependent numerical
  mechanism. Prior wording that asserted the "prefill-padding seam" as established fact is
  downgraded to PLAUSIBLE. **No new experiment is run for closure.**

## 4. Recurrent-cache lifecycle ≠ complete generation checkpoint

Preserved strong positives (scoped to the recurrent tensors):
- `MAMBA2CACHE_FULL_RECURRENT_STATE = {conv_states, ssm_states}` (all 48 layers).
- `RECURRENT_CACHE_CHECKPOINT_RESTORE_ON_SEGMENTED_PATH = BIT_EXACT` (Claim C decode-from-
  restored-state == in-memory segmented reference; Claim D branches BIT_EXACT).

Scoping caveat (auditor-correct): Claim C retained `seed_last_C` — the prefix's final
prediction logit — from OUTSIDE the serialized recurrent snapshot and carried it across the
destroy/reload; branch testing likewise seeded from a recomputed prefix last-logit. The decode
steps that were compared are computed from the restored recurrent tensors (that part is
genuinely BIT_EXACT), but the **generation frontier was supplied externally**, not regenerated
from the recurrent tensors alone. Therefore:
- `COMPLETE_AUTOREGRESSIVE_GENERATION_CHECKPOINT = NOT_PROVEN_BY_06A`.

A future runtime resume contract may need, depending on the checkpoint boundary: the last
prediction/logits or the already-selected next token; sampler configuration; RNG state if
stochastic; and other generation-loop metadata. This does **not** invalidate the recurrent-
cache lifecycle evidence and does **not** block deterministic known-token continuation
experiments (which need only the recurrent tensors + the known next tokens).

## 5. Positive lifecycle evidence retained (not erased by the aggregate headline)

All of the following remain qualified sub-results:
- `EXECUTED_SOURCE_IDENTITY = PROVEN` (runner git blob `c64c2494…` == committed, dirty=∅;
  `modeling_mamba2.py` sha256 `83685d78…`; config hash `41284e51…`).
- Full recurrent-cache component inventory (2 components, 48 layers).
- Serialize/deserialize exactness (round-trip G exact).
- Runtime destroy→reload→restore→continue exactness vs the segmented in-memory reference (C).
- Branch BIT_EXACT (D); parent-snapshot immutability; no cross-branch contamination.
- Neighbor-invariance BIT_EXACT (E, non-preregistered substitute — see §1).
- Reset==fresh exact (F); recurrent-state round-trip exact (G).
- Weights immutable; no training (eval, no optimizer, no `.backward()`).

## 6. Measured state economics

Preserved: `state_bytes_per_sequence = 52,002,816` (≈ 49.59 MiB, bf16) — conv 1,671,168 +
ssm 50,331,648. Earlier research-era O(1–10 MB) "constant-size recurrent state" estimates are
**superseded for this exact candidate/backend** (the ssm state dominates at ~48 MiB/seq for
48 layers × 64 heads × 64 head_dim × 128 state). No snapshot-cadence redesign here; carry:
- `HISTORICAL_SNAPSHOT_ECONOMICS = REQUIRES_RECALCULATION_BEFORE_06C_06D`.

## 7. No post-hoc requalification of historical 06A

`RNN-06A-MAMBA_STRICT_CONTRACT = NOT_QUALIFIED` (permanent). The previously-suggested
"RNN-06A-EXT" whose purpose was to make historical Claim B pass under a token-decision/argmax
criterion is **withdrawn** — retro-fitting a looser criterion onto a frozen preregistration is
not permitted. If the downstream program decides that operational *continuation* semantics
(not value-identical full-prefill reproduction) are the correct contract, that must be a NEW,
independently preregistered experiment — `RNN-06A2-MAMBA-CONTINUATION` — using new
deterministic held-out lifecycle sequences and justifying its criteria from RNN-06B/C
requirements BEFORE observing outcomes. **Not executed in this session.**

## 8. Boundaries unchanged
`GDN_COMPATIBILITY_GAP = OPEN` (untouched). `QWEN_GDN_TRANSPLANT_GATE = DEFER`. No
`FIXED_BACKBONE_GRADED_REGION`, no RNN-06B `qualificationSetSha256`, no historical-state test,
no Memory Caching, no backend change, no Qwen, no serving change, nothing pushed. P0
`P0_GRADED_BAND = PLAUSIBLE_EXPLORATORY` unchanged.

## 9. Confirmations
`NO_GPU_RERUN = TRUE` · `NO_MEASURED_RESULT_MODIFIED = TRUE` · `NO_THRESHOLD_CHANGED = TRUE` ·
`NO_PREREGISTRATION_EDIT = TRUE` · `NO_HISTORICAL_ARTIFACT_REWRITTEN = TRUE` ·
`NO_HISTORICAL_COMMIT_REWRITTEN = TRUE` · `NOTHING_PUSHED = TRUE`.

## 10. Exactly one next recommendation (NOT executed)
**OPEN `RNN-06A2-MAMBA-CONTINUATION` IN A NEW SESSION** — a new, independently preregistered
experiment on the pinned bf16 backend that (a) defines the operational continuation contract
and its equivalence criterion from RNN-06B/C requirements before observing outcomes, (b) uses
new deterministic held-out lifecycle sequences, and (c) includes the generation-frontier
metadata needed for a complete autoregressive-generation checkpoint. Do NOT repair GDN, run
RNN-06B, test historical state, or change backend.
