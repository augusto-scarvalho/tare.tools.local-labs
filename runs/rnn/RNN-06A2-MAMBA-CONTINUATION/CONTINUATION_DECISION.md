# RNN-06A2-MAMBA-CONTINUATION — DECISION

## Verdict

**`CONTINUATION_LIFECYCLE = QUALIFIED`** (12/12 preregistered gate checks pass at BIT_EXACT).

Exact subject: `AntonV/mamba2-1.3b-hf` @ `703e19a43f397c70315244a3424d79456b54fb34`,
transformers-native naive `torch_forward` (transformers 4.48.3, torch 2.6.0+cu124, bf16, no
mamba_ssm/causal_conv1d, `is_fast_path_available=False`), **pinned `chunk_size=256`**. Verdict
is scoped to this exact checkpoint/backend/config; it is NOT generalized to "Mamba-2".

## Executed-source identity (PROVEN)

Runner git blob `90cd9fd995103ed16c541ea591b8ca104db1013f` == committed `5abc29b`; dirty = ∅;
`modeling_mamba2.py` sha256 `83685d78…`; `is_fast_path_available=False`;
`lifecycleQualificationSetSha256 = 72fa7f49…` re-verified (canonical, EOL-independent);
challenge set disjoint from RNN-06A's exact sequences. Runtime 31.7 s, peak VRAM 9.81 GB.

## What was qualified (and why it differs from historical 06A)

The contract qualifies **continuation equivalence** under the SAME continuation algorithm on
both sides, with the **generation frontier carried inside the serialized snapshot**:

| Test | Property | Result |
|---|---|---|
| A | fresh determinism (logits + state) | BIT_EXACT |
| B | checkpoint + frontier restore + greedy continuation (in-memory) | BIT_EXACT, tokens identical |
| C | serialize → **destroy runtime** → reload → restore → greedy continue | BIT_EXACT, tokens identical, reload weights match |
| D | multiple checkpoint boundaries t∈{4,8,12} | BIT_EXACT (all) |
| E | branch replay (2 forced streams) + no cross-branch contamination | BIT_EXACT (all) |
| F | parent snapshot immutability | unchanged |
| G | **neighbor invariance** (P logits + P state slices + P restored continuation) | BIT_EXACT |
| H | reset==fresh (zeros) + prefill-after-reset==fresh | BIT_EXACT |
| I | state serialization round-trip + byte accounting (52,002,816) | exact |
| — | weight immutability (pre-destroy AND across reload) | invariant |
| — | full-module state (conv+ssm, 2 components) | 52,002,816 B/seq |
| J | stochastic frontier / RNG-state | NOT_APPLICABLE_BY_CONTRACT (deterministic greedy) |

**Two historical 06A defects are resolved by construction, not by relaxation:**
1. **Cross-algorithm error (06A Claim B).** 06A compared single full-prefill vs
   prefill+segmented-decode — different numerical algorithms — and the bf16 values differed.
   06A2 compares the SAME greedy algorithm on both sides; the restored path resumes from a
   bit-exact state and reproduces the reference exactly. This is the operationally relevant
   property for RNN-06B/06C.
2. **Frontier sufficiency (06A R5 gap).** 06A seeded the continuation frontier from OUTSIDE
   the serialized snapshot (`COMPLETE_AUTOREGRESSIVE_GENERATION_CHECKPOINT=NOT_PROVEN_BY_06A`).
   06A2 carries `{frontier_token, next_cache_position}` INSIDE the snapshot and restores from
   the snapshot alone; greedy generation reproduces the uninterrupted run bit-for-bit — the
   frontier is demonstrably sufficient for deterministic greedy continuation.

## The isolation criterion decision (validated empirically)

The preregistered **primary** isolation test is **neighbor invariance** (`[P,Q1]` vs
`[P,Q2]`), which is BIT_EXACT. The **diagnostic** P-alone-vs-in-batch is `NOT_EQUIVALENT`
(max_abs 1.0) — a batch-shape GEMM/reduction-order artifact, exactly the effect that
(incorrectly) sank 06A's Claim E when used as a primary test. Because it is preregistered
**non-gating**, it does not fail the gate: a numerical-path artifact is not masqueraded as
request leakage. This directly confirms the property RNN-06B's batched scoring depends on —
changing another request does NOT change mine.

## This is NOT a reclassification of RNN-06A

Historical `RNN-06A-MAMBA_STRICT_CONTRACT = NOT_QUALIFIED` remains permanent and unchanged.
06A2 is a NEW experiment on NEW held-out data answering a different question (the operational
continuation contract for downstream work), preregistered before outcomes. No 06A artifact,
threshold, or verdict was altered.

## Downstream effect

`CONTINUATION_LIFECYCLE = QUALIFIED` ⇒ the dependency gate opens: **RNN-06B-MAMBA-BASE may
execute**. The frozen subject satisfies R1 (determinism), R2 (request isolation under
batching), R3 (frozen weights) required by 06B, plus R4–R6 (checkpoint/restore/branch,
frontier sufficiency) anticipated for RNN-06C (out of scope here). No
`FIXED_BACKBONE_GRADED_REGION` is minted by 06A2 (that is 06B's exclusive gate).
