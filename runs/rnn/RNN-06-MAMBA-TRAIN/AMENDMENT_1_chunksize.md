# RNN-06 Train — AMENDMENT 1 (chunk_size), appended BEFORE substantive 06B outcomes

**Status:** protocol amendment permitted by train §24 ("If an ordinary engineering issue
requires a protocol amendment BEFORE substantive outcomes, append it explicitly and commit it
before continuing"). **No substantive 06B outcome existed when this was written** — the first
06B execution attempt raised `torch.OutOfMemoryError` during model forward and produced NO
results file. This amendment is committed before re-running.

## The engineering constraint (measured, not assumed)

The pinned transformers-native naive `Mamba2Mixer.torch_forward` materializes an intermediate
`G_intermediate` of shape `(b, c, l, s, h, n)` where `l = s = chunk_size`
(`modeling_mamba2.py:597`). Its memory therefore scales as **O(chunk_size²)**. At the pinned
`chunk_size=256`, for the 06B MQAR sequence lengths (up to 514 tokens at dose P=128), this
tensor demands **~45 GiB per sequence** (observed: a 90 GiB allocation attempt at batch=2),
which is infeasible on the 24 GiB RTX 3090 even at batch=1. This is the same reason
exploratory RNN-06-P0 ran at `chunk_size=32`.

## The amendment

**Pin `chunk_size = 32` for BOTH train items** (06A2 and 06B), superseding the initial
`chunk_size = 256`. Rationale and scope:

1. **Not a backend change.** The backend remains transformers-native `Mamba2ForCausalLM`,
   naive `torch_forward`, transformers 4.48.3, torch 2.6.0+cu124, bf16, no
   mamba_ssm/causal_conv1d (`is_fast_path_available=False`). Only the SSD chunk-tiling
   hyperparameter changes. Train §2's stop-condition is about the *backend* being impossible
   to reproduce; the backend reproduces fine — only a memory-tiling knob is adjusted.
2. **chunk_size is a numerical-tiling knob.** The chunked SSD scan is mathematically
   equivalent for any chunk_size (a tiling of the same recurrence); only bf16 rounding differs
   slightly. Historical `CHUNK_SIZE_IS_PART_OF_EXECUTION_IDENTITY=TRUE` records that the exact
   bf16 values differ with chunk_size — so this IS an execution-identity change and is
   recorded, not hidden. The macroscopic retrieval behaviour 06B measures (a graded band) is
   not sensitive to such tiny per-logit bf16 differences.
3. **Train-consistency preserved by re-qualification.** Because the dependency gate requires
   that the substrate 06B runs on is the substrate 06A2 qualified, **06A2 is re-qualified at
   `chunk_size=32`** (`CONTINUATION_RESULTS_cs32.json`, `DECISION_ADDENDUM_cs32.md`). The
   original `chunk_size=256` 06A2 qualification (committed `364147e`,
   `CONTINUATION_LIFECYCLE=QUALIFIED`) remains valid and is NOT deleted; it stands as
   additional evidence that the continuation contract holds across chunk sizes (expected,
   since the continuation/decode path is the single-token decode branch that does not use
   chunk_size, and the cache tensor structure is chunk-size-independent).
4. **Calibration continuity bonus.** P0's exploratory band was observed at `chunk_size=32`;
   running 06B at `chunk_size=32` makes P0's calibration directly comparable on the same
   numerical path (P0 examples remain quarantined; 06B uses the disjoint qualification set).

## What does NOT change

Model checkpoint/revision, backend implementation, dtype, quantization (none), task generator,
`qualificationSetSha256`, `stressGridSha256`, `lifecycleQualificationSetSha256`, all
preregistered thresholds (τ_hi, τ_lo, mid-band count, monotonicity tolerance, robustness,
confound separation), the primary endpoint, and every gate definition. Only the pinned
`chunk_size` value (256 → 32) and the consequent re-qualification of 06A2 change.

This amendment is committed together with the chunk-size-parameterized runners BEFORE any
06A2-cs32 or 06B outcome is produced.
