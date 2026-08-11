# RNN-06A2-MAMBA-CONTINUATION — DECISION ADDENDUM (chunk_size=32 re-qualification)

Per train **AMENDMENT 1** (`RNN-06-MAMBA-TRAIN/AMENDMENT_1_chunksize.md`), the operative
train substrate is pinned to `chunk_size=32` (the original `chunk_size=256` is memory-
infeasible for downstream 06B's MQAR sequence lengths on 24 GiB). To keep the dependency gate
valid — 06B must run on the substrate 06A2 qualified — the continuation contract was
**re-qualified at `chunk_size=32`** on the identical held-out challenge set.

## Verdict (cs=32)

**`CONTINUATION_LIFECYCLE = QUALIFIED`** — all 12 preregistered gate checks BIT_EXACT,
identical to the cs=256 qualification. Evidence: `CONTINUATION_RESULTS_cs32.json`,
`CONTINUATION_MATRIX_cs32.csv`, `stdout_continuation_cs32.log`.

- `effective_chunk_size = [32, 32]`; executed-source PROVEN (runner blob `2bed403d` ==
  committed `bd701a4`, dirty = ∅); `is_fast_path_available=False`;
  `lifecycleQualificationSetSha256 = 72fa7f49…` re-verified; runtime 15.6 s, peak VRAM 4.27 GB.
- A determinism, B/C/D greedy checkpoint-restore (incl. destroy/reload, frontier-from-snapshot
  only), E branch + no contamination, F parent immutable, **G neighbor-invariance BIT_EXACT**,
  H reset, I round-trip — all BIT_EXACT. Diagnostic P-alone-vs-in-batch `NOT_EQUIVALENT`
  (non-gating), same as cs=256.

## Interpretation

The continuation contract holds identically at cs=32 and cs=256, as expected: the
continuation/decode path is the single-token decode branch (which does not use chunk_size at
all), and the recurrent-cache tensor structure ({conv_states, ssm_states}, 52,002,816 B/seq)
is chunk-size-independent. Both qualifications stand; the cs=32 one is the **operative**
qualification for the 06B dependency gate. Historical RNN-06A remains permanently
NOT_QUALIFIED and is not touched.
