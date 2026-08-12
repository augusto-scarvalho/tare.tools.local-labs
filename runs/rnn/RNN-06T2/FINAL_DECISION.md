# RNN-06T2-MAMBA-REQUALIFICATION — FINAL DECISION

Prospective remediation of the RNN-06T strict-preregistration defects on the **official** Mamba-2
fast path, plus fresh recovery confirmation and corrected apples-to-apples economics. Append-only
successor to RNN-06T (see `runs/rnn/RNN-06T/AUDIT_RECONCILIATION.md`). Nothing pushed.

**Substrate:** `state-spaces/mamba2-1.3b` @ `c5b59d00ec85d313adea86a08cad2a43c962dd3b`, official
`mamba_ssm` 2.2.4 / causal_conv1d 1.5.0.post8 / triton 3.2.0 / torch 2.6.0+cu124 / CUDA 12.4 / bf16 /
RTX 3090 / chunk_size 256. Qualification is conditional on this exact frozen checkpoint.

## All mints

| gate | verdict |
|---|---|
| `OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE` | **QUALIFIED** |
| `BATCH_SHAPE_NUMERICAL_PORTABILITY` | **OUT_OF_SCOPE_NOT_QUALIFIED** (preregistered) |
| `SINGLE_PASS_HISTORICAL_CAPTURE_T0R` | **QUALIFIED** |
| `HISTORICAL_RECOVERY_NARROW` | **QUALIFIED** |
| `ADAPTIVE_SELECTION_NARROW` | **DIRECTIONAL** (not required) |
| `WIDE_TARGET_RECOVERY_T1R` | **QUALIFIED** |
| `ADAPTIVE_SELECTION_T1R` | **QUALIFIED** |
| `END_TO_END_RECOVERY_UTILITY_T1R` | **QUALIFIED** |

## What was fixed relative to RNN-06T

1. **Property split.** `FIXED_BATCH_REQUEST_ISOLATION` (the operational contract) vs
   `BATCH_SHAPE_NUMERICAL_PORTABILITY` (batch1-vs-batchB, preregistered OUT_OF_SCOPE). The historical
   0.5 batch-shape divergence is preserved as negative/out-of-scope evidence and **not** re-labeled
   "benign." It never gates the lifecycle mint.
2. **Real branch/fork** — fresh per-branch reference reconstructed from the frozen parent + cross
   non-interference, with persisted hashes. No `... or True` tautology.
3. **Reset/REUSE equivalence** — a reset cache is reused for a continuation and compared BIT_EXACT to
   a genuinely fresh cache. (RNN-06T only checked "tensors became zero.")
4. **Serialization roundtrip + CONTINUATION** — continue after the roundtrip and compare to a
   no-roundtrip continuation. (RNN-06T only checked immediate hash equality.)
5. **Gate ordering honored** — T1R ran only after T0R qualified both required gates.
6. **Apples-to-apples economics** — every timed arm executes the same semantic task incl. the query
   readout; the primary utility comparator is `RECOVERY_ENABLED − FINAL_STEP` (the marginal cost of
   enabling recovery on the capture-capable path), gated on a robust p95 within a freshly
   preregistered 250 ms envelope (not the reused 1000 ms).

## Headline scientific result

On a real pretrained recurrent LM, at a fixed batch shape, the full single-pass historical-recovery
lifecycle is BIT_EXACT-qualified, and adaptive `MAX_CONFIDENCE` recovery over in-run snapshots beats
both FINAL (+0.542) and the strongest pre-committed fixed snapshot (+0.323) in the wide forgetting
regime — at a marginal serving cost well inside the preregistered envelope.

## Corrected economics (apples-to-apples)

Warm per-query (80 samples/arm, 2 process starts): FINAL_FUSED 37.7 ms · FINAL_STEP 976.7 ms ·
RECOVERY 1010.2 ms. Primary premium `RECOVERY − FINAL_STEP` = median +41.3 ms, **p95 +192.7 ms ≤ 250
ms** preregistered envelope → `END_TO_END_RECOVERY_UTILITY_T1R = QUALIFIED`. The true recovery
machinery is ~20 ms/q (restore+readout 12.2, GPU→CPU copy 8.4, selection 0.007); the shared 889 ms
trajectory is the capture-capable step path you run regardless. The descriptive `RECOVERY − FINAL_FUSED`
= 972 ms is the orthogonal step-vs-fused path cost, not the recovery premium — the RNN-06T conflation
of these two is now separated.

**Cross-process note (negative evidence):** in-process replay is BIT_EXACT; across process starts, bf16
kernel-autotuning flips ~1–3/192 borderline argmaxes (verdicts stable; margins ≫ noise).

## Scope / non-claims

Conditional on this frozen checkpoint. Token logits are not treated as independent replications. No
realistic-workload scout in this train (the old NL needle scout remains EXPLORATORY_NO_SIGNAL, neither
extended nor relied upon). No Qwen, no selector/reader training, no DART/StateX/SDM/GDN-2/INT8/
ReplaySSM, no host-policy change. Nothing pushed.
