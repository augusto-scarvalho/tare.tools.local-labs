# Published Independent Audit — RNN-06T

**Document ID:** `experiment.local-llm.rnn-06t.audit-reconciliation-2026-08-12`  
**Document type:** `experiment`  
**Status:** `EXPERIMENTAL`  
**Authority:** independent audit/reconciliation of an experimental train; no canonical architecture authority.

> The text below is preserved from the independent audit artifact. Its SHA-256 is recorded in `DOCUMENT_MANIFEST.json`.

---

# AUDIT RECONCILIATION — RNN-06T Official Mamba Transportability + Single-Pass Historical Recovery
Date: 2026-08-12
Auditor: independent ChatGPT audit

## Overall verdict

RNN_06T_TRAIN = ACCEPTED_AS_EXPLORATORY_WITH_MAJOR_GATE_RECONCILIATION

BUNDLE_INTEGRITY = PASS

OFFICIAL_MAMBA_FASTPATH = RUNNABLE_SUPPORTED

SINGLE_PASS_HISTORICAL_CAPTURE = SUPPORTED_STRONGLY

OFFICIAL_MAMBA_LIFECYCLE_STRICT_PREREG =
NOT_QUALIFIED

T0_REVISED_FIXED_BATCH_LIFECYCLE =
NOT_CONFIRMATORILY_QUALIFIED

PROTOCOL_GATE_ORDERING =
FAILED

HISTORICAL_RECOVERY_TRANSPORT_3A =
EXPLORATORY_STRONG_POSITIVE

ADAPTIVE_SELECTOR_ADVANTAGE_3A =
EXPLORATORY_DIRECTIONAL

WIDE_TARGET_RECOVERY_3B =
EXPLORATORY_STRONG_POSITIVE

ADAPTIVE_SELECTION_3B =
EXPLORATORY_STRONG_POSITIVE

END_TO_END_RECOVERY_UTILITY =
NOT_QUALIFIED_AS_APPLES_TO_APPLES

NL_NEEDLE_SCOUT =
EXPLORATORY_NO_SIGNAL

QWEN_TRANSFER =
DEFER

No old 06D result is reclassified by this audit.

The major issue is not that the downstream observations are false. The issue is that the train's hard T0 dependency gate was not prospectively satisfied under the frozen T0 preregistration. Therefore downstream outcomes cannot carry the confirmatory authority claimed in the handoff.

---

## 1. Bundle integrity

Attached archive:
RNN-06T-MAMBA-OFFICIAL-audit-bundle.zip

ZIP SHA-256:
cf4ecc0b02452dbba6e184b067f61003553139e8e06d99148b82ae85a134d370

External handoff SHA-256:
fed83cd4b07a6efdcc45c75a0fe8033527def614c08709429352cf3cea8c206b

Checks:
- 44 ZIP entries total
- CRC/testzip PASS
- 42 payload entries
- TRAIN_MANIFEST.json + SHA256SUMS.txt metadata
- archive payload set == manifest payload set
- manifest payload set == SHA256SUMS payload set
- 42/42 manifest sizes PASS
- 42/42 manifest SHA-256 PASS
- 42/42 SHA256SUMS PASS
- skipped_missing = []
- external handoff is byte-identical to internal HANDOFF.md

BUNDLE_INTEGRITY = PASS

---

## 2. Strong results that survive the reconciliation

### 2.1 Official fast path

The packet gives a coherent pinned execution identity:

state-spaces/mamba2-1.3b
revision c5b59d00ec85d313adea86a08cad2a43c962dd3b
mamba_ssm 2.2.4
causal_conv1d 1.5.0.post8
triton 3.2.0
torch 2.6.0+cu124
bf16
RTX 3090

The instrumented runner records nonzero official prefill/step kernel calls and no fallback path.

Correct scope:

OFFICIAL_MAMBA_FASTPATH = RUNNABLE_SUPPORTED

This does not imply all lifecycle properties are qualified.

### 2.2 Genuine in-run historical capture

The shared library's canonical path:
- prefills the first slot;
- advances subsequent tokens with the recurrent step path;
- clones actual cache state at declared boundaries;
- continues the same run to FINAL.

The T0 single-pass test separately reruns the same deterministic trajectory and compares state hashes at each boundary.

Observed:
- one trajectory;
- monotonic boundaries;
- captured state hashes equal independent same-path replay hashes;
- restore/readout from historical states succeeds;
- official step kernels fire.

This closes the RNN-06D "historical states were reconstructed by re-prefill" limitation at the implementation-mechanics level.

Correct classification:

SINGLE_PASS_HISTORICAL_CAPTURE = SUPPORTED_STRONGLY

The later 3A/3B `snapshotBoundaryChecks` counters are not themselves proof: in the final source they are incremented without performing a per-example boundary hash check. The proof comes from T0 plus the code structure of `run_trajectory`, not from those counters.

---

## 3. Finding A — the frozen T0 D/G preregistration was changed after observing a failure

This is the load-bearing audit finding.

T0_PRE_REGISTRATION.md states:

D neighbor/request isolation:
"a sequence's captured boundary states and its readout are invariant to which other sequences share its batch. Primary: readout argmax-invariant AND state max-abs-diff <= TOL_BATCH = 3e-2 vs run-alone."

G batch slice ownership:
"row i's captured state BIT_EXACT vs example i run alone at batch 1 (with the same TOL_BATCH fallback recorded as in D)."

The preregistered T0 gate states:
- G is among the checks required BIT_EXACT;
- D must remain within TOL_BATCH with argmax invariance.

The first implementation then observed:

batch1 vs batch6 state max_abs_diff = 0.5

0.5 > 0.03

The implementation was subsequently changed in commit:
3e48b41
"fix(rnn): RNN-06T T0 D/G test measures preregistered neighbor-isolation (not batch-size)"

The final runner redefines D/G around:
- fixed-batch row permutation;
- fixed-batch neighbor-content replacement.

Those are useful tests and they pass BIT_EXACT.

But they are NOT the complete frozen preregistered D/G criteria.

The handoff statement:

"the preregistered property is neighbor identity at fixed batch"

is inconsistent with the actual T0_PRE_REGISTRATION.md, which explicitly says "vs run-alone" and G explicitly says "batch 1".

Therefore:

T0_D_ORIGINAL_PREREG =
FAIL
(max_abs 0.5 > 0.03)

T0_G_ORIGINAL_PREREG =
NOT_MET_AS_BIT_EXACT_VS_BATCH1

T0_DG_FIXED_BATCH_REVISED_TEST =
POSITIVE_OBSERVATION_AFTER_FAILURE

T0_PROTOCOL_CONFORMANCE =
DEVIATION_AFTER_OUTCOME

This is a post-outcome change to a load-bearing gate, even though the result artifact itself was committed only later.

"Not committed yet" is not equivalent to "no outcome observed yet."

---

## 4. Finding B — C, E and F do not fully implement their frozen preregistered tests

### C branch/fork

Preregistration requires branch/fork independence.

Final runner records:

`branches_independent = bool(not torch.equal(predP, predQ) or True)`

This expression is always TRUE.

The load-bearing `lifecycle_pass` only consumes:
`parent_unchanged_bit_exact`

It does not test branch P/Q equivalence against independently replayed branches or another substantive branch-isolation oracle.

Therefore:

C_PARENT_IMMUTABILITY = PASS
C_BRANCH_INDEPENDENCE = NOT_PROVEN_BY_FINAL_TEST

### E reset/reuse

Preregistration requires:
- reset to fresh/zero state;
- reused-cache run BIT_EXACT vs fresh-cache run.

Final runner:
- dirties a cache;
- manually zeroes it;
- checks zero state.

It does NOT run the post-reset reused cache and compare its continuation to a fresh-cache run.

Therefore:

E_ZERO_RESET = PASS
E_REUSE_EQUIVALENCE = NOT_TESTED

### F serialization round-trip

Preregistration requires:
state -> CPU bytes -> state -> continue
and continuation BIT_EXACT vs no-roundtrip continuation.

Final F:
- serializes/deserializes the state;
- compares the immediate state hash.

It does not perform F's declared continuation comparison.

B already gives a related save/reload/continue positive result, but B does not retroactively execute F as preregistered.

Therefore:

F_STATE_BYTES_ROUNDTRIP = PASS
F_DECLARED_CONTINUATION_CHECK = NOT_EXECUTED

These defects independently prevent a strict claim that "Lifecycle A–J all passed as preregistered."

---

## 5. Correct T0 verdict and gate consequence

Under the frozen T0 contract:

OFFICIAL_MAMBA_LIFECYCLE_STRICT_PREREG =
NOT_QUALIFIED

At minimum:
- D exceeds preregistered TOL_BATCH;
- G does not meet the preregistered batch1 BIT_EXACT requirement;
- C branch independence is not substantively tested;
- E post-reset reuse equivalence is not tested;
- F continuation-after-roundtrip is not tested.

The train protocol says:

Both
OFFICIAL_MAMBA_LIFECYCLE = QUALIFIED
and
SINGLE_PASS_HISTORICAL_CAPTURE = QUALIFIED

are required for Item B.

Otherwise:
persist negative evidence, package, STOP.

Instead the runner redefined D/G, minted lifecycle QUALIFIED and ran 3A/3B/economics/scout.

Therefore:

PROTOCOL_GATE_ORDERING =
FAILED

3A / 3B / economics / scout are genuine observations but:

DOWNSTREAM_CONFIRMATORY_AUTHORITY =
EXPLORATORY_NON_LOAD_BEARING

This is the same evidence-first discipline used in prior RNN audit reconciliations:
do not delete downstream results; downgrade their authority when an upstream hard gate was not actually satisfied.

---

## 6. Finding C — the 3A observations are strong and internally coherent

Even though 3A is downstream of a failed strict gate, its observed effect is scientifically important.

Reported fresh 3A observations:

FINAL = 0.21875

FIXED_SLOT_76 = 0.770833
delta vs FINAL = +0.5521
CI [0.4792, 0.6198]

MAX_CONFIDENCE = 0.822917
delta vs FINAL = +0.6042
CI [0.5313, 0.6771]

MAX_CONFIDENCE - FIXED_SLOT_76 =
+0.0521
CI [-0.0156, 0.1147]

The narrow-band result independently reproduces the qualitative RNN-06D audit finding:

historical recovery is very large;
adaptive selection over a strong fixed historical state is only directional in this narrow target band.

Correct classification:

HISTORICAL_RECOVERY_TRANSPORT_3A =
EXPLORATORY_STRONG_POSITIVE

ADAPTIVE_SELECTOR_ADVANTAGE_3A =
EXPLORATORY_DIRECTIONAL

Not confirmatory until T0 is prospectively requalified and a fresh outcome set is used.

---

## 7. Finding D — 3B contains a strong adaptive signal, but its causal explanation is overstated

3B target band:
[8,144]

Schedule:
[38,76,115,153]

The preregistration and runner comments state:

"no single fixed snapshot is guaranteed to have observed every target."

This is false.

Because:
153 > 144

slot153 is post-target for EVERY target in the declared band.

So the experiment does NOT establish adaptive value because "no fixed snapshot has seen every target."

What it actually tests is more interesting:

a very late fixed snapshot has seen every target,
but may have forgotten older targets;
an earlier snapshot may preserve an old target,
but may precede a later target.

The adaptive problem is therefore:

choose a useful temporal state under the tradeoff
NOT_YET_WRITTEN vs ALREADY_FORGOTTEN.

That is a defensible and stronger interpretation.

Observed qualification-set fixed accuracies:
slot38  = 0.2396
slot76  = 0.3958
slot115 = 0.4375
slot153 = 0.4844

MAX_CONFIDENCE = 0.7760

Therefore MAX_CONFIDENCE exceeds even the post-hoc best fixed qualification arm (slot153) by:

0.7760 - 0.4844 = +0.2917

This post-hoc contrast is descriptive only because slot153 was not the preregistered primary fixed control.

It nevertheless shows that the adaptive signal is not an artifact of choosing a particularly weak fixed comparator.

Correct interpretation:

WIDE_TARGET_ADAPTIVE_SIGNAL =
EXPLORATORY_STRONG_POSITIVE

Mechanistic narrative:
confidence selection appears to track the temporal "seen but not yet forgotten" window.

Do NOT claim:
"adaptive selection is needed because no fixed snapshot has observed every target."

---

## 8. Finding E — the calibration fixed-control tie had no preregistered tie-break

3B calibration:

slot38  = 0.234375
slot76  = 0.401042
slot115 = 0.494792
slot153 = 0.494792

slot115 and slot153 are EXACTLY TIED.

Preregistration says:
select the single schedule slot with the highest mean calibration accuracy.

It does not declare a tie-break rule.

Final code uses Python `max(dict, key=dict.get)`, so insertion order selects slot115.

Therefore:

BEST_FIXED_CALIBRATION =
UNDERDETERMINED_BY_PREREG

SELECTED_FIXED_CONTROL =
slot115_by_implicit_insertion_order

This weakens the formal primary comparator.

It does not erase the observed adaptive signal because slot153, the tied alternative and actual best fixed arm on qualification, remains far below MAX_CONFIDENCE.

Future design:
- preregister a deterministic tie-break; or preferably
- if calibration arms tie within a declared equivalence band, carry all tied controls into qualification and gate the adaptive method against the strongest one.

---

## 9. Finding F — end-to-end economics is not apples-to-apples

The economics preregistration requires a FINAL-only normal-deployment baseline and recovery-enabled execution.

Final source implements `final_only_fused` as:

`model(ctx).logits[:, -1, :]`

It times the 768-token context only.

It does NOT process the 2-token target query and does not compute the same semantic answer produced by the step/recovery arms.

The source comment explicitly acknowledges that the query is not included.

The recovery and final-step arms DO process the query.

Therefore:

FINAL_FUSED_BASELINE_SEMANTIC_EQUIVALENCE =
FAIL

The measured:

+991.45 ms/query vs fused

is not a strict same-work end-to-end contrast.

The correct fused baseline should process:
context + identical query
and materialize the same scored answer.

This defect likely makes the current fused baseline artificially cheaper, so a corrected comparison may actually make recovery look slightly better, but that cannot be assumed.

A second issue:
the gate margin is only

1000.00 - 991.45 = 8.55 ms/query

< 1% of the envelope.

Six warm iterations are reduced to a median, but the packet does not persist/report dispersion or a latency confidence interval.

Therefore the threshold crossing is not robustly demonstrated even if the baseline were semantically equivalent.

Correct classification:

ECONOMICS_PROFILE =
EXPLORATORY_USEFUL

END_TO_END_RECOVERY_UTILITY =
NOT_QUALIFIED_AS_APPLES_TO_APPLES

A custom/capture-exposing kernel "would collapse" the cost is a hypothesis, not an established result.

---

## 10. Finding G — the "non-synthetic" scout is semi-synthetic and very narrow

The scout JSON correctly describes itself as:

self-contained RULER-style NL needle-in-haystack
real-English filler
controlled single-token needle
N=64
context length 768

Observed:
FINAL = 1.0
MAX_CONFIDENCE = 1.0
fixed slot115 = 0.8125

This is useful negative evidence:
the tested simple NL needle condition creates no recoverable final-state failure.

But it is not a realistic workload qualification and not fully "non-synthetic."

Correct classification:

NL_SEMI_SYNTHETIC_NEEDLE_SCOUT =
NO_SIGNAL_AT_CTX768

Do NOT generalize to:
"ordinary natural language does not saturate Mamba-2"
or
"the phenomenon is specific only to dense unique-load MQAR."

Those remain hypotheses.

---

## 11. Finding H — weights fingerprint is not a strong immutable-weights proof

`weights_identity()` hashes formatted per-parameter sums rather than the full tensor bytes.

That is useful as a cheap mutation sentinel but is not collision-resistant evidence that all weight bytes are unchanged.

The pinned official checkpoint revision and absence of training code are stronger provenance context.

Correct wording:

WEIGHT_MUTATION_SENTINEL =
UNCHANGED

Do not describe the sum fingerprint alone as a cryptographic weight identity.

Future packets may retain the cheap sentinel but should distinguish:
- artifact/checkpoint identity;
- loaded-weight fingerprint;
- mutation sentinel.

---

## 12. Corrected scientific ledger after audit

CURRENT evidence from this train:

OFFICIAL_MAMBA_FASTPATH =
RUNNABLE_SUPPORTED

OFFICIAL_MAMBA_SINGLE_PASS_CAPTURE_MECHANICS =
SUPPORTED_STRONGLY

OFFICIAL_MAMBA_LIFECYCLE_STRICT_PREREG =
NOT_QUALIFIED

FIXED_BATCH_NEIGHBOR_ISOLATION =
POSITIVE_OBSERVATION

BATCH_SIZE_NUMERICAL_INVARIANCE =
NOT_QUALIFIED
(observed state max_abs 0.5 batch1 vs batch6)

HISTORICAL_RECOVERY_OFFICIAL_MAMBA =
EXPLORATORY_STRONG_POSITIVE

ADAPTIVE_CONFIDENCE_WIDE_TARGET =
EXPLORATORY_STRONG_POSITIVE

END_TO_END_DEPLOYMENT_UTILITY =
NOT_QUALIFIED

REALISTIC_WORKLOAD_OPERATING_POINT =
NOT_TESTED

NL_SEMI_SYNTHETIC_NEEDLE_CTX768 =
NO_SIGNAL

QWEN =
DEFER

OLD_SYNTHETIC_DENSE_POST_HOC_MEMORY_CACHING =
PARKED

Nothing in this audit claims the historical observations are fabricated or uninteresting.
It changes their evidence class because the upstream qualification gate did not hold prospectively.

---

## 13. Exactly one next recommendation

OPEN a NEW RNN-06T2 REQUALIFICATION + CONFIRMATION TRAIN.

Do NOT proceed directly to realistic long-context workload discovery.

The next train should contain two gated backlog items:

1. T0R — fresh lifecycle requalification
   - preserve the original T0 failure;
   - define a new fixed-batch operational contract prospectively;
   - explicitly decide whether batch-size invariance is required or out of scope;
   - fresh held-out lifecycle set;
   - substantive C branch equivalence/isolation;
   - actual E reset -> reuse -> fresh comparison;
   - actual F serialize -> restore -> continue comparison;
   - proper boundary checks;
   - no `or True`;
   - qualification artifact persisted before downstream outcomes.

2. T1R — fresh recovery confirmation + corrected economics
   only if T0R qualifies:
   - fresh narrow transport set if exact 06D transport remains a desired claim;
   - fresh wide-target set;
   - MAX_CONFIDENCE frozen;
   - fixed controls all retained;
   - deterministic calibration tie policy;
   - adaptive primary contrast against the strongest preregistered fixed-control set;
   - semantic interpretation = seen/not-yet-written vs forgotten, not "no fixed state saw all targets";
   - apples-to-apples FINAL fused baseline processes the identical query and produces the identical scored answer;
   - persist latency samples/dispersion and require robust cost-gate margin.

Only after RNN-06T2 independently closes those gates should the program open the realistic high-interference long-context operating-point investigation.

Do not run Qwen in RNN-06T2.
Do not train a selector.
Do not implement DART/StateX/SDM/GDN-2/INT8/ReplaySSM.
