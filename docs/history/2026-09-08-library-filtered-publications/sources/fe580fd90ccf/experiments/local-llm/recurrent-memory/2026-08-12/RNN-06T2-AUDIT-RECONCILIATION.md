# AUDIT RECONCILIATION — RNN-06T2 Official Mamba Requalification + Recovery Confirmation
Date: 2026-08-12
Auditor: independent ChatGPT audit

## Overall verdict

RNN_06T2_TRAIN = ACCEPTED_WITH_ECONOMICS_RECONCILIATION

BUNDLE_INTEGRITY = PASS

OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE = QUALIFIED
BATCH_SHAPE_NUMERICAL_PORTABILITY = OUT_OF_SCOPE_NOT_QUALIFIED
SINGLE_PASS_HISTORICAL_CAPTURE_T0R = QUALIFIED

HISTORICAL_RECOVERY_NARROW = QUALIFIED
ADAPTIVE_SELECTION_NARROW = DIRECTIONAL

WIDE_TARGET_RECOVERY_T1R = QUALIFIED
ADAPTIVE_SELECTION_T1R = QUALIFIED

END_TO_END_RECOVERY_UTILITY_T1R_HISTORICAL_MINT = RECONCILED
END_TO_END_RECOVERY_UTILITY_T1R = NOT_COMPARABLE
MARGINAL_STEP_PATH_TIMING_SIGNAL = POSITIVE_BUT_NON_LOAD_BEARING

REALISTIC_WORKLOAD_DISCOVERY = AUTHORIZED_WITH_ECONOMICS_CLOSURE_CARRIED_FORWARD
QWEN_TRANSFER = DEFER

No recovery/lifecycle GPU rerun is required.
A small economics semantic-closure rerun is required before claiming end-to-end utility.

---

## 1. Bundle integrity

Attached archive:
RNN-06T2-MAMBA-REQUALIFICATION-audit-bundle.zip

ZIP SHA-256:
52fcf4d00430bb8b24da3c2cfd8b5a4c1c2473c701b2939acbd0f633e4a35426

External handoff SHA-256:
d6d409f7f1a7db00f01af9f6b005d467487969f4a2a94dc2ae9ba464b59cbc53

Checks:
- 41 ZIP entries total
- CRC/testzip PASS
- 39 payload entries
- archive payload set == TRAIN_MANIFEST payload set
- archive payload set == SHA256SUMS payload set
- 39/39 payload size + SHA-256 PASS
- skipped_missing = []
- external handoff differs from internal HANDOFF.md in exactly one expected line:
  the internal `<!--ZIP_SHA-->` placeholder is replaced by the outer-envelope ZIP SHA-256.

BUNDLE_INTEGRITY = PASS

---

## 2. T0R prospective governance

The new T0R preregistration prospectively separates:

A. FIXED_BATCH_REQUEST_ISOLATION
B. BATCH_SHAPE_NUMERICAL_PORTABILITY

and explicitly freezes Property B as:

OUT_OF_SCOPE_FOR_FIXED_BATCH_RECOVERY

before substantive outcomes.

The historical RNN-06T batch1-vs-batchB result remains:

max_abs_diff = 0.5
historical TOL_BATCH = 0.03

and is not relabeled benign.

Git evidence records:

bfe4ed8 audit reconciliation
dc7ab12 T0R prereg
6e07785 T0R runner
ca301bd pre-outcome continuation-offset debug fix
e281329 T0R outcome/decision

The executed runner self-records:
- runner SHA-256
- Git blob
- clean runner status
- HEAD = ca301bd...
- protocol SHA-256
- fresh qualification-set SHA-256
- model/backend identity

The included runner bytes independently reproduce the self-recorded Git blob/SHA identity.

T0R_GATE_ORDERING = PASS

---

## 3. T0R lifecycle result

The final runner substantively repairs the material RNN-06T defects:

- branch/fork is no longer tautological;
- parent immutability and branch-state non-interference are exercised;
- reset state is actually reused and compared to a fresh cache;
- serialization is followed by continuation and compared to no-roundtrip continuation;
- fixed-batch sibling/order isolation is exercised;
- actual in-run boundary states are compared to independent same-path replay;
- official recurrent kernels are instrumented and observed firing.

Observed lifecycle state:
- A PASS
- B PASS
- C PASS
- D PASS
- E PASS
- F PASS
- G PASS
- H PASS
- I PASS
- J PASS

The exact subject remains:

state-spaces/mamba2-1.3b
revision c5b59d00ec85d313adea86a08cad2a43c962dd3b
mamba_ssm 2.2.4
causal_conv1d 1.5.0.post8
triton 3.2.0
torch 2.6.0+cu124
bf16
RTX 3090
chunk_size 256

Correct classification:

OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE = QUALIFIED

BATCH_SHAPE_NUMERICAL_PORTABILITY = OUT_OF_SCOPE_NOT_QUALIFIED

Scope is important:
this does NOT prove numerical portability when batch shape changes.

### Minor direct-assertion caveat

The frozen prose asks for direct readout checks for every branch/sibling case.
The runner sometimes proves the stronger immediate condition — BIT_EXACT recurrent/continuation state under identical fixed-batch inputs — without separately persisting every implied readout comparison.

Because the readout path is deterministic and separately exercised, this is not treated as a failed lifecycle gate. Future lifecycle packets should nevertheless persist the direct assertion whenever the preregistration names it.

---

## 4. Single-pass capture

T0R captures actual intermediate recurrent states from one canonical trajectory and independently replays the same trajectory.

Observed:
- 5 declared boundaries
- 5/5 captured state hashes match replay
- 0 boundary failures
- monotonic boundary positions
- official recurrent step kernels fire

Correct classification:

SINGLE_PASS_HISTORICAL_CAPTURE_T0R = QUALIFIED

### Provenance caveat: run-id collision

In T1R, the derived `runId` hashes only the first eight context token IDs.
The anti-oracle construction has a common sentinel prefix, so narrow and wide qualification artifacts receive the same displayed runId (`fcae7b6dd97fa2fa`) despite being different runs/sets.

This does NOT alter the state hashes or scientific outcomes, but the run ID is not globally unique.

Future fix:
bind run identity to at least
packet + qualificationSetSha256 + process/run nonce or deterministic full-example-set identity.

RUN_ID_CANONICAL_UNIQUENESS = NOT_QUALIFIED
SCIENTIFIC_IMPACT = NONE

---

## 5. Fresh-set identity / separation

Independently recomputed from the packaged JSON using the same canonical JSON hashing rule:

NARROW QUAL:
34d276ced58eddd34332ad3a17ea658edc488b8f9896fb3e1635c200b73316cc
PASS

WIDE CALIB:
dc4010f15f0d56228693d3fefea92b8dadf8ddd24e245eef285fb31d37e18614
PASS

WIDE QUAL:
97f303a2573cd07716054752fb4529e2ceabf69582a971f214c250082ddeab3f
PASS

All contain 192 examples and report zero signature overlap with the preregistered historical/sibling sets.

The wide calibration result is:

slot38  = 0.2448
slot76  = 0.3802
slot115 = 0.4583
slot153 = 0.5365

With TAU_TIE = 0.02, only slot153 is carried.

Therefore the old implicit-tie defect is closed prospectively.

FIXED_CONTROL_SELECTION = PASS_PROSPECTIVE

---

## 6. Independent raw-readout reproduction — narrow

Packaged NPZ:
- pool_logits [192,4,256]
- final_logits [192,256]
- golds [192]
- target slots [8,64]
- 4 strata
- vset [256]

The bundled scored-value token mapping is sufficient to reconstruct gold-column indices.

Independent recomputation:

FINAL
= 31/192
= 0.1614583

fixed slot38
= 0.4947917

fixed slot76
= 0.7343750

fixed slot115
= 0.4375000

fixed slot153
= 0.2500000

MAX_CONFIDENCE
= 0.8385417

selector histogram
= [115,72,2,3]

FIXED76 vs FINAL:
delta = +0.5729167
recovered = 112
harmed = 2
net = 110
bootstrap CI = [0.5052083, 0.6458333]
positive in 4/4 strata

MAX_CONFIDENCE vs FIXED76:
delta = +0.1041667
bootstrap CI = [0.046875, 0.1614583]
per-stratum delta =
[+0.3958,+0.2292,-0.1667,-0.0417]

Therefore:

HISTORICAL_RECOVERY_NARROW = QUALIFIED
ADAPTIVE_SELECTION_NARROW = DIRECTIONAL

The implementer correctly does not overclaim the narrow adaptive effect.

---

## 7. Independent raw-readout reproduction — wide

Packaged NPZ independently reproduces:

FINAL
= 52/192
= 0.2708333

fixed slot38
= 0.2187500

fixed slot76
= 0.3697917

fixed slot115
= 0.4427083

fixed slot153
= 0.5000000

MAX_CONFIDENCE
= 0.8125000

selector histogram
= [56,50,48,38]

MAX_CONFIDENCE vs FINAL:
delta = +0.5416667
recovered = 106
harmed = 2
net = 104
bootstrap CI = [0.4738281, 0.609375]
positive in 4/4 strata

MAX_CONFIDENCE vs preregistered strongest carried fixed control slot153:
delta = +0.3125000
bootstrap CI = [0.2395833, 0.3751302]
per-stratum delta =
[+0.6875,+0.3958,+0.3125,-0.1458]
positive in 3/4 strata

The frozen adaptive interpretation is correctly stated:

NOT_YET_WRITTEN
↔ SEEN_AND_RETAINED
↔ ALREADY_FORGOTTEN

slot153 is after every possible target in [8,144], so the result is not explained by "no fixed snapshot saw every target."

Correct classification:

WIDE_TARGET_RECOVERY_T1R = QUALIFIED
ADAPTIVE_SELECTION_T1R = QUALIFIED

Scope:
exact frozen official Mamba-2 checkpoint/backend + synthetic anti-oracle forgetting construction.

This is now confirmatory evidence for adaptive target-agnostic historical-state selection under the tested wide-target regime.

---

## 8. Post-outcome instrumentation source change

Git evidence shows:

cd294a2
T1R prereg + challenge sets + runner, before outcomes

a65b7fb
first T1R recovery results

49327ba
runner changed to add Section-12 mechanism counters

The final packaged readouts/results were produced with the later runner identity.

This is a post-outcome source change on the same qualification set.

The final source conforms to the frozen scientific contract, the raw readouts are independently reproducible, the added fields are mechanism counters, and the reported first/second-run numerical differences are small relative to the qualification margins.

Therefore:

POST_OUTCOME_INSTRUMENTATION_RERUN =
ACCEPTED_WITH_PROVENANCE_CAVEAT

It does not overturn the recovery/adaptive mints.

However, the final bundle does not include the exact a65b7fb→49327ba patch, so an independent auditor cannot prove from this bundle alone that the change was instrumentation-only.

Future rule:
any outcome-exposed runner modification should package the exact diff, even when believed non-semantic.

---

## 9. Cross-process numerical evidence

The handoff reports ~1–3/192 borderline argmax changes across process starts under bf16/kernel autotuning while lifecycle same-process replay remains BIT_EXACT.

This is appropriately negative evidence.

Correct scope:

IN_PROCESS_SAME_PATH_DETERMINISM = QUALIFIED
CROSS_PROCESS_BIT_EXACT_SEMANTICS = NOT_QUALIFIED
CROSS_PROCESS_VERDICT_STABILITY = OBSERVED_POSITIVE

Do not convert this into a universal "autotuning causes harmless noise" claim.

---

## 10. Major audit finding — economics arms are NOT semantically comparable as implemented

The T1R preregistration states:

every timed arm must return the same scored-answer type

and:

END_TO_END_RECOVERY_UTILITY_T1R = NOT_COMPARABLE
if they do not.

The final source violates this condition.

In `ops/rnn_06t2_econ.py`:

`final_fused_equiv(...)`
returns:
`vt[sub.argmax(-1)]`

That is a scored-value TOKEN ID.

`final_step_equiv(...)`
returns the result of `L.readout(...)`,
also a scored-value TOKEN ID.

But `recovery_equiv(...)` computes:

`pred = pl.argmax(-1)[..., sel]`

where `pl` has shape [B,K,V].

`pred` is therefore the COLUMN INDEX in the scored vocabulary, range 0..V-1.

It then returns:

`torch.tensor(pred)`

rather than:

`vt[pred]`

Thus:

FINAL_FUSED output domain = token IDs
FINAL_STEP output domain  = token IDs
RECOVERY output domain    = scored-vocabulary column indices

The timing functions ignore the returned values, so this does not falsify the measured runtimes.

But by the train's own frozen decision rule:

END_TO_END_RECOVERY_UTILITY_T1R = NOT_COMPARABLE

until the recovery arm returns the same semantic output type and the economics measurement is rerun.

This is a direct false-green in the economics mint.

Corrected state:

END_TO_END_RECOVERY_UTILITY_T1R_HISTORICAL_MINT
= RECONCILED_FALSE_GREEN

END_TO_END_RECOVERY_UTILITY_T1R
= NOT_COMPARABLE

---

## 11. Economics signal retained as non-load-bearing measurement

The packaged timing data itself is useful.

80 pooled warm samples across 2 process starts:

FINAL_FUSED_EQUIVALENT_WORK
median 37.73 ms/query
p95 38.80

FINAL_STEP_EQUIVALENT_WORK
median 976.66
p95 1064.45

RECOVERY_ENABLED_EQUIVALENT_WORK
median 1010.25
p95 1121.26

Reported paired-by-index added-vs-step:
median +41.27
p95 +192.70

Reported added-vs-fused:
median +972.12
p95 +1083.45

Memory:
- final fused peak VRAM ≈4.024 GB
- final step ≈4.403 GB
- recovery ≈7.731 GB
- batch K+final state bytes ≈4.16 GB

Two additional interpretation limits matter:

1. The `RECOVERY - FINAL_STEP` comparison measures MARGINAL recovery overhead conditional on already selecting the capture-capable step execution path.
2. A normal fused deployment alternative remains almost one second/query faster in this implementation.

Therefore even after fixing the output-domain bug, do not equate:

MARGINAL_RECOVERY_UTILITY_ON_STEP_PATH

with:

GENERAL_END_TO_END_DEPLOYMENT_UTILITY_VS_BEST_AVAILABLE_BASELINE

Those are separate systems questions.

Additionally, the warm arms are benchmarked in separate blocks and nth samples are subtracted by index. Future economics should interleave/randomize arms within repeated cycles or otherwise model temporal drift rather than treating independently timed block positions as natural pairs.

---

## 12. Correct experimental ledger after RNN-06T2 audit

The experimental state should now be:

OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE
= QUALIFIED

BATCH_SHAPE_NUMERICAL_PORTABILITY
= OUT_OF_SCOPE_NOT_QUALIFIED

SINGLE_PASS_HISTORICAL_CAPTURE_T0R
= QUALIFIED

HISTORICAL_RECOVERY_NARROW
= QUALIFIED

ADAPTIVE_SELECTION_NARROW
= DIRECTIONAL

WIDE_TARGET_RECOVERY_T1R
= QUALIFIED

ADAPTIVE_SELECTION_T1R
= QUALIFIED

HISTORICAL_RECURRENT_STATE_AS_MEMORY_ON_OFFICIAL_MAMBA
= SUPPORTED_STRONGLY_IN_SYNTHETIC_FORGETTING_REGIME

TARGET_AGNOSTIC_ADAPTIVE_TEMPORAL_STATE_SELECTION
= QUALIFIED_IN_WIDE_SYNTHETIC_REGIME

END_TO_END_RECOVERY_UTILITY_T1R
= NOT_COMPARABLE

MARGINAL_STEP_PATH_TIMING_SIGNAL
= POSITIVE_NON_LOAD_BEARING

REALISTIC_NATURAL_WORKLOAD_GENERALIZATION
= NOT_TESTED

QWEN
= DEFER

OLD_SYNTHETIC_DENSE_POST_HOC_MEMORY_CACHING
= PARKED

This still does NOT justify the broad phrase:

"Memory Caching works."

The supported statement is narrower:

On the exact official Mamba-2 substrate and a prospectively qualified fixed-batch lifecycle, historical recurrent states retain large recoverable signal in a controlled synthetic forgetting regime, and a frozen target-agnostic confidence selector provides a large incremental advantage over the prospectively selected strongest fixed historical state in the wide-target condition.

---

## 13. Authorization decision

The lifecycle/recovery scientific dependency is now clean enough to move beyond the synthetic operating point.

Therefore:

REALISTIC_WORKLOAD_DISCOVERY = AUTHORIZED

The economics false-green does NOT require rerunning T0R or T1R quality.

Carry a small economics semantic-closure task into the next train before any production/economic conclusion.

Do not let the economics repair block scientific discovery of whether the phenomenon occurs in natural workloads.

---

## 14. Exactly one next recommendation

Open one NEW train with two work items:

1. `RNN-06T2-E1 ECONOMICS SEMANTIC CLOSURE`
   - make all arms return the same token-ID answer domain;
   - assert output-domain/type and deterministic scorer equivalence;
   - interleave/randomize timing arms;
   - rerun economics only;
   - keep the old timing results immutable.

2. `RNN-07A REALISTIC OPERATING-POINT DISCOVERY`
   - same official checkpoint/backend;
   - no trained selector;
   - MAX_CONFIDENCE frozen;
   - bounded natural/semi-natural benchmark discovery;
   - establish short-context/task competence before interpreting long-context failures;
   - locate a natural forgetting regime without synthetic sentinel/DS construction;
   - only then test historical recovery.

Do not start Qwen in this train.
Do not train a recovery reader.
Do not implement DART/StateX/SDM/GDN-2/INT8/ReplaySSM.
