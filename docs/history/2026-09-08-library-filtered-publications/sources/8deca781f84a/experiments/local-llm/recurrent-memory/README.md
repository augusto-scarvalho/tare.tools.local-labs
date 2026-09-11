# Recurrent Memory / RNN / Memory Caching — Experimental Lineage

**Document ID:** `experiment.local-llm.recurrent-memory.lineage-2026-08-12`  
**Document type:** `experiment`  
**Status:** `EXPERIMENTAL`  
**Last updated:** `2026-08-12`  
**Primary contexts:** Runtime / Model / Inference · Memory / Context · Validation / Assurance · Observability / Economics / Resources  
**Authority:** research/experimental evidence only; does not mint tare.tools TARGET or canonical CURRENT.

## 1. Purpose

This lineage preserves the scientific path from TPTT and synthetic Memory Caching through controlled recurrent forgetting, historical-state information, parameter-free recovery and official-Mamba transportability.

The program's question has progressively narrowed:

```text
Can recurrence / Memory Caching improve a Transformer?
        ↓
Does historical-memory utility require co-adaptation?
        ↓
Can a frozen recurrent model exhibit controlled forgetting?
        ↓
Does unique recurrent-state load cause retrieval loss?
        ↓
Does an earlier recurrent state preserve information the final state lost?
        ↓
Can target-agnostic historical-state selection recover that information?
        ↓
Does the phenomenon transport to an official pretrained Mamba checkpoint?
        ↓
Does it occur on realistic long-context workloads and remain economically useful?
```

## 2. Evidence rule

```text
preregistration + executed source + raw results
        >
independent audit / reconciliation
        >
implementer handoff / narrative
        >
research synthesis
```

A later audit may downgrade an earlier mint without deleting the historical artifact. Downstream outcomes behind a failed hard gate remain observations, but lose confirmatory authority.

## 3. Compact ledger

| Stage | Audited experimental result | Meaning |
|---|---|---|
| RNN-08/09 | tested TPTT configuration `PARKED` | corrected lifecycle, then tested TPTT+LoRA regressed vs LoRA; not a universal TPTT falsification |
| RNN-04 | exploratory co-trained MC signal | first historical-memory signal, but not frozen/post-hoc proof |
| RNN-05A | frozen LA post-hoc MC `NO_EFFECT / NEGATIVE` | historical memory was not a free attach-on in that substrate |
| RNN-05B | co-adaptation interaction `SUPPORTED` | train/inference memory-mode alignment matters in DN/GDN |
| RNN-05B-EXT | `BLOCKED_BY_UNSTABLE_BASE` | per-condition training conflated forgetting with optimization instability |
| RNN-05B-EXT2 | fixed-backbone graded region `BLOCKED` | no qualified forgetting region; no MC; no EXT3 |
| RNN-06-P0 | Mamba graded band `PLAUSIBLE` exploratory | motivated real recurrent-LM path |
| RNN-06A | strict lifecycle `NOT_QUALIFIED` | numerical path divergence + prereg mismatch preserved |
| RNN-06A2 | continuation lifecycle `QUALIFIED` | exact pinned Transformers-native continuation/checkpoint contract passed |
| RNN-06B | original causal contract `BLOCKED` | same-space competition unsupported; length vs general load unresolved |
| RNN-06B2 | fixed-length state-load contract `BLOCKED` | order/full-packing confounds found |
| RNN-06B3 | controlled unique-binding-load forgetting `QUALIFIED` | fixed length/gap/order with robust load-dependent loss |
| RNN-06C | historical-state information `QUALIFIED (PRESENCE)` | earlier state remains behaviorally informative after final-state loss |
| RNN-06D0 | recovery ceiling `QUALIFIED` | scheduled historical pool contains large recoverable signal |
| RNN-06D1 | parameter-free recovery gate `QUALIFIED` | target-agnostic historical recovery strongly positive on exact synthetic substrate |
| RNN-06D audit | adaptive increment in narrow band `OPEN` | a strong fixed state captured most of the gain |
| RNN-06T | official fast path + real in-run capture observed | strict lifecycle prereg still failed; downstream downgraded exploratory |
| **RNN-06T2** | **`ACCEPTED_WITH_ECONOMICS_RECONCILIATION`** | prospective fixed-batch lifecycle and wide adaptive recovery qualify; economics mint has a semantic false-green |

## 4. Controlled forgetting and historical information

### RNN-06B3 — controlled unique-binding load

On the exact frozen Mamba subject, with total length, target→query gap, binding order and packing confounds controlled, DS constrained accuracy declined:

```text
U=1     0.990
U=24    0.917
U=48    0.865
U=72    0.771
U=96    0.755
U=128   0.651
U=152   0.568
U=176   0.573
```

Paired U1→U176 loss ≈ `0.417`, reported 95% interval `[0.349, 0.490]`.

Correct scope:

`GENERAL_UNIQUE_BINDING_STATE_LOAD_EFFECT = QUALIFIED`

Literal recurrent-state capacity saturation remains a mechanistic hypothesis rather than a separately isolated cause.

### RNN-06C — information presence

Same-aged branches:

| Condition | Accuracy |
|---|---:|
| historical-direct | 0.849 |
| repeated-sentinel aged control | 1.000 |
| high unique-load | 0.547 |

Primary `N-L = 0.453125`, reported interval `[0.385, 0.526]`, with 87 N-correct→L-wrong transitions and 0 reverse.

`HISTORICAL_STATE_INFORMATION = QUALIFIED (PRESENCE)`

The sentinel arm is a matched control, not a universally neutral transform.

## 5. RNN-06D — recovery on the synthetic Transformers-native Mamba substrate

Independent recomputation from raw arrays:

```text
FINAL            0.1302
fixed slot76     0.7708
ORACLE_BEST      0.9063
MAX_CONFIDENCE   0.8333
```

MAX_CONFIDENCE recovered 135 examples and harmed 0 relative to FINAL.

A missing fixed-state audit control showed that slot76 alone already supplied `+0.6406`, so the narrow-band result supports historical-state utility much more strongly than adaptive-selector incremental value.

```text
PARAMETER_FREE_HISTORICAL_RECOVERY_EXISTS = SUPPORTED_STRONGLY
ADAPTIVE_SELECTION_INCREMENTAL_VALUE_06D = OPEN
```

## 6. RNN-06T — transport attempt and audit downgrade

Official subject:

```text
state-spaces/mamba2-1.3b
revision c5b59d00ec85d313adea86a08cad2a43c962dd3b
mamba_ssm 2.2.4
causal_conv1d 1.5.0.post8
triton 3.2.0
torch 2.6.0+cu124
bf16
RTX 3090
```

Accepted observations:

```text
OFFICIAL_MAMBA_FASTPATH = RUNNABLE_SUPPORTED
SINGLE_PASS_HISTORICAL_CAPTURE = SUPPORTED_STRONGLY
```

But the frozen T0 prereg required run-alone/batch1 comparisons under `TOL_BATCH=0.03`; an observed batch-shape difference `0.5` was followed by a test rewrite. Additional branch/reset/serialization checks were incomplete.

Therefore:

```text
OFFICIAL_MAMBA_LIFECYCLE_STRICT_PREREG = NOT_QUALIFIED
PROTOCOL_GATE_ORDERING = FAILED
RNN-06T downstream = EXPLORATORY_NON_LOAD_BEARING
```

This failure is preserved permanently; RNN-06T2 is a new prospective contract, not a rewrite of RNN-06T.

## 7. RNN-06T2 — prospective requalification and confirmation

Independent audit: [`2026-08-12/RNN-06T2-AUDIT-RECONCILIATION.md`](2026-08-12/RNN-06T2-AUDIT-RECONCILIATION.md).

### 7.1 Lifecycle

The new protocol prospectively separates fixed-batch request isolation from batch-shape numerical portability.

```text
OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE = QUALIFIED
BATCH_SHAPE_NUMERICAL_PORTABILITY = OUT_OF_SCOPE_NOT_QUALIFIED
SINGLE_PASS_HISTORICAL_CAPTURE_T0R = QUALIFIED
```

The fixed-batch runner repairs the old false greens with substantive branch/fork, reset→reuse, serialization→restore→continue and temporal-boundary replay checks. Five of five captured historical boundaries match independent same-path replay; zero boundary failures.

Batch1-vs-batchB `max_abs_diff=0.5` remains negative evidence and is not called benign.

### 7.2 Narrow recovery

Independent raw-readout reproduction:

```text
FINAL             0.161458
fixed slot76      0.734375
MAX_CONFIDENCE    0.838542
```

Fixed76 vs FINAL:

```text
delta = +0.572917
95% CI = [0.505208, 0.645833]
recovered = 112
harmed = 2
net = 110
positive strata = 4/4
```

MAX_CONFIDENCE vs fixed76:

```text
delta = +0.104167
95% CI = [0.046875, 0.161458]
positive strata = 2/4
```

Therefore:

```text
HISTORICAL_RECOVERY_NARROW = QUALIFIED
ADAPTIVE_SELECTION_NARROW = DIRECTIONAL
```

### 7.3 Wide-target adaptive recovery

Fresh calibration prospectively carried only slot153 as strongest fixed control under the frozen tie policy.

Independent qualification reproduction:

```text
FINAL             0.270833
slot38            0.218750
slot76            0.369792
slot115           0.442708
slot153           0.500000
MAX_CONFIDENCE    0.812500
```

MAX_CONFIDENCE vs FINAL:

```text
delta = +0.541667
95% CI = [0.473828, 0.609375]
recovered = 106
harmed = 2
net = 104
positive strata = 4/4
```

MAX_CONFIDENCE vs preregistered strongest fixed state slot153:

```text
delta = +0.312500
95% CI = [0.239583, 0.375130]
positive strata = 3/4
```

Therefore:

```text
WIDE_TARGET_RECOVERY_T1R = QUALIFIED
ADAPTIVE_SELECTION_T1R = QUALIFIED
```

This is confirmatory evidence for the frozen target-agnostic confidence selector **only in the exact wide synthetic forgetting regime**.

The supported temporal interpretation is:

```text
NOT_YET_WRITTEN
        ↕
SEEN_AND_RETAINED
        ↕
ALREADY_FORGOTTEN
```

### 7.4 Provenance caveats

- T1R had a post-outcome instrumentation rerun on the same qualification set. The final source conforms to the frozen scientific computation and results remain far from thresholds, so the audit accepts it with a provenance caveat; future outcome-exposed source changes must package the exact diff.
- T1R `runId` hashes too short a common prefix, causing a display-ID collision between distinct sets. State/result hashes are unaffected, but `RUN_ID_CANONICAL_UNIQUENESS = NOT_QUALIFIED`.
- same-process lifecycle replay is bit-exact; cross-process bf16 output is not proven bit-exact.

## 8. Economics false-green

The RNN-06T2 handoff minted:

`END_TO_END_RECOVERY_UTILITY_T1R = QUALIFIED`

The independent audit rejects this mint.

`FINAL_FUSED` and `FINAL_STEP` return scored token IDs, while `RECOVERY_ENABLED` returns the selected scored-vocabulary **column index** rather than mapping it back through the value-token set.

The frozen protocol says mismatched answer domains make the comparison `NOT_COMPARABLE`.

Correct ledger:

```text
END_TO_END_RECOVERY_UTILITY_T1R_HISTORICAL_MINT = RECONCILED_FALSE_GREEN
END_TO_END_RECOVERY_UTILITY_T1R = NOT_COMPARABLE
MARGINAL_STEP_PATH_TIMING_SIGNAL = POSITIVE_NON_LOAD_BEARING
GENERAL_END_TO_END_DEPLOYMENT_UTILITY = OPEN
```

The timing samples remain useful observations, but they do not currently mint economic qualification.

A small economics-only closure is required; lifecycle and recovery must not be rerun for this defect.

## 9. Memory Caching terminology

Keep separate:

```text
SYNTHETIC_DENSE_POST_HOC_MEMORY_CACHING_RECIPE = PARKED
```

from:

```text
HISTORICAL_RECURRENT_STATE_AS_MEMORY = SUPPORTED_STRONGLY_IN_CONTROLLED_SYNTHETIC_REGIME
TARGET_AGNOSTIC_ADAPTIVE_TEMPORAL_STATE_SELECTION = QUALIFIED_IN_WIDE_SYNTHETIC_REGIME
```

Do not publish the broad sentence “Memory Caching works.”

Natural-workload generalization, production economics, batch-shape portability and Qwen transfer are still unproven.

## 10. Current frontier

The lifecycle/recovery dependency is now clean enough to leave the synthetic operating point.

Authorized next research train:

```text
RNN-06T2-E1
small economics semantic closure
        +
RNN-07A
realistic long-context operating-point discovery
```

RNN-07A is a discovery stage, not confirmation. If it finds a credible natural operating point, a fresh RNN-07B must prospectively confirm it.

```text
REALISTIC_WORKLOAD_DISCOVERY = AUTHORIZED
REALISTIC_NATURAL_WORKLOAD_GENERALIZATION = NOT_TESTED
QWEN_TRANSFER = DEFER
```

## 11. Experimental identities

- RNN-06T2 T0R set: `ca92cfad0d0aac4ae20aa8612f259c559ad592415a71797561b3e5909103cafe`
- narrow qualification: `34d276ced58eddd34332ad3a17ea658edc488b8f9896fb3e1635c200b73316cc`
- wide calibration: `dc4010f15f0d56228693d3fefea92b8dadf8ddd24e245eef285fb31d37e18614`
- wide qualification: `97f303a2573cd07716054752fb4529e2ceabf69582a971f214c250082ddeab3f`
- RNN-06T2 audit bundle: `52fcf4d00430bb8b24da3c2cfd8b5a4c1c2473c701b2939acbd0f633e4a35426`
- external handoff: `d6d409f7f1a7db00f01af9f6b005d467487969f4a2a94dc2ae9ba464b59cbc53`
- independent audit: `72fab88e53391692e80803e0bddb8fecde85c309f7d02f33560e9d001aa69b48`

Raw workstation audit ZIPs are not published into this public research tree by default; exact digests and derived audit evidence are retained.

## 12. Publication boundary

This is an `EXPERIMENTAL` research record.

It does not:

- mint tare.tools canonical CURRENT/TARGET;
- establish a production memory subsystem;
- authorize Qwen work;
- override the local experiment repo/Git;
- replace original preregistrations or raw results;
- turn discovery pointers into implementation backlog without a separate promotion path.
