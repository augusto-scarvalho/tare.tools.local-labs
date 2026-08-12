# RNN-06T — Independent Audit Reconciliation (APPEND-ONLY)

**Status:** Append-only. This document is added *after* the historical RNN-06T packet was
committed at HEAD `06013bd`. It records the reconciled interpretation of an independent audit
(`AUDIT_RECONCILIATION_RNN-06T_2026-08-12.md`). It does **NOT** edit, relax, or overwrite any
historical RNN-06T evidence.

The following historical files are preserved verbatim as historical evidence and are **NOT**
modified by this reconciliation:

- `T0_PRE_REGISTRATION.md` (T0 preregistration)
- `T0_RESULTS.json`, `T0_DECISION.md` (T0 results + decision)
- `T1_3A_*` (3A results/decision)
- `T1_3B_*` (3B results/decision)
- `T1_ECONOMICS.json`, `T1_ECON_PRE_REGISTRATION.md` (economics)
- `T1_NONSYNTH_SCOUT.json` (scout results)

Repository / Git / source remain authoritative. This audit is reconciliation evidence, not a
replacement for repo truth.

---

## 1. Reconciled interpretation of the historical RNN-06T packet

| Property | Reconciled verdict |
|---|---|
| `OFFICIAL_MAMBA_FASTPATH` | **RUNNABLE_SUPPORTED** |
| `SINGLE_PASS_HISTORICAL_CAPTURE` | **SUPPORTED_STRONGLY** |
| `OFFICIAL_MAMBA_LIFECYCLE_STRICT_PREREG` | **NOT_QUALIFIED** |
| `PROTOCOL_GATE_ORDERING` | **FAILED** |
| RNN-06T 3A / 3B / ECONOMICS / SCOUT | **EXPLORATORY_NON_LOAD_BEARING** |

**Interpretation.** The official `mamba_ssm` fast path is genuinely runnable and fires (kernel
counters, no reachable fallback), and in-run single-pass historical capture is strongly supported.
However, the *strict* lifecycle preregistration was not met: the T0 gate minted
`OFFICIAL_MAMBA_LIFECYCLE = QUALIFIED` while (a) a preregistered numerical property was replaced by
a substitute property at analysis time, and (b) the gate that must precede the downstream
3A/3B/economics/scout work was not honored in ordering. Because the lifecycle gate did not hold
under strict preregistration and the gate ordering failed, the historical 3A / 3B / economics /
scout results are reclassified as **exploratory, non-load-bearing** — they retain descriptive value
but do not carry qualification weight.

The RNN-06T *decision-bearing* mints (`OFFICIAL_MAMBA_LIFECYCLE=QUALIFIED`,
`SINGLE_PASS_HISTORICAL_CAPTURE=QUALIFIED`, `WIDE_TARGET_RECOVERY=QUALIFIED`,
`ADAPTIVE_SELECTION=QUALIFIED`, `END_TO_END_RECOVERY_UTILITY=QUALIFIED`) are therefore NOT accepted
as prospective qualifications. They are superseded by the fresh, prospective RNN-06T2 packet.

---

## 2. Preserved negative / diagnostic evidence

### 2.1 Batch-shape numerical observation (preserved as negative evidence)

The original observed batch-size result is preserved:

```
batch1 vs batch6 : state max_abs_diff = 0.5
```

against the historical preregistered tolerance:

```
TOL_BATCH = 0.03
```

`0.5 >> 0.03`. Under the strict preregistration this is a **FAIL** of the preregistered
batch-shape numerical property. The historical T0 runner recorded this value descriptively
(`lifecycle_tests.D_neighbor_isolation.batch_size_sensitivity_max_abs_diff_batch1_vs_batchB = 0.5`)
and substituted a *fixed-batch neighbor-invariance* property (which was bit-exact) as the isolation
claim, then annotated the 0.5 as a "benign Triton-tiling artifact."

Reconciliation:
- The 0.5 difference **remains negative evidence** with respect to the preregistered `TOL_BATCH`.
- The later fixed-batch D/G test is a **different property** and must NOT be described as though it
  had been the original preregistered property.
- The 0.5 difference must **NOT** be called "benign" as a *causal conclusion*. It is an
  unexplained numerical divergence under batch-shape change; its downstream irrelevance is a
  *scoping choice* to be preregistered prospectively (see RNN-06T2), not a proven causal claim.

### 2.2 Lifecycle test-construction findings (preserved)

| Audit finding | Reconciled status |
|---|---|
| `C_PARENT_IMMUTABILITY` | **PASS** — parent state hash unchanged after branch readouts. |
| `C_BRANCH_INDEPENDENCE` | **NOT_PROVEN_BY_FINAL_TEST** — the historical branch-independence assertion used a tautology (`bool(not torch.equal(predP, predQ) or True)`), which is unconditionally `True`. Real fork independence (fresh per-branch reference reconstruction + cross-non-interference) was not proven. |
| `E_ZERO_RESET` | **PASS** — reset zeroes the cache tensors. |
| `E_REUSE_EQUIVALENCE` | **NOT_TESTED** — reusing a reset cache for a continuation and comparing to a genuinely fresh cache was never executed (zero-check alone is insufficient). |
| `F_STATE_BYTES_ROUNDTRIP` | **PASS** — serialize→deserialize preserves the state hash. |
| `F_DECLARED_CONTINUATION_CHECK` | **NOT_EXECUTED** — continuing execution *after* a serialization roundtrip and comparing to a no-roundtrip continuation was not executed (immediate hash equality alone is insufficient). |

### 2.3 No GPU rerun belongs to the historical packet

This reconciliation performs **no** GPU rerun against the historical packet. All historical numbers
are preserved as-is. The prospective remedial work is a **new** packet (`RNN-06T2`), executed fresh.

---

## 3. Successor packet

A new packet is opened:

```
RNN-06T2-MAMBA-REQUALIFICATION
```

It prospectively qualifies the *fixed-batch* single-pass historical-recovery lifecycle contract that
is actually needed downstream, and (only if that gate holds) re-confirms recovery + corrected
end-to-end economics. It does NOT rewrite or relax historical RNN-06T T0. See
`runs/rnn/RNN-06T2/` for the fresh preregistration, frozen identities, results, and decisions.

**Authority of this document:** reconciliation/record only. It changes no historical file and mints
no new qualification. All new qualification is minted prospectively under RNN-06T2.
