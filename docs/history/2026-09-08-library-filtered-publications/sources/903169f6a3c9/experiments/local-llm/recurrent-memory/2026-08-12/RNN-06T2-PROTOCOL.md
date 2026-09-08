# RNN-06T2 — Prospective Requalification Protocol

**Document ID:** `experiment.local-llm.rnn-06t2.requalification-protocol-2026-08-12`  
**Document type:** `experiment`  
**Status:** `PROPOSED`  
**Execution state at publication:** `IN_PROGRESS_EXTERNALLY / NO_RESULT_MINTED_HERE`  
**Authority:** experimental protocol only.  
**Exact supplied protocol SHA-256:** `3cd04d0ca56bdc3ff2c980fff1cb806534114fe1dade6b288f134c40ea1de5b4`

This document records the scientific contract supplied after the RNN-06T independent audit. The exact full prompt is preserved outside this normalized publication by the SHA above. The research repository does not infer a result from the fact that implementation is currently running.

## Hard dependency structure

```text
RNN-06T2-T0R
fresh official-Mamba fixed-batch lifecycle requalification
        ↓ HARD GATE
RNN-06T2-T1R
fresh recovery confirmation + corrected economics
```

T1R may execute only if both:

```text
OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE = QUALIFIED
SINGLE_PASS_HISTORICAL_CAPTURE_T0R = QUALIFIED
```

Otherwise `RNN-06T2-T1R = BLOCKED_BY_T0R` and the train stops.

## T0R contract

Use the same pinned official subject unless live repository evidence proves it unavailable:

```text
state-spaces/mamba2-1.3b
revision c5b59d00ec85d313adea86a08cad2a43c962dd3b
mamba_ssm 2.2.4
causal_conv1d 1.5.0.post8
torch 2.6.0+cu124
bf16
```

The new contract explicitly separates:

1. `FIXED_BATCH_REQUEST_ISOLATION`
   - neighbor order;
   - neighbor content/identity;
   - row permutation;
   - state + continuation/readout under frozen batch shape.

2. `BATCH_SHAPE_NUMERICAL_PORTABILITY`
   - batch1 vs batchB is a separate property;
   - the historical `max_abs_diff=0.5` remains negative evidence;
   - the new preregistration must declare prospectively whether portability is required or `OUT_OF_SCOPE_NOT_QUALIFIED` for a fixed-batch recovery contract.

The lifecycle qualification set must be fresh and disjoint from RNN-06A/A2/06T T0.

Required tests include:

- deterministic same-path replay;
- destroy/reload/restore/continue;
- substantive branch/fork equivalence and isolation with no tautological checks;
- fixed-batch neighbor isolation;
- reset → reuse → fresh-cache continuation comparison;
- serialize → destroy → restore → continue comparison;
- fixed-batch slice ownership;
- temporal snapshot identity against independent same-path replay;
- checkpoint identity plus a separately named loaded-weight mutation sentinel;
- proof that the intended official fast-path kernels actually fire.

Single-pass historical capture must use actual states captured from one canonical run; independent prefix re-prefills do not satisfy the claim.

T0R mints separately:

```text
OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE
BATCH_SHAPE_NUMERICAL_PORTABILITY
SINGLE_PASS_HISTORICAL_CAPTURE_T0R
```

No failed historical gate is rewritten.

## T1R contract — conditional

`MAX_CONFIDENCE` is frozen. No selector tournament is allowed.

### Narrow replication

If retained for formal comparability:

```text
M=192
target band=[8,64]
schedule=[38,76,115,153]
```

Required primary controls include `FINAL`, `FIXED_SLOT_76`, frozen `MAX_CONFIDENCE` and oracle diagnostics.

### Wide-target confirmation

A fresh calibration set and disjoint qualification set are required. The previous `[8,144]` band and `[38,76,115,153]` schedule may be reused, but the preregistered interpretation must be corrected:

```text
NOT_YET_WRITTEN
↔ SEEN_AND_RETAINED
↔ ALREADY_FORGOTTEN
```

Slot153 is after every possible target in `[8,144]`; therefore the scientific claim is not that no fixed state has seen every target.

If calibration fixed controls tie or are equivalent under a preregistered tolerance, all tied controls should be carried into qualification, or a deterministic tie-break must be frozen before outcomes. The adaptive selector is gated against the strongest preregistered fixed control.

Recovery/harm denominators, per-example recomputation material and the scored-value token mapping must be packaged.

## Corrected economics

Every compared arm must execute the same semantic work:

```text
same context
+ same target query
+ same constrained answer readout
```

Required baselines:

- `FINAL_FUSED_EQUIVALENT_WORK`
- `FINAL_STEP_EQUIVALENT_WORK`
- `RECOVERY_ENABLED_EQUIVALENT_WORK`

Raw warm latency samples and dispersion are required; a cost gate cannot rely only on one median sitting marginally inside a threshold. Compile/autotune, cold and warm phases must remain separate.

The old `1000 ms/query` threshold is not automatically inherited as scientific authority.

## Stop boundary

No realistic-workload discovery, Qwen, trained reader/selector, DART, StateX, Sparse Delta Memory, GDN-2, INT8 archive, ReplaySSM or serving integration belongs in this train.

The next research stage opens only after independent audit accepts RNN-06T2.
