# RNN-06C-MAMBA-HISTORICAL-INFO — BLOCKED_BY_06B2 (not executed)

**Status: `RNN-06C = BLOCKED_BY_06B2`. No outcome-bearing 06C was executed.**

## Why

RNN-06C is CONDITIONAL on Backlog Item 1 qualifying. Its upstream gate did not open:

> `FIXED_LENGTH_STATE_LOAD_REGION = BLOCKED` (reasons `IMMEDIATE_CLIFF` +
> `NOT_ROBUST_ACROSS_STRATA`) — see `../RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD/B2_DECISION.md`.

Per train §9 / §16 / the dependency gate: BLOCKED upstream ⇒ do NOT execute 06C. No
`historicalInfoSetSha256` challenge set was generated; no state-readout runs were performed;
no `HISTORICAL_STATE_INFORMATION` verdict is minted; no recovery mechanism was built; no reader
was trained.

## Scientific reason it is correct to stop here

06C asks whether target information present in an EARLIER recurrent state is no longer available
in the FINAL high-load state. That question presupposes a **qualified, stable high-load regime**
in which the target is reliably *lost at the final state* across a graded, robust region. B2
found a real but **cliff-like, non-robust** state-load effect that reaches material loss only at
the fully-packed endpoint (U=128 = M) and populates only one interior mid-band dose. Running 06C
on a non-qualified regime would rest the historical-information contrast on an unstable,
boundary-dominated operating point — exactly the kind of unqualified substrate the dependency
gate exists to forbid. The honest action is to stop.

## What was preserved for a future, independently-qualified attempt

- The frozen 06A2 continuation/lifecycle semantics (checkpoint/restore BIT_EXACT) that 06C's
  state-readout design depends on remain qualified and available.
- The predeclared 06C dose-selection rule (HIGH=max load; LOW=largest DS≥0.80; MID=midpoint)
  is recorded in `B2_RESULTS.json → c06_dose_selection` with `applicable=False`.
- No snapshot files were written; snapshot economics (52,002,816 B/seq) carried forward
  unmeasured for 06C.

This artifact exists so the train bundle explicitly represents the blocked dependency rather
than silently omitting 06C.
