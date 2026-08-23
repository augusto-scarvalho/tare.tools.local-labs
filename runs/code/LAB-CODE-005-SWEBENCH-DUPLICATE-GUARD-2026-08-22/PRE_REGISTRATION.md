# LAB-CODE-005 duplicate-action middleware guard preregistration

Status: **FROZEN BEFORE GENERATION**  
Date: 2026-08-22

LAB-CODE-004 showed that a prompt-only instruction did not stop an exact command from repeating 21
times. This arm reverts to the exact LAB-CODE-003 prompt/config and changes one lever: after two
successive executions of an identical action batch, the middleware blocks the third and later identical
execution with return code 125 plus an explicit `LOOP_GUARD_BLOCKED` observation. No shell command is
run for a blocked action.

The dataset, ordered ten IDs, model/runtime, Docker environments, sampling, 2,048-token response cap,
40-call cap, one trajectory and official evaluator remain identical. Stage A is the same first three
IDs; all must submit nonempty patches before the remaining seven open. Empty predictions remain failures.

Decision rule: **PROMOTE** at >=6/10 resolved, >=6/10 submitted and >=4/5 resolved on the prior
submitters; **HOLD** if submission improves without meeting promotion; **REJECT** if Stage A fails,
submission does not improve or prior-submitter resolution falls below 4/5.
