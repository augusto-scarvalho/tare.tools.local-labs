# LAB-CODE-005B command-only duplicate guard preregistration

Status: **FROZEN BEFORE GENERATION**  
Date: 2026-08-22

LAB-CODE-005 is invalid because unique tool-call IDs contaminated its duplicate signature. This
corrective arm keeps the intended single lever and signs only the ordered `command` strings. A third
successive identical command is not executed and instead receives `LOOP_GUARD_BLOCKED` with return
code 125. The corrected self-test requires different tool-call IDs to collide when commands match.

Everything else remains identical to LAB-CODE-003. Stage A and the frozen decision rule remain:
all first three must submit; PROMOTE at >=6/10 resolved, >=6/10 submitted and >=4/5 resolved among
prior submitters; HOLD for improved submissions below that rule; otherwise REJECT.
