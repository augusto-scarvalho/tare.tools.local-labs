# LAB-CODE-004 prompt-only loop guard result

Decision: **REJECT AT STAGE A / 2 OF 3 SUBMITTED**  
Date: 2026-08-22

The single system-prompt instruction reproduced the two prior controls with the same patch sizes
(504 and 920 bytes), but `django__django-13401` again exhausted all 40 calls with an empty patch.
Its final exact command repeated 21 consecutive times. The frozen gate therefore stopped before the
remaining seven instances and no official correctness evaluation was spent.

This rejects a prompt-only fix for the observed deterministic action loop. It does not alter or rescore
LAB-CODE-003. The next falsifiable lever is a middleware guard that blocks execution of a third
consecutive identical action and returns an explicit observation requiring a different action.

Predictions SHA-256: `fd1a00eb95db3ca21923d461e65594462fe1b9fd2550080688eb9eba1c02bbba`.
The separate pre-model Docker CLI failure is preserved in `INFRA_ATTEMPT.md` and excluded.
