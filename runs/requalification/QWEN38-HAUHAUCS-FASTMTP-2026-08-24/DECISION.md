# HauhauCS native MTP versus FastMTP — dependency-gate decision

## Decision

`NO-GO BEFORE INSTALL`.

No FastMTP source, patch, sidecar, executable, or model artifact was downloaded,
installed, built, or executed. Native MTP remains the only speculative decoder
used by the qualified HauhauCS profile.

## Why the gate is closed

The candidate's frozen qualification contract allowed review of the third-party
FastMTP patch only after all earlier quality, alignment, termination, and
serving gates passed. Subsequent bounded work now shows:

- agent/tool core: HauhauCS 8/8, positive;
- normal Portuguese questions: a raw language-adherence loss, mitigated by the
  frozen locale contract;
- GSM8K-200: HauhauCS 191/200 versus Fable 195/200, with 8/200 truncations and
  192/200 format adherence;
- frozen math verdict: `MATERIAL_MATH_LOSS` under the 512-token operational
  contract because termination exceeded the allowed defect rate.

FastMTP could only change serving mechanics; it does not repair the demonstrated
candidate-level output-termination contract. Opening a third-party patch and
maintenance surface before the broad-default quality gate passes would invert
the declared dependency order.

## Existing native-MTP baseline retained

The qualified 131k HauhauCS profile already measured:

- 91.37 tok/s median over five forced 256-token coding generations;
- 940/995 accepted draft tokens (94.47%);
- b10165 commit `71676e46c`;
- embedded GGUF NextN/MTP head, draft n3.

That baseline remains sufficient for an optional coding-focused profile. A
FastMTP A/B may be reconsidered only after a new, separately pre-registered
termination mitigation passes held-out evidence and the candidate again becomes
a realistic promotion target.

## Operational effect

None. Fable-TC remains active on 8080; embeddings and locale proxy remain active
on 8081/8082. No reboot, service mutation, commit, or push was performed for
this decision.
