# LAB-COLD-FUSION-002 — embedded-MTP result

Decision: **MTP_REJECTED / BASE ROLE REMAINS REJECTED**  
Executed: 2026-08-22

All nine counterbalanced server cells loaded and generated successfully beside the embedding service. Every
task/arm was deterministic across its three replicates, but the embedded MTP arms failed the frozen
task-correct equivalence and uniform-speed gates.

| Task | Off tok/s | n2 tok/s | n2 delta | n3 tok/s | n3 delta | Correct off/n2/n3 | MTP byte-equal |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact arithmetic | 48.79 | 41.46 | -15.03% | 34.43 | -29.45% | 3/3 · 3/3 · 3/3 | yes |
| bounded code | 42.21 | 71.85 | +70.24% | 72.72 | +72.29% | 3/3 · 3/3 · 3/3 | **no** |
| red-black prose | 41.81 | 61.14 | +46.24% | 60.40 | +44.48% | 0/3 · 0/3 · 0/3 | **no** |

The prose arm reached the exact 384-token cap in all nine runs before covering the required root-handling
oracle. That is a task/budget failure, not a runtime failure. The first summarizer incorrectly collapsed
task correctness into cell validity and printed `BLOCKED_RUNTIME`; the raw `mtp-ab.partial.json` retained all
nine cells. Offline reanalysis separated those concepts and supersedes that preliminary label.

Acceptance was workload-sensitive. n2/n3 accepted 1/6 and 1/9 drafts on the five-token arithmetic answer,
27/32 and 30/39 on code, and 218/328 and 243/417 on prose. This explains why speculation hurts tiny outputs
while helping longer predictable continuations. It does not rescue promotion because output equivalence is a
hard gate and the base artifact already failed context, MBPP and GSM role qualification.

Residency was stable at approximately 17,765 MiB off, 18,329 MiB n2 and 18,479 MiB n3 including the resident
embedding service. Port 8092 was closed after every cell and at campaign end; 8081 remains healthy. Canonical
8080 remains intentionally stopped for the ongoing authorized LAB queue.

## Evidence seals

- Frozen packet SHA-256: `614bc17d724aa5caa6e63a29a16c7e225334a4644356ce379d8390c897474295`.
- Final JSON SHA-256: `095a9a2317e72557f06adbad61f96cdcbf37933a0e29223068402aa945e7dbf4`.
- Raw partial JSON SHA-256: `93de74184c1c9796d64d249099ee371e7d3f8ae1ff082c30e9db8978e5624d60`.
- Harness SHA-256: `305db43f16a3871f11287e63ad7bba5402f05d4ad9d16b4fdd31d713c55540fc`.

