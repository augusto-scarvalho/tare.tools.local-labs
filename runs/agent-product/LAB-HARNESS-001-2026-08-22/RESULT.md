# LAB-HARNESS-001 result

## Decision: PASS - bounded orchestration primitives qualified

The pilot passed its frozen gates. `TaskContract` digest-bound deltas preserve
the objective, constraints and required-test set; stale and cross-contract
deltas fail closed. The test-baseline gate rejects both a formerly passing test
that regresses and a baseline test that disappears.

The structural `RepositoryEvidencePack` recalled all five required files. Using
the exact incumbent Qwen3.8 tokenizer, it reduced the frozen full-file controls
from 23,238 to 3,753 tokens (83.85% aggregate). Every task cleared the 30% gate:

| Task | Required recall | Control tokens | Pack tokens | Reduction |
|---|---:|---:|---:|---:|
| mode | yes | 7,528 | 748 | 90.06% |
| provenance | yes | 2,200 | 701 | 68.14% |
| visual | yes | 2,931 | 747 | 74.51% |
| agent | yes | 7,071 | 574 | 91.88% |
| context | yes | 3,508 | 983 | 71.98% |

Five unit tests passed, including stale-delta, cross-contract, missing-test and
regression sentinels. Machine-readable evidence is in `results.json`.

## Boundary

This is a retrieval/contract/gating qualification, not a full coding-agent
product qualification. Independent model-written mutation tests, calibrated
critic judgments, retry telemetry and integration into the absent tare.tools
product repository remain separate work. No model-serving or deployment default
was changed.
