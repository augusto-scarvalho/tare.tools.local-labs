# Rejected/HOLD recovery wave handoff — 2026-08-28

This handoff records execution and independent closeout of the first recovery
wave after the 43-packet GPT-5.6 Sol xhigh audit. The canonical state remains
`config/research_backlog.json`; the ordered machine-readable queue is
`config/research_audit_queue_2026-08-28_recovery.json`.

Independent review was completed sequentially by a fresh GPT-5.6 Sol xhigh
instance. The full recomputation ledger is
[`research/INDEPENDENT_AUDIT_LEDGER_2026-08-28_RECOVERY_GPT56_SOL_XHIGH.md`](research/INDEPENDENT_AUDIT_LEDGER_2026-08-28_RECOVERY_GPT56_SOL_XHIGH.md).

## Outcome first

All six successor packets were independently reviewed. Two were promoted,
three were held fail-closed, and one received a bounded scientific rejection.
The review recovered two false negatives and found scorer false positives that
were not visible in the executor's aggregate metrics.

| Packet | Independent disposition | Audited conclusion |
| --- | --- | --- |
| `BACKLOG-ADAPT-MECHANISMS-COMPLETE-02` | `BLOCKED` / `HOLD_FAIL_CLOSED` | Executor false negative confirmed: eight arms omit a redundant per-arm seed field, while digest-bound `seed.json` and five frozen commands establish seed 20260827. The packet remains superseded by R3. |
| `BACKLOG-ADAPT-MECHANISMS-COMPLETE-03` | `PROMOTED` | The seed-only correction changes no row, arm, score, threshold or estimand. All 768 rows and 16 arms reproduce; `lokr_5ep` is 16/32 math and 5/16 QA. |
| `BACKLOG-ADAPT-TRACE-DEPLOY-RESCORE-02` | `BLOCKED` / `HOLD_FAIL_CLOSED` | Stored 76/256 versus 105/256 and CI [0.050781, 0.175781] reproduce, but the scorer fails 4/4 same-unit adversarial probes and has observed in-panel false positives and negatives. No deployment claim is signed. |
| `BACKLOG-QWEN38-Q8-KV-UTILITY-02` | `BLOCKED` / `HOLD_FAIL_CLOSED` | Physical F16/Q8 execution, throughput ratio 0.977282 and 872 MiB saving are valid. One incidental-number false positive changes Q8 from 37/128 to 36/128; corrected CI [-0.046875, 0.015625] still passes, but the receipt requires a provenance-bound rescore. |
| `BACKLOG-AGY-SYSTEM-BLOCKERS-03` | `PROMOTED` | Whole-tree inspection confirms a predecessor false negative: immutable commit `87a416bd` materially implements bounded GDN snapshot-to-cache fusion. This is source materialization only, not build, deployment or performance qualification. |
| `BACKLOG-MTP-PERSISTENCE-FUNCTIONAL-03` | `REJECTED` | Sixteen valid cache files and exact continuations coexist with `cache_n=0` in 16/16 cycles, identical 6,739-token cold/warm prefill and zero prefill reduction. Serialization exists; functional restored-prefix reuse does not for this route. |

## Independent-review protocol used

For every packet, the reviewer verified the receipt SHA-256 and fingerprint
from the queue, recomputed gates from raw evidence, inspected the implementation
digest, and looked for both false positives and false negatives. The packets
were not batch-signed.

Specific adversarial checks:

1. For trace rescore, test whether question-unit ranking can select incidental
   same-unit numbers. Confirm the scorer signature receives only question and
   text, then recompute all 512 labels and the 20,000-replicate bootstrap.
2. For Q8, inspect actual process argv for both K and V cache types, recheck
   model/binary identities, and independently rescore the paired panel. The
   claim is utility noninferiority on this panel, never exact parity.
3. For AGY blockers, inspect the exact commit, not the dirty worktree. SLX-03 is
   bounded to direct GDN snapshot cache fusion; it does not prove the historical
   N16/EOS cadence or hardware counter reduction.
4. For MTP, distinguish save/restore byte success from computational reuse.
   Exact output after full re-prefill is nondiscriminating; `cache_n` and prefill
   work are the decisive negative evidence.
5. For ADAPT, audit R2 before R3. R2 should expose the executor false negative;
   R3 may pass only if it changes no row, threshold, arm, score or service claim.

## Claim boundaries

- Trace: bounded to Qwen3.5-0.8B, seed 20260832, the frozen third panel and the
  fixture-validated scorer. It is not a fourth-panel replication or production
  deployment decision.
- Q8: bounded to Qwen3.8, this 128-task panel, one fresh process per arm and
  the measured RTX 3090 runtime. It does not qualify Q4_0 or exact equivalence.
- SLX-03: source materialization only. No performance or deployed-callability
  qualification is made.
- MTP: rejects functional restored-prefix reuse for the tested slot path; it
  does not reject file serialization or deterministic regeneration.
- ADAPT: completes the frozen five-mechanism matrix at seed 20260827; it does
  not establish cross-seed repeatability or universal mechanism superiority.

## Remaining salvage queue

The immediate no-inference repair queue is:

1. Trace: add frozen same-unit and observed-panel adversarial fixtures, then
   rescore all 512 retained outputs and recompute paired statistics.
2. Q8: provenance-bind a semantic scorer, rescore all 256 retained outputs and
   issue a successor receipt. Do not rerun physical inference unless the raw
   output binding fails.

After those repairs, the next cheap, high-confidence recovery wave should address:

1. `BACKLOG-FLEET-CONTEXT-ENVELOPE-03` and
   `BACKLOG-FLEET-CONTEXT-INTERFERENCE-01`: both reproduced 72/72 and are held
   only by inherited mutable-source identity. Use immutable scientific inputs;
   do not weaken the retrieval constructs.
2. `BACKLOG-FLEET-REGRESSION-SCREEN-01` and gateway route stress: bind final
   runner state before receipt construction and keep route/repeatability claims
   separate from cap-sensitive quality scores.
3. `BACKLOG-SLX11-OFFICIAL-HYBRID-01`: retain fresh logits for the 24 frozen
   forwards so the topology result can be independently recomputed.
4. `BACKLOG-ADAPT06-SLOP-LIVE-05`: digest-bind cache/schedule rows and compare
   switched outputs with route-correct isolated counterfactuals.

## Operational baseline

At audit close, gateway 8080 serves `qwen38`, the backend is healthy, embedding
8081 is healthy, the pipeline gate passes, and all experiment watchers exited.
The independent reviewer performed no commit or push; repository publication
is a separate maintainer action.
