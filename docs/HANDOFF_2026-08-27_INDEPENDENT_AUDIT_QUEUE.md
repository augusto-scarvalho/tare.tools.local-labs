# Handoff: independent audit queue and independent closeout

Date: 2026-08-27  
Repository: `tare.tools.local-labs`  
Canonical manifest: `config/research_backlog.json`  
Machine-readable audit order: `config/research_audit_queue.json`  
Executor lineage: Codex  
Required independent reviewer: AGY or another genuinely independent actor

## Purpose

This was the restart point for the independent audit. The 43-packet snapshot
has now been processed in its original machine-readable order. The original
protocol and executor-reported focus remain below as historical audit context.

Do not reconstruct the queue from chat transcripts, the old 52-record handoff,
or only the last experiment wave. The manifest is authoritative for state. The
JSON audit queue is a review-order snapshot, not a second state machine.

## Independent audit closeout - 2026-08-28

Reviewer: `GPT-5.6 Sol xhigh independent audit instance`  
Ledger: `docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md`

| Independent disposition | Count |
|---|---:|
| Queue packets processed | 43 |
| `PROMOTED` by this audit | 10 |
| `REJECTED` by this audit | 14 |
| `HOLD_FAIL_CLOSED` | 19 |
| Pending in this 43-item queue | 0 |

The manifest now contains 19 `EXECUTED`, 32 `BLOCKED`, 13 `PROMOTED` and 16
`REJECTED` records. The 19 held packets deliberately remain `EXECUTED`; their
audit disposition exists only in the ledger and regenerated audit-queue
metadata because a digest mismatch, missing recomputable evidence, construct
ambiguity or lack of a legal transition prevented signing. No `REVIEW.json`
or state transition was fabricated for those holds, except the already
documented item 15 review whose scientific rejection could not be represented
by the state machine because all receipt gates were marked pass.

`config/research_audit_queue.json` now retains the original order and records
the exact disposition for every item. The canonical manifest remains the sole
source for pipeline state. Final gate and test observations belong in the
ledger rather than being inferred from this historical handoff.

## Original verified snapshot

| State | Count |
|---|---:|
| Total manifest records | 80 |
| `EXECUTED`, independent audit pending | 43 |
| `BLOCKED` | 32 |
| `PROMOTED` | 3 |
| `REJECTED` | 2 |
| Dependency-ready `PROPOSED` | 0 |

At handoff creation:

- `backlog_pipeline.py gate`: `PASS`;
- repository tests: `256 passed`;
- `backlog_pipeline.py next`: no dependency-ready proposed item;
- Git branch/HEAD: `master` at
  `9d46725a5f9e58eff4172ec46ea4bd1eb0b84a02`;
- worktree: already dirty, with 118 status entries before this handoff;
- gateway 8080: healthy, `qwen38` resident, backend healthy;
- embedding 8081: healthy.

These observations are time-sensitive. Recheck them; do not copy them into an
audit receipt as if they were newly observed.

## Authority boundary

The 43 packets below were executed by the Codex lineage. Codex stopped at
`EXECUTED`, did not write their `REVIEW.json`, and did not promote or reject
them. AGY may independently review them.

The success path is:

```text
EXECUTED -> VERIFIED -> PROMOTED
        +-> REJECTED
        +-> BLOCKED only for a genuine external audit blocker
```

A failed preregistered scientific gate is evidence and normally leads to
`REJECTED`; it is not an operational blocker. Passing all gates permits review,
not an automatic promotion. Every decision must stay inside the packet's
`allowed_claim_codes`.

## Sources of truth for each packet

Read and bind these in order:

1. The item in `config/research_backlog.json`.
2. `runs/research/<ID>/PRE_REGISTRATION.md`.
3. `runs/research/<ID>/PIPELINE.json`, especially `implementation_digest`.
4. `runs/research/<ID>/raw/receipt.json` and its canonical fingerprint.
5. Every raw evidence path named by the receipt.
6. `runs/research/<ID>/RESULT.md`, as a bounded executor narrative only.
7. `runs/research/<ID>/REVIEW.template.json` when authoring the independent
   `REVIEW.json`.

Never approve from `RESULT.md` alone. Recompute metric values, operators,
thresholds, sample counts and scorer outcomes from raw evidence.

## Audit queue A: conclusion-changing successor chains

Review each packet separately, but keep each chain together so that a later
successor is not mistaken for a silent rewrite of its predecessor.

| Order | Packet | Executor-reported audit focus |
|---:|---|---|
| 1 | `BACKLOG-ADAPT-REQUAL-02` | Process-isolated adapter requalification; all gates passed, but the historical `target_mlp_only` ranking did not reproduce. |
| 2 | `BACKLOG-ADAPT-MECHANISMS-RERUN-01` | Mixed ADAPT-01..05 rerun; receipt reports failed coverage and service-restore gates. Do not accept the narrative label without resolving those gates. |
| 3 | `BACKLOG-ADAPT01-640-EVAL-01` | 640-step LoKr arm failed natural-EOS and length gates. |
| 4 | `BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-02` | First broad LoKr panel showed +8.59 pp but failed evaluation coverage; R3 is its completion successor. |
| 5 | `BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-03` | Completed 256-math/48-QA panel; +8.59 pp with positive paired-bootstrap lower bound and no failed gates. |
| 6 | `BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-04` | Second panel gained +5.86 pp, but its confidence interval crossed zero. |
| 7 | `BACKLOG-ADAPT01-BROAD-ARTIFACT-EVAL-05` | Frozen three-panel synthesis gained +9.38 pp, CI `[+5.73,+13.15]` pp; family stops here. |
| 8 | `BACKLOG-ADAPT-TRACE-DISTILL-03` | First matched trace false-negative successor; inspect the smaller three-seed boundary before later replications. |
| 9 | `BACKLOG-ADAPT-TRACE-DISTILL-07` | Seven paired seeds, 504 steps, +14.17 pp mean, all seven positive, no failed gates. |
| 10 | `BACKLOG-ADAPT-TRACE-DISTILL-08` | Second teacher-disjoint panel replicated +14.45 pp with all seven seeds positive. |
| 11 | `BACKLOG-ADAPT-TRACE-VS-FINALIST-01` | Across two panels/seven trace seeds, trace exceeded reproduced behavioral finalists by +6.85 pp; CI lower bound positive. |
| 12 | `BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01` | Seed 20260832 was selected before panel three and beat answer-only by +7.81 pp, but absolute accuracy 38.28% missed the 40% finalist gate. |
| 13 | `BACKLOG-ADAPT-TRACE-VS-FINALIST-02` | Selected trace beat the two behavioral checkpoints by +4.10 pp pointwise, but hierarchical CI `[-2.73,+11.13]` pp crossed zero; practical-superiority gate failed. |
| 14 | `BACKLOG-MTP-PERSISTENCE-01` | Original runner reported universal failures and failed its original-failure, controls and fixed-repeat gates. Audit together with the telemetry rescore. |
| 15 | `BACKLOG-MTP-PERSISTENCE-RESCORE-02` | Immutable evidence rescored as 44/44 physical successes; historical 0/44 was caused by reading `timings.cache_n` instead of top-level `tokens_cached`. |
| 16 | `BACKLOG-QWEN38-KV-PRECISION-01` | Q4 KV saved 1,384 MiB but exact answer parity was 75%; semantic gate failed. |
| 17 | `BACKLOG-QWEN38-KV-PRECISION-02` | Q8 KV saved 872 MiB at 0.9997x throughput; 84.38% literal parity still failed the frozen exact-parity gate. |
| 18 | `BACKLOG-QWEN38-Q8-KV-UTILITY-01` | Successor tests actual task utility: F16/Q8 38/128 vs 37/128, noninferiority CI `[-4.69,+2.34]` pp, 865 MiB saved. All utility gates passed. |

Important synthesis boundaries:

- LoKr R5 supports a pooled three-panel gain for exactly the frozen artifact;
  it does not establish universal LoKr superiority.
- Trace training replicated as an average treatment effect, but the selected
  deployment seed did not clear both practical-selection gates on panel three.
- Q8's literal output identity failed, while its separately preregistered task
  utility passed noninferiority. These are different hypotheses, not permission
  to weaken the earlier gate after seeing the data.
- The MTP rescore can confirm a telemetry false negative without proving that
  intermittent persistence can never fail under other conditions.

## Audit queue B: qualified-fleet evidence

| Order | Packet | Executor-reported audit focus |
|---:|---|---|
| 19 | `BACKLOG-GATEWAY-ROUTE-STRESS-01` | 30/30 switches and 120/120 requests, exact repeat rate 1.0. |
| 20 | `BACKLOG-FLEET-REGRESSION-SCREEN-01` | 448 requests over four routes, success and exact-repeat rates 1.0; quality values are descriptive. |
| 21 | `BACKLOG-FLEET-SEEDED-STABILITY-01` | 288 requests, exact seeded repeat rate 1.0. |
| 22 | `BACKLOG-FLEET-CONTEXT-ENVELOPE-03` | Four routes each passed 18/18 bounded retrieval cases across position buckets. |
| 23 | `BACKLOG-FLEET-CONTEXT-INTERFERENCE-01` | Four routes each passed 18/18 with 31 hard decoys per case. |
| 24 | `BACKLOG-FLEET-MBPPPLUS-01` | First 100-task panel failed HauhauCS noninferiority. |
| 25 | `BACKLOG-FLEET-MBPPPLUS-02` | Frozen 200-task synthesis: HauhauCS 149/200 vs Qwen3.8 148/200; noninferiority passed. No third MBPP panel is allowed. |
| 26 | `BACKLOG-FLEET-HUMANEVALPLUS-02` | Initial full HumanEval+ run reported HauhauCS/Qwen3.8 tied at 146/164. |
| 27 | `BACKLOG-FLEET-HUMANEVALPLUS-03` | Bounded truncation correction produced 147/164 vs 148/164, but five rows remained truncated and the recovery gate failed. Family stops here. |

Do not turn route stability, bounded retrieval, or benchmark panels into a
general model-quality ranking. Preserve each packet's route, token cap,
context, scorer and task-family scope.

## Audit queue C: prior systems and mechanism packets

These 16 packets were already described in detail by
`docs/HANDOFF_2026-08-26_CONSOLIDATED_RESEARCH_BACKLOG.md`. They remain
`EXECUTED`; that older document did not constitute independent review.

| Order | Packet | Gate posture to verify |
|---:|---|---|
| 28 | `BACKLOG-CUDAGRAPH-SERVING-02` | Failed paired-speedup threshold; explicit OFF/ON treatment supersedes the old 1.5115x interpretation. |
| 29 | `BACKLOG-NEGATIVE-KV-REAL-SCREEN-02` | Multiple frozen mechanism gates failed on real Qwen tensors; retain only bounded negative conclusions. |
| 30 | `BACKLOG-SLX08-REAL-FIDELITY-01` | Fidelity gates passed; selected-block TTFT integration remains outside scope. |
| 31 | `BACKLOG-HYPER01-REAL-ADAPTER-01` | Physical adapter screen failed overhead gate. |
| 32 | `BACKLOG-DISTILL01-FLEET-REAL-01` | Failed fleet-gain and math-specialist gates. |
| 33 | `BACKLOG-CTRL01-REAL-TOKEN-06` | Failed validity, control and runtime-binding gates. |
| 34 | `BACKLOG-RSH02-PACKED-GPU-02` | Exact decode but failed compression, throughput and penalty gates. |
| 35 | `BACKLOG-GDN02-LEARNED-STATE-01` | Failed target-leakage and update-fidelity gates. |
| 36 | `BACKLOG-BEE-L4-LIVE-MTP-01` | All bounded observable slot-isolation gates passed. |
| 37 | `BACKLOG-BEE-L5-LIVE-GUARD-04` | Behavior passed; guard-overhead gate failed. |
| 38 | `BACKLOG-ADAPT06-SLOP-LIVE-05` | All gates passed for real LoRA routing plus client affinity, not a server-native scheduler. |
| 39 | `BACKLOG-BEE-L3-REAL-TELEMETRY-01` | Failed semantic-parity and static-gain gates. |
| 40 | `BACKLOG-AGY-SYSTEM-BLOCKERS-02` | Registers six absent integrations; does not falsify their algorithms. |
| 41 | `BACKLOG-SLX10-PACKED-RUNTIME-02` | Failed packed-artifact, quality and semantic-stability gates. |
| 42 | `BACKLOG-SPEC01-LIVE-HYBRID-01` | Failed historical-speedup and proposer-attribution gates. |
| 43 | `BACKLOG-SLX11-OFFICIAL-HYBRID-01` | All artifact/architecture gates passed; no historical speed or recall claim. |

## Independent review procedure

For each ID, one at a time:

1. Confirm it is still `EXECUTED` in the live manifest.
2. Hash `raw/receipt.json` and verify its canonical fingerprint.
3. Verify every implementation file against `implementation_digest` in
   `PIPELINE.json`.
4. Resolve every required evidence role and independently recalculate stored
   metrics, scorer decisions, sample counts and gate booleans.
5. Inspect model/runtime/treatment identity and source hashes. For service
   experiments, verify restoration rather than trusting a healthy endpoint
   observed later.
6. Enforce abort conditions and the narrowest allowed claim code.
7. Write `REVIEW.json` from `REVIEW.template.json`, binding the exact receipt
   SHA-256 and implementation digest.
8. If approved and every mandatory gate passes, advance to `VERIFIED` with the
   allowed success claim, then to `PROMOTED` only if the evidence class permits.
9. If a mandatory scientific gate fails and the evidence is sound, advance
   from `EXECUTED` to `REJECTED` with the allowed bounded rejection claim.
10. Rerun the pipeline gate after every decision batch. Never edit
    `research_backlog.json` or `PIPELINE.json` directly.

The exact transition command must use the claim listed in the live manifest;
do not copy a claim from this narrative without checking it.

## Startup commands for the reviewer

Run from `C:\projects\tare.tools.local-labs`:

```powershell
git status --short
git rev-parse HEAD
python tools/analysis/backlog_pipeline.py gate
python tools/analysis/backlog_pipeline.py status
python tools/analysis/backlog_pipeline.py next
Get-Content config/research_audit_queue.json
python -m pytest -q
python tools/agents/modelctl.py status --json
```

Then start with `BACKLOG-ADAPT-REQUAL-02` unless live evidence reveals drift.

## Execution incidents and preserved evidence

- Several predecessor IDs remain `BLOCKED` because they are immutable aborted
  attempts with executed successors. Do not count them as new experiments or
  delete them.
- `BACKLOG-ADAPT-TRACE-VS-FINALIST-02` had one zero-generation preflight abort
  because the reused worker rejected the label `behavioral`. The corrected
  runner was rebound through legal state transitions and the full 512 fresh
  generations were then executed. Both watcher records remain under
  `runs/autonomous/`.
- The workspace contains extensive uncommitted experiment artifacts from the
  autonomous waves. They are intentional evidence. Do not clean, reset or
  overwrite them while auditing.
- No commit or push was performed as part of the final execution wave or this
  handoff.

## Stop conditions

Stop and report instead of deciding a packet if:

- the pipeline gate fails;
- receipt SHA-256/fingerprint or implementation digest does not bind;
- a required evidence file is absent or mutated;
- raw recomputation disagrees with the receipt;
- reviewer independence is not genuine;
- the intended claim exceeds the allowed claim codes or evidence class;
- a passing decision would require weakening a frozen threshold;
- a historical raw packet would need modification;
- service maintenance would disturb 8081 or overlap large generation models.

Create a successor packet for a real correction. Do not repair executor
evidence inside an independent review.

## Completion definition

This audit queue is closed only when all 43 current `EXECUTED` packets have an
independent disposition, all superseded human-facing conclusions are labeled
explicitly, the pipeline passes, and the machine-readable queue is regenerated
from the resulting manifest. The 32 `BLOCKED` records remain a separate
trigger/archival reconciliation problem; they do not become executable merely
because the audit queue finishes.

The 43-item independent pass met this closeout definition on 2026-08-28: the
ledger records every disposition and claim boundary, the regenerated queue has
zero pending entries, the final pipeline gate passed and the full suite reported
256 passing tests. The 19 fail-closed holds remain intentionally unresolved
execution records and require successor evidence, not edits to historical raw
packets.
