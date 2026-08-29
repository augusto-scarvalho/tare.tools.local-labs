# Backlog checking and implementation pipeline

## Purpose

The current remediation backlog is executable policy, not a prose checklist.
The pipeline prevents the failure modes found in the Gemini wave: fabricated or
random decisive fields, proxy results promoted as hardware evidence, missing
provenance, mutable receipts, unregistered work, changed thresholds after a
run, dependency bypass and self-review.

The canonical machine-readable queue is
[`config/research_backlog.json`](../../config/research_backlog.json). The CLI is
[`tools/analysis/backlog_pipeline.py`](../../tools/analysis/backlog_pipeline.py),
and CI runs its repository-wide gate on every push and pull request.

## Consolidated state on 2026-08-26

The original 15-item queue below has now accumulated provenance-preserving
successors and fail-closed abort packets. The canonical manifest contains 52
operational records: 20 `EXECUTED`, 27 `BLOCKED`, three `PROMOTED`, and two
`REJECTED`. A `BLOCKED` manifest record can be an aborted implementation
attempt followed by a successful successor; it is not automatically a blocked
scientific hypothesis.

The human-facing reconciliation of the 36 AGY claims is maintained separately:

- [`HANDOFF_2026-08-26_CONSOLIDATED_RESEARCH_BACKLOG.md`](../HANDOFF_2026-08-26_CONSOLIDATED_RESEARCH_BACKLOG.md) is the restart-safe operational handoff: review order, genuine blockers, archived failed attempts, integration gaps, service baseline, and stop conditions.
- [`AGY_36_INDEPENDENT_RERUN_TRACKER_2026-08-25.md`](../AGY_36_INDEPENDENT_RERUN_TRACKER_2026-08-25.md) maps every original rank to its decisive successor or objective unblock condition.
- [`AUDIT_2026-08-26_CODEX_AGY_36_FULL_RERUN.md`](../AUDIT_2026-08-26_CODEX_AGY_36_FULL_RERUN.md) consolidates the three false negatives, eight false positives/overclaims, retained results, physical blockers, service restoration, and claim limits.

All 36 rows have a final disposition: 31 have at least one decisive physical
successor and five remain exclusively blocked by an absent implementation.
SLX-08 overlaps both groups because its fidelity gate was corrected while its
selected-block TTFT route remains absent.

## Initial 15-item queue

| Priority | ID | Initial state | Entry condition |
|---|---|---|---|
| P0 | `BACKLOG-ADAPT-REQUAL-01` | `PROPOSED` | Freeze adapter/config, datasets and scorers |
| P1 | `BACKLOG-ADAPT-TRAIN-01` | `BLOCKED` | P0 promotes a finalist |
| P1 | `BACKLOG-DISTILL-REAL-01` | `PROPOSED` | Freeze real teacher/student generation and scoring |
| P2 | `BACKLOG-CUDAGRAPH-SERVING-01` | `PROPOSED` | Identify the actual serving integration point |
| P2 | `BACKLOG-PROXY-REALIZATION-01` | `BLOCKED` | Select exactly one real implementation target |
| P3 | `BACKLOG-PACKED-HARDWARE-01` | `BLOCKED` | Produce an actual packed artifact/runtime |
| P2 | `BACKLOG-ADAPT-TRACE-DISTILL-01` | `BLOCKED` | Promoted reproducible behavioral finalist plus a new hypothesis |
| P1 | `BACKLOG-MTP-PERSISTENCE-01` | `BLOCKED` | Falsifiable cache-lifecycle mechanism hypothesis |
| P2 | `BACKLOG-THINKINGCAP-QWEN38-01` | `BLOCKED` | Official Qwen3.8 ThinkingCap weights and 3090-fit artifact |
| P3 | `BACKLOG-THINKINGCAP-MTP-IDENTITY-01` | `BLOCKED` | Publisher receipt for the exact legacy digest |
| P3 | `BACKLOG-QUANTIZER-PROVENANCE-01` | `BLOCKED` | Exact publisher build receipts |
| P2 | `BACKLOG-HUMAN-JUDGE-CALIBRATION-01` | `BLOCKED` | 50–100 genuine blind human labels |
| P3 | `BACKLOG-RETNET-OFFICIAL-01` | `BLOCKED` | Official Microsoft/TorchScale pretrained checkpoint |
| P3 | `BACKLOG-BEE-L2-KV-CODEC-01` | `BLOCKED` | Physical immutable codec and effective-route receipts |
| P3 | `BACKLOG-APEX4-E2E-01` | `BLOCKED` | Corrected checkpoint shards and reproducible E2E package |

`next` selects the lowest-priority-number ready item and will not cross an
unmet dependency. This makes cheap, discriminating gates precede expensive
training or hardware work.

Portfolio priority can be reassessed independently of scientific state through
the frugal policy documented in
[`BACKLOG_PRIORITY_POLICY.md`](BACKLOG_PRIORITY_POLICY.md). `rank --explain` and
`rebalance --dry-run` are read-only. `rebalance --apply --actor <identity>`
atomically updates only explicitly assessed priority metadata; unassessed items
keep their current P-band. The harness does not own this policy.

These 15 items reconcile the Gemini-remediation queue with the residual trigger
register from the 2026-08-24 closeout. Parked themes, cancelled soaks and closed
`HOLD`/negative results are deliberately not executable manifest items.

## State machine and authority

```text
PROPOSED -> PREREGISTERED -> IMPLEMENTED -> EXECUTED -> VERIFIED -> PROMOTED
     |             |              |             |            |
     +-------------+--------------+-------------+------------+-> BLOCKED
                                               +-> REJECTED
BLOCKED -> PROPOSED
```

- Gemini may preregister, implement and execute.
- The preregistration and implementation are SHA-256-bound before execution.
- The receipt is bound to command, Git state, script, inputs and environment.
- The gate engine compares every receipt metric/operator/threshold with the
  frozen manifest and recomputes `pass` from `actual`.
- `VERIFIED` and `PROMOTED` require a reviewer and transition actor independent
  of Gemini and the executor.
- `proxy_realization` is deliberately non-promotable. A proxy must become an
  eligible real evidence class in a separately reviewed backlog change.
- Original raw receipts are immutable. A correction gets a successor packet.

## Executor workflow

Start with a clean gate and select the task:

```powershell
python tools/analysis/backlog_pipeline.py gate
python tools/analysis/backlog_pipeline.py status
python tools/analysis/backlog_pipeline.py next
python tools/analysis/backlog_pipeline.py scaffold BACKLOG-ADAPT-REQUAL-01 --actor "Gemini 3.7"
```

Complete the generated `PRE_REGISTRATION.md`, including the exact executable
command and no placeholders, then freeze it:

```powershell
python tools/analysis/backlog_pipeline.py advance BACKLOG-ADAPT-REQUAL-01 --to PREREGISTERED --actor "Gemini 3.7"
```

After implementation and its deterministic tests pass, bind every relevant
file. Repeat `--implementation` for each script, module or test:

```powershell
python tools/analysis/backlog_pipeline.py advance BACKLOG-ADAPT-REQUAL-01 --to IMPLEMENTED --actor "Gemini 3.7" `
  --implementation tools/research/run_adapter_requalification.py `
  --implementation tests/test_adapter_requalification.py
```

Run only the preregistered command. It must write
`runs/research/<ID>/raw/receipt.json`, then:

```powershell
python tools/analysis/backlog_pipeline.py advance BACKLOG-ADAPT-REQUAL-01 --to EXECUTED --actor "Gemini 3.7"
```

At this point Gemini writes `RESULT.md` but does not approve the work. It hands
the packet to an independent reviewer.

## Receipt contract

The receipt uses `local-labs-backlog-receipt-v1` and contains:

`scaffold` generates `RECEIPT.template.json` from the task's exact gates and
evidence requirements. Copy that structure into `raw/receipt.json`; do not
retype, remove or rename its keys. It also generates `REVIEW.template.json` for
the independent reviewer.

```json
{
  "schema": "local-labs-backlog-receipt-v1",
  "task_id": "BACKLOG-ADAPT-REQUAL-01",
  "provenance": {"schema": "local-labs-experiment-provenance-v1"},
  "provenance_complete": true,
  "gates": {
    "base_control": {
      "metric": "base_control_present",
      "operator": "eq",
      "threshold": true,
      "actual": true,
      "pass": true
    }
  },
  "evidence": {
    "artifact_hashes": "raw/artifact_hashes.json",
    "raw_samples": "raw/samples.jsonl"
  },
  "receipt_fingerprint": "<canonical SHA-256 of the receipt without this field>"
}
```

All gates and evidence keys required by the task must be present. Extra or
renamed gates fail. The helper in `tools/analysis/experiment_provenance.py`
should be used to construct provenance and the canonical fingerprint.

## Portable large-artifact receipts

Physical GGUF and codec derivatives can exceed GitHub's ordinary object limit.
They remain local and immutable, while their byte count and SHA-256 are stored
in committed packet evidence. If another manifest item declares such a file as
a source, bind it through the canonical CLI before excluding the derivative:

```powershell
python tools/analysis/backlog_pipeline.py externalize-source `
  BACKLOG-SLX10-PACKED-RUNTIME-02 `
  runs/research/BACKLOG-ADAPT06-SLOP-LIVE-05/raw/qwen3.5-0.8b-base-f16.gguf `
  --evidence runs/research/BACKLOG-SLX10-PACKED-RUNTIME-02/raw/artifact_hashes.json `
  --actor "Codex portability consolidation"
```

The command verifies the live file against the committed evidence and writes
an atomic `external_source_receipts` entry. On a portable checkout, `gate`
accepts the absent derivative only when the evidence file contains the exact
same path, digest, and byte count. Missing unbound inputs still fail closed.

## Persistent watcher for background execution

Every experiment launched as a detached or background process must have a
persistent watcher launched in the same operational handoff. A background PID
without a live watcher is not a complete launch.

The watcher must record the experiment PID, packet directory, expected progress
count, pipeline stage, GPU state, relevant endpoint health, and append-only
progress events. When the process exits, it must require a canonical receipt,
validate the repository gate, and may advance only `IMPLEMENTED -> EXECUTED` as
the executor. It must never self-review, verify, reject, or promote a result.

If the receipt is absent, validation fails, or the service baseline does not
recover, the watcher records a fail-closed alert. Each launch reports clickable
paths for the live status, event stream, and final record. The reusable watcher
is `tools/analysis/watch_experiment_processes.py`.
Its complete lifecycle, progress-marker contract, artifact schemas, controller
handoff, failure matrix, fixtures and live-canary runbook are documented in
[`EXPERIMENT_WATCHER.md`](EXPERIMENT_WATCHER.md).

## Independent review and closeout

The reviewer inspects implementation, raw evidence, recomputed gates and scope
of claims. `REVIEW.json` uses `local-labs-independent-review-v1` and binds the
exact receipt SHA-256 plus the packet's `implementation_digest`:

```json
{
  "schema": "local-labs-independent-review-v1",
  "reviewer": "Codex",
  "verdict": "APPROVED",
  "receipt_sha256": "<sha256>",
  "implementation_digest": "<digest from PIPELINE.json>",
  "findings": []
}
```

An approved independent actor may then run:

```powershell
python tools/analysis/backlog_pipeline.py advance BACKLOG-ADAPT-REQUAL-01 --to VERIFIED --actor Codex --claim-code ARTIFACT_REQUALIFIED
python tools/analysis/backlog_pipeline.py advance BACKLOG-ADAPT-REQUAL-01 --to PROMOTED --actor Codex
```

If a frozen gate fails, use `REJECTED` with the appropriate rejection claim.
If an external prerequisite is absent, use `BLOCKED --reason "exact unblock
condition"`. Never weaken a gate after seeing the result; create a reviewed
successor task when the scientific question genuinely changes.

## CI enforcement

`gate` validates manifest schema, evidence classes, full required-evidence
sets, source paths, packet registration, dependency cycles, legal state
history, file hashes, receipt fingerprint/provenance, frozen gate calculations,
claim codes and reviewer independence. The unit suite also exercises a complete
happy path and confirms that placeholders, malformed receipts, self-review and
proxy promotion fail closed.

The GitHub job runs 139 portable tests. It deselects exactly two local
materialization assertions: the 1.56 GB frozen F16 GGUF and four predecessor
PEFT checkpoints. Their absence is not silently accepted: the preceding
repository-wide `gate` requires an exact committed path/SHA-256/byte receipt.
On the research host, where those inputs remain materialized, the complete
141-test suite still runs and passes.
