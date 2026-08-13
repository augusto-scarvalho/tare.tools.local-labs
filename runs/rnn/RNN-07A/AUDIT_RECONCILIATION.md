# RNN-07A — AUDIT RECONCILIATION (append-only)

This file is APPEND-ONLY. It does not edit any historical RNN-07A artifact. All LongBench-v2 results,
scout numbers, and the `REALISTIC_TASK_COMPETENCE = INSUFFICIENT` verdict are **preserved unchanged and
are NOT retroactively reinterpreted**.

## Reconciled finding (2026-08-13)

The parent RNN-07A decision closed the train after minting
`REALISTIC_FORGETTING_OPERATING_POINT = BLOCKED` (natural workload, LongBench v2), and declined to invoke
the NoLiMa controlled bridge on the ground that "the dataset was usable — it's the model that's at
chance." Independent audit reconciles this as a **protocol defect**:

- The prereg's `CONTROLLED BRIDGE FALLBACK (NoLiMa)` condition is met not only when the dataset is
  *unusable*, but whenever the natural workload **fails to establish a competence/forgetting operating
  point** — i.e. exactly the `REALISTIC_TASK_COMPETENCE = INSUFFICIENT → OPERATING_POINT = BLOCKED`
  outcome that was observed. The fallback condition **WAS met**.
- Therefore RNN-07A is **NOT COMPLETE**: the parent minted BLOCKED but did not execute the controlled
  bridge that the BLOCKED outcome triggers.

## Reconciled verdicts (carried into the corrective packet)

| item | reconciled state |
|---|---|
| LongBench-v2 scout results (`SCOUT_16K/32K/SUMMARY.json`) | UNCHANGED (authoritative) |
| `REALISTIC_TASK_COMPETENCE` (natural / LongBench v2) | **INSUFFICIENT** — NOT reinterpreted |
| `REALISTIC_FORGETTING_OPERATING_POINT` (natural) | **BLOCKED** — unchanged |
| `REALISTIC_HISTORICAL_RECOVERY_SIGNAL` / `..._ADAPTIVE_...` (natural) | N/A — unchanged |
| NoLiMa bridge condition | **MET but NOT executed by parent** → corrective required |
| NoLiMa status label | **SEMI_SYNTHETIC_CONTROLLED_BRIDGE** — never a natural-workload qualification |

## Corrective scope (RNN-07A-BRIDGE)

1. Repair external workload provenance BEFORE any new outcome (`EXTERNAL_WORKLOAD_PROVENANCE.json`:
   source repo, revision commit, per-file SHA-256, for both LongBench v2 and NoLiMa).
2. Keep the exact qualified subject/backend: `state-spaces/mamba2-1.3b` @ `c5b59d00…`, official
   `mamba_ssm` fast path; `MAX_CONFIDENCE` frozen; no training/tournament.
3. Gated execution on the semi-synthetic controlled bridge (real NoLiMa book haystack + planted needle):
   - **Establish SHORT-context NoLiMa competence first** → `BRIDGE_SHORT_CONTEXT_COMPETENCE`.
   - **Only if SHORT competence QUALIFIES**, test bounded long-context degradation →
     `BRIDGE_LONG_CONTEXT_DEGRADATION`.
   - **Only if degradation QUALIFIES**, run the frozen historical-state / MAX_CONFIDENCE recovery
     machinery → `BRIDGE_HISTORICAL_RECOVERY_SIGNAL`, `BRIDGE_ADAPTIVE_SELECTION_SIGNAL`.
4. NoLiMa results are labelled `SEMI_SYNTHETIC_CONTROLLED_BRIDGE` and never upgrade or overwrite the
   natural-workload negative. An optional LongBench 64K descriptive cell is permitted but must not
   rewrite the existing negative result.
5. Remain inside the original 3-hour train ceiling (parent used ~58 GPU min). Nothing pushed.
