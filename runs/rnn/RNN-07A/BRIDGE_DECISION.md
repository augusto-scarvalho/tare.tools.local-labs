# RNN-07A-BRIDGE — NoLiMa SEMI-SYNTHETIC CONTROLLED BRIDGE — DECISION

Corrective execution of the NoLiMa controlled bridge whose fallback condition was met (parent natural
workload `OPERATING_POINT = BLOCKED`) but not executed. Subject unchanged: `state-spaces/mamba2-1.3b` @
`c5b59d00…`, official `mamba_ssm` fast path (`fallbackPathCalls=0`; fast path proven firing). MAX_CONFIDENCE
frozen; no training/tournament. **NoLiMa is `SEMI_SYNTHETIC_CONTROLLED_BRIDGE` — never a natural-workload
qualification.** All LongBench-v2 results and `REALISTIC_TASK_COMPETENCE = INSUFFICIENT` are unchanged.
Prereg: `BRIDGE_PRE_REGISTRATION.md`. Provenance: `EXTERNAL_WORKLOAD_PROVENANCE.json`. Nothing pushed.

## Mints

| mint | verdict |
|---|---|
| `BRIDGE_SHORT_CONTEXT_COMPETENCE` | **SUFFICIENT** |
| `BRIDGE_LONG_CONTEXT_DEGRADATION` | **QUALIFIED** |
| `BRIDGE_HISTORICAL_RECOVERY_SIGNAL` | **NO_SIGNAL** |
| `BRIDGE_ADAPTIVE_SELECTION_SIGNAL` | **NO_SIGNAL** |

## Construction (controlled bridge)

NoLiMa `ONLYDirect` needles ("Actually, {CHAR} lives next to {1}.") with the DIRECT literal question
("Which character lives next to {1}?"), scored as 4-way option-likelihood over character names (gold +3
seeded distractors from the needle's character_set) — the same frozen readout as the parent. Real book
text (`rand_shuffle/rand_book_1.txt`, natural prose) is the haystack; the needle is planted at a fixed
a-priori depth `0.15`. Pool = 112 examples (28 needle×test combos × 4 seeded char assignments). B=1.

## Gate 1 — SHORT competence (SUFFICIENT)

n=112, SHORT (512-token) MC accuracy **0.866** (Wilson LB **0.791** > 0.50 bar), 97 eligible ≥ 20 →
`BRIDGE_SHORT_CONTEXT_COMPETENCE = SUFFICIENT`. The 1.3B base model reliably retrieves a directly-stated
needle at short context (chance 0.25). This is the competence the natural LongBench-v2 workload lacked.

## Gate 2 — LONG degradation (QUALIFIED)

On the 90 short-competence-eligible examples (eligibility = short-correct, independent of the long
outcome), FINAL accuracy collapses to ~chance at every bounded length:

| cell | FINAL acc | degradation (short−long) | CI_lo | forgotten | qualifies |
|---|---|---|---|---|---|
| 8K | 0.256 | 0.744 | 0.656 | 67 | yes |
| 16K | 0.256 | 0.744 | 0.656 | 67 | yes |
| 32K | 0.222 | 0.778 | 0.689 | 70 | yes |

All cells qualify (degradation ≥ 0.15, CI_lo > 0, forgotten ≥ 15). Recovery cell = **32K** (max
degradation 0.778). This is genuine forgetting of a natural-language needle — a real forgetting operating
point that the natural workload could not even reach (it lacked competence).

## Gate 3 — RECOVERY on 32K (NO_SIGNAL / NO_SIGNAL)

Recovery-eligible n=48, forgotten=32. Arm accuracy (frozen readout from each state):

| arm | acc |
|---|---|
| FINAL | 0.333 |
| SNAP_25 | 0.229 |
| SNAP_50 | 0.125 |
| SNAP_75 | 0.250 |
| SNAP_90 | 0.188 |
| MAX_CONFIDENCE (frozen) | **0.208** |
| ORACLE_BEST_GOLD (diagnostic) | 0.604 |

- **`BRIDGE_HISTORICAL_RECOVERY_SIGNAL = NO_SIGNAL`.** No fixed earlier snapshot recovers the forgotten
  population: per-snapshot recovery rates over the forgotten set are 0.13–0.22 and every snapshot's
  accuracy delta vs FINAL is **negative** (SNAP_25 −0.104, SNAP_50 −0.208, SNAP_75 −0.083, SNAP_90 −0.146;
  CIs include or sit below 0). Snapshots are no better than FINAL — often worse.
- **`BRIDGE_ADAPTIVE_SELECTION_SIGNAL = NO_SIGNAL`.** Frozen `MAX_CONFIDENCE` (0.208) is **worse** than
  simply using FINAL (0.333): delta **−0.125**, CI [−0.292, 0.042]. The selector histogram
  `{SNAP_25:12, SNAP_50:19, SNAP_75:8, SNAP_90:6, FINAL:3}` shows it usually prefers a short earlier
  snapshot (higher option-likelihood confidence is a length artifact), but those are less accurate.

## Why recovery fails here (mechanism) — the key finding

The synthetic-MQAR recovery/adaptive-selection result from RNN-06T2 **does NOT transport** to this
semi-natural NoLiMa workload, for two concrete reasons:

1. **Forgetting outpaces snapshot granularity.** The needle sits at 15% depth; the earliest positional
   snapshot (SNAP_25) is already ~3.2K tokens past it at 32K. The needle is forgotten *before* any
   captured snapshot, so no 25/50/75/90% snapshot retains it. In the synthetic MQAR the target was placed
   so a snapshot could straddle it; natural fast forgetting defeats coarse positional snapshots.
2. **Confidence is not calibrated to correctness on natural needles.** `ORACLE_BEST_GOLD = 0.604` proves
   *some* snapshot often holds the answer, but `MAX_CONFIDENCE` (0.208) cannot find it — model confidence
   is anti-correlated with correctness (it tracks context length, not needle retention). The frozen
   selector that won on constructed MQAR (calibrated confidence) is actively harmful here.

## Scope / non-claims

Conditional on this frozen checkpoint and this controlled construction. Semi-synthetic controlled bridge
— NOT a natural-workload qualification and it does not upgrade or overwrite the natural-workload negative
(`REALISTIC_TASK_COMPETENCE = INSUFFICIENT`, `OPERATING_POINT = BLOCKED`, unchanged). No goalpost moved,
no seed screening, no selector tournament/training. The optional LongBench 64K descriptive cell was
deliberately NOT run (it is optional, cannot change the natural negative, and the bridge already answered
the recovery question); budget was preserved. Total train GPU ≈ **~71 GPU min ≤ 3-hour ceiling**.

## Headline

On a semi-natural NoLiMa needle workload the model is **competent (short) and genuinely forgets (long)**,
but the frozen historical-state + `MAX_CONFIDENCE` recovery machinery provides **no recovery benefit
(and is mildly harmful)** — because natural forgetting is faster than the positional snapshot grid and
model confidence does not identify the retaining snapshot. RNN-06T2's recovery utility remains
**construction-specific**; it does not generalize to this bridge.

## Recommended next step (NOT executed)

If recovery is to be pursued as a real capability, a future preregistered run should test whether
**finer / adaptive snapshot spacing** (denser near the natural forgetting timescale) plus a **calibrated
retention signal** (not raw option-likelihood confidence) can recover natural needles — still with no
training of the base model, on a fresh set. Deferred; opens only after audit accepts RNN-07A-BRIDGE.
