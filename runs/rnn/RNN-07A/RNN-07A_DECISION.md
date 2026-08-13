# RNN-07A — REALISTIC LONG-CONTEXT OPERATING-POINT DISCOVERY — DECISION

Discovery run on the qualified official subject `state-spaces/mamba2-1.3b` @ `c5b59d00…` (official
`mamba_ssm` 2.2.4 fast path; `fallbackPathCalls=0`, `chunk_scan_combined=4128`,
`selective_state_update=1,081,152` — fast path proven firing). MAX_CONFIDENCE frozen; no training. This
is DISCOVERY, not confirmation. Prereg: `RNN-07A_PRE_REGISTRATION.md` (frozen before outcomes). Nothing
pushed.

## Mints

| mint | verdict |
|---|---|
| `REALISTIC_TASK_COMPETENCE` | **INSUFFICIENT** |
| `REALISTIC_FORGETTING_OPERATING_POINT` | **BLOCKED** |
| `REALISTIC_HISTORICAL_RECOVERY_SIGNAL` | **N/A_NO_OPERATING_POINT** (recovery not run — gate honored) |
| `REALISTIC_ADAPTIVE_SELECTION_SIGNAL` | **N/A_NO_OPERATING_POINT** (recovery not run — gate honored) |

## What was run (scout, ~10.5 GPU min of the 90-min budget)

Bounded candidate pool: LongBench v2 (`THUDM/LongBench-v2`, 4-way MC), difficulty=easy, priority domains
(Single-Doc QA, Long-dialogue History, Multi-Doc QA, Long Structured Data), native token length ≤ cell
budget. Per example: target-agnostic question-conditioned BM25 RAG control (2048 tok) vs the native
full context; deterministic teacher-forced length-normalized option-likelihood (PRIMARY content readout)
+ enumerated letter-constrained readout (SECONDARY format probe). B=1 (no context padding).

The `~8K` cell is INFEASIBLE on natural LongBench v2 context (0 examples native ≤ 8K; median short-class
context ≈ 28K tokens) and was NOT prefix-truncated (truncation would remove evidence and confound
"forgetting" with "evidence absent"). Cells run: `~16K` (n=8, underpowered) and `~32K` (n=35, primary).

| cell | n | control acc (Wilson LB) | full/FINAL acc (Wilson LB) | control−full | n_eligible | n_forgotten |
|---|---|---|---|---|---|---|
| 16K | 8 | 0.125 (0.022) | 0.250 (0.071) | −0.125 | 1 | 0 |
| 32K | 35 | 0.257 (0.142) | 0.314 (0.186) | −0.057 | 9 | 4 |

## Why INSUFFICIENT / BLOCKED (honest reading)

- **Competence bar not met.** The frozen bar for `REALISTIC_TASK_COMPETENCE = SUFFICIENT` was control
  Wilson-95% LB `> 0.35` and `≥ 20` eligible. The best cell (32K) gives control acc 0.257 with LB
  **0.142** — statistically indistinguishable from the 0.25 four-way-MC chance floor — and only **9**
  eligible. A 1.3B **base** pretrained Mamba-2 is at chance on LongBench v2 easy MC via option-likelihood,
  even when question-relevant text is concentrated by RAG. There is no genuine task competence to lose.
- **The "full context degrades" premise is falsified here.** At both cells full/FINAL accuracy is
  **≥** the RAG-control accuracy (32K: 0.314 vs 0.257; 16K: 0.250 vs 0.125). The model is not forgetting
  — both conditions sit at chance. `degradation = control − full` is **negative** with a CI spanning 0.
- **Not a format artifact.** Format adherence did not drop on full context (32K drop = −0.222, i.e.
  adherence higher on full); the PRIMARY metric is option-likelihood, which needs no letter emission, so
  any degradation would be content by construction. Moot given no competence.
- **Gate honored.** Because `REALISTIC_FORGETTING_OPERATING_POINT ≠ FOUND`, the recovery evaluation was
  NOT run; `REALISTIC_HISTORICAL_RECOVERY_SIGNAL` and `REALISTIC_ADAPTIVE_SELECTION_SIGNAL` are N/A
  (`RECOVERY_RESULTS.json` status `NOT_RUN`).

## What this does and does not say

- It does **not** show recurrent LMs cannot forget on realistic workloads. It shows that *this* 1.3B base
  checkpoint lacks the underlying LongBench-v2 competence needed to *observe* a natural forgetting→recovery
  operating point within budget. The RNN-06T2 recovery/adaptive results therefore remain
  **synthetic-construction-specific** (anti-oracle MQAR); no natural analogue is established here.
- No goalpost was moved: the competence bar, cells, pool, and thresholds were frozen before outcomes and
  are reported as-is. No seed/config screening. The NoLiMa controlled bridge was **not** invoked — its
  precondition (LongBench v2 unusable) was not met; the dataset ran fine, so relabeling a controlled
  bridge as a natural result would be improper.

## Budget

Scout 629 s (~10.5 GPU min) ≤ 90-min scout target. Total GPU for the two-item train (RNN-06T2-E1 ≈ 47
min + scout ≈ 10.5 min + recovery-gate no-GPU) ≈ **~58 GPU min ≤ 3-hour ceiling**.

## Recommended next step (NOT executed; for RNN-07B or a re-scoped RNN-07)

A realistic operating-point discovery needs a base checkpoint with **demonstrable** LongBench-v2-class
competence (control accuracy robustly above chance). Options for a future, freshly preregistered run:
(a) a larger official recurrent LM with real MC competence, or (b) a realistic-but-easier natural task
where a 1.3B base model clears the competence bar (e.g. long-document extractive QA with exact-match
scoring), keeping the same lifecycle/MAX_CONFIDENCE machinery. Deferred; opens only after audit accepts
RNN-07A.
