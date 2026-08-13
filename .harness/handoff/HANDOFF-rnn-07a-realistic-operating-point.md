# HANDOFF — RNN-07A Realistic Long-Context Operating-Point Discovery (+ RNN-06T2-E1 economics closure)

Two-work-item train. **Nothing pushed.**

## Git / run identity

- **START HEAD:** `ceb657733b6972e14a701384ae7ce8fc4983a10e` (RNN-06T2 final tip)
- **FINAL HEAD:** `716c8de0aab61d492d293a2e6b1e9d91e99e94f0` (branch `master`)
- **Model / revision:** `state-spaces/mamba2-1.3b` @ `c5b59d00ec85d313adea86a08cad2a43c962dd3b`
- **Backend:** official `mamba_ssm` 2.2.4 fast path (chunk_scan_combined prefill + selective_state_update
  decode), bf16, RTX 3090. Fast path proven firing in the scout (`fallbackPathCalls=0`,
  `chunk_scan_combined=4128`, `selective_state_update=1,081,152`).
- **GPU budget:** scout 629 s (~10.5 GPU min) ≤ 90-min scout target; total train ≈ **~58 GPU min** ≤
  3-hour ceiling. `MAX_CONFIDENCE` frozen; no training; no Qwen/DART/StateX/SDM/GDN-2/INT8/ReplaySSM; no
  host-policy change.

## Verdicts

### Work item 1 — RNN-06T2-E1 (economics semantic closure)

| mint | verdict |
|---|---|
| `ECONOMICS_OUTPUT_COMPARABILITY_E1` | **QUALIFIED** |
| `MARGINAL_RECOVERY_UTILITY_ON_STEP_PATH_E1` | **QUALIFIED** |
| `RECOVERY_PATH_VS_FUSED_BASELINE_E1` | **NOT_COMPETITIVE_WITH_FUSED** |
| `GENERAL_END_TO_END_DEPLOYMENT_UTILITY` | **OPEN** |

Fixed the one economics false-green: the recovery arm returned a scored-vocab **column index** while
FINAL_FUSED/FINAL_STEP returned scored **VALUE TOKEN IDs**. Now all arms return the same token-id domain,
proven by executable output-domain assertions on **2 clean process starts** (`vt` ids ≥746 vs column
indices <256 ⇒ the old output is provably out-of-domain; `recovery_old_colidx_all_in_vt=False`,
`recovery_all_in_vt=True`). Timing used **randomized/interleaved** cycles (one shuffled iter per arm per
cycle). Primary marginal comparator `RECOVERY − FINAL_STEP` p95 **+222.6 ms ≤ 250 ms** frozen envelope
(80 pooled warm samples). Descriptive `RECOVERY − FINAL_FUSED` p95 **+1120.6 ms** ⇒ the recovery path is
NOT competitive with a bare fused answer; it only pays off where the workload already needs the
capture-capable step path AND has an exploitable forgetting regime — hence `GENERAL_…UTILITY = OPEN`.
Historical RNN-06T2 economics mint preserved unedited (append-only). Files: `runs/rnn/RNN-06T2-E1/`.

### Work item 2 — RNN-07A (realistic operating-point discovery)

| mint | verdict |
|---|---|
| `REALISTIC_TASK_COMPETENCE` | **INSUFFICIENT** |
| `REALISTIC_FORGETTING_OPERATING_POINT` | **BLOCKED** |
| `REALISTIC_HISTORICAL_RECOVERY_SIGNAL` | **N/A_NO_OPERATING_POINT** (recovery not run — gate honored) |
| `REALISTIC_ADAPTIVE_SELECTION_SIGNAL` | **N/A_NO_OPERATING_POINT** (recovery not run — gate honored) |

Workload: LongBench v2 (natural, 4-way MC), bounded native-length cells, difficulty=easy, priority
domains, target-agnostic question-conditioned BM25 RAG control (2048 tok), deterministic teacher-forced
option-likelihood (content) + enumerated letter-constrained readout (format), B=1.

- `~8K` cell is INFEASIBLE on natural LongBench v2 (0 examples ≤ 8K native; median short-class ≈ 28K
  tokens) — recorded as a boundary condition; NOT prefix-truncated (would confound forgetting with
  evidence-absent). Cells run: `~16K` (n=8, underpowered), `~32K` (n=35, primary).
- **Competence bar (frozen) not met:** best cell (32K) control accuracy **0.257, Wilson LB 0.142** ≈ the
  0.25 four-way-MC chance floor; only **9** eligible (< 20). A 1.3B **base** Mamba-2 is at chance on
  LongBench-v2 easy MC even under RAG.
- **"Full degrades" premise falsified here:** full/FINAL accuracy is **≥** RAG control at both cells (32K
  0.314 ≥ 0.257; 16K 0.250 ≥ 0.125); `control − full` negative, CI spans 0. Not a format artifact
  (format adherence higher on full). No competence to lose ⇒ no natural forgetting operating point.
- **Gate honored:** operating point ≠ FOUND ⇒ recovery eval NOT run (`RECOVERY_RESULTS.json` =
  `NOT_RUN`); recovery/adaptive signals N/A. No goalpost moved, no seed/config screening, NoLiMa bridge
  NOT invoked (its precondition — dataset unusable — was not met).

Files: `runs/rnn/RNN-07A/` (prereg, SCOUT_16K/32K/SUMMARY, RECOVERY_RESULTS, decision, scout log).

## Scientific reading

The RNN-06T2 recovery/adaptive-selection results remain **synthetic-construction-specific** (anti-oracle
MQAR). No natural analogue could be observed here because *this* 1.3B base checkpoint lacks the
LongBench-v2 task competence required to even exhibit a forgetting→recovery operating point within
budget. This is an honest negative discovery, not evidence that recurrent LMs don't forget naturally.

## Executed source (paths / functions)

- `ops/rnn_06t2_e1_econ.py::output_domain_assertions` — DOMAIN_MEMBERSHIP / DTYPE_RANGE / NOT_COLUMN_INDEX
  hard asserts; `recovery_equiv(..., return_colidx=True)` returns both token-id and column-index for proof.
- `ops/rnn_06t2_e1_econ.py::main` — interleaved shuffled timing cycles; `ops/rnn_06t2_e1_decide.py` — 4
  separate mints.
- `ops/rnn_07a_lib.py` — `bm25_rag_control` (target-agnostic), `prefill_state` (num_last_tokens=1, no
  OOM), `readout_from_state` (teacher-forced option-likelihood), `letter_readout` (enumerated format).
- `ops/rnn_07a_scout.py::run_cell` — control vs full; competence + degradation mints.
- `ops/rnn_07a_recovery.py::main` — conditional on FOUND; wrote NOT_RUN stub (gate honored).

## Committed diffs (append-only; nothing pushed)

`8ab7b73` E1 prereg+tools · `6a6c887` 07A prereg+tools · `88a7daa` 07A recovery runner ·
`49bea62` E1 results+decision · `716c8de` 07A results+decision.

## Authority / effect status

Record + discovery. E1 corrects the RNN-06T2 economics output-domain semantics (marginal recovery
utility on the step path remains QUALIFIED; deployment utility OPEN). RNN-07A finds no realistic
operating point for this checkpoint. No production/deploy effect. Nothing pushed.

## Exactly one next recommendation (NOT executed)

Re-scope a future **RNN-07B** (fresh preregistered set) to a subject/task where the base model clears the
competence bar — either a larger official recurrent LM with real LongBench-v2 MC competence, or a
realistic-but-easier natural long-context task (e.g. long-document extractive QA with exact-match
scoring) — keeping the same qualified lifecycle + frozen MAX_CONFIDENCE machinery. Opens only after
independent audit accepts RNN-07A.
