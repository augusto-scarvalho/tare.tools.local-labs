# RNN-06T2-E1 — ECONOMICS SEMANTIC CLOSURE — DECISION

Append-only remediation of one semantic false-green in the RNN-06T2-T1R economics arm. The historical
economics mint (`END_TO_END_RECOVERY_UTILITY_T1R = QUALIFIED`) is **preserved unedited** in
`runs/rnn/RNN-06T2/`. E1 wrote only into `runs/rnn/RNN-06T2-E1/` and added
`ops/rnn_06t2_e1_econ.py` / `ops/rnn_06t2_e1_decide.py`. Recovery/lifecycle qualification were NOT rerun.
Nothing pushed.

## The false-green, and the fix

- **Before (historical `ops/rnn_06t2_econ.py`):** `FINAL_FUSED` and `FINAL_STEP` returned scored **VALUE
  TOKEN IDs**, but `RECOVERY` returned `pl.argmax(-1)` — a **column index into the scored-vocab tensor
  `vt`** (range `[0, 256)`), not a token id. The three timed arms did not provably compute the same
  answer object.
- **After (E1):** the recovery arm maps its selected column index back through `vt`, returning the same
  scored VALUE TOKEN ID domain as the other two arms. This is the ONLY semantic change; timings are
  unaffected (a single extra gather).

## Executable output-domain assertions (ran BEFORE timing, both process starts)

`vt_size=256`, `vt_min=746`, `vt_max=50220`. Because every scored value token id ≥ 746 while column
indices lie in `[0, 256)`, a column-index output can never be in the `vt` set — so the OLD output is
provably out-of-domain and the fix is provably necessary.

| assertion | process 0 | process 1 |
|---|---|---|
| `ASSERTIONS_PASSED` | True | True |
| `final_fused_all_in_vt` (DOMAIN_MEMBERSHIP) | True | True |
| `final_step_all_in_vt` | True | True |
| `recovery_all_in_vt` (fix works) | True | True |
| `recovery_old_colidx_all_in_vt` (must be False = NOT_COLUMN_INDEX proof) | False | False |
| `recovery_fix_changed_output` | True | True |

## Timing (randomized/interleaved cycles, 2 clean process starts, 80 pooled warm samples/arm)

Each of the 40 warm cycles per process timed all three arms once in a **shuffled** order (seed 20261200
/ 20261201), removing fixed-slot drift bias. Warm per-query (batch-amortized):

| arm | median | p95 |
|---|---|---|
| FINAL_FUSED_EQUIVALENT_WORK | 37.79 ms | 39.85 ms |
| FINAL_STEP_EQUIVALENT_WORK | 991.30 ms | 1113.07 ms |
| RECOVERY_ENABLED_EQUIVALENT_WORK | 1014.63 ms | 1159.14 ms |

- Primary marginal comparator `RECOVERY − FINAL_STEP`: median **+37.0 ms**, **p95 +222.6 ms ≤ 250 ms**
  frozen envelope (per-process p95: 191.2 ms, 222.6 ms). Recovery machinery ≈ 21 ms/q (restore+readout
  ~12, GPU→CPU copy ~9, selection ~0.01); the shared ~897 ms trajectory+capture is the step path run
  regardless.
- Descriptive `RECOVERY − FINAL_FUSED`: median **+975 ms**, p95 **+1120.6 ms** — the orthogonal
  step-vs-fused path cost, NOT the recovery premium.

## Mints (kept separate)

| mint | verdict |
|---|---|
| `ECONOMICS_OUTPUT_COMPARABILITY_E1` | **QUALIFIED** (all output-domain assertions pass on both starts; all arms return the same scored VALUE TOKEN ID domain) |
| `MARGINAL_RECOVERY_UTILITY_ON_STEP_PATH_E1` | **QUALIFIED** (comparable outputs + p95 222.6 ms ≤ 250 ms + cited frozen wide quality gate Δ0.542 / net 104 > 0) |
| `RECOVERY_PATH_VS_FUSED_BASELINE_E1` | **NOT_COMPETITIVE_WITH_FUSED** (p95 `RECOVERY − FINAL_FUSED` = 1120.6 ms ≫ envelope; descriptive — recovery needs the capture-capable step path, which is far costlier than a bare fused answer) |
| `GENERAL_END_TO_END_DEPLOYMENT_UTILITY` | **OPEN** (asserted unconditionally; whether recovery pays off in a real deployment depends on a realistic forgetting regime — RNN-07A) |

## Interpretation

The recovery premium is small **on the step path you already pay for** (marginal ≤ 250 ms), but the step
path itself is ~26× the cost of a bare fused answer. So recovery is only economically interesting where
the workload *already* requires the capture-capable step path AND exhibits a forgetting regime that
recovery can exploit. Establishing whether such a realistic regime exists is exactly
`GENERAL_END_TO_END_DEPLOYMENT_UTILITY = OPEN`, addressed (as discovery) by RNN-07A.

## Constraints honored

No lifecycle or synthetic-recovery-qualification rerun. No threshold redefinition (envelope 250 ms was
frozen in `runs/rnn/RNN-06T2/T1R_PRE_REGISTRATION.md` before T1R outcomes). No Qwen, no selector/reader
training, no host-policy change. Append-only; nothing pushed.
