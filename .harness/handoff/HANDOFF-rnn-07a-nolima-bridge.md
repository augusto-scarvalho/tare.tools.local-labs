# HANDOFF — RNN-07A-BRIDGE (NoLiMa Semi-Synthetic Controlled Bridge, corrective)

Corrective completion of RNN-07A: the NoLiMa controlled-bridge fallback condition was met (parent natural
workload `OPERATING_POINT = BLOCKED`) but not executed. **Nothing pushed.**

## Git / run identity

- **START HEAD:** `57daef43fe0c201773d2079c6c644c5656a536b9` (RNN-07A parent delivery tip)
- **FINAL HEAD:** `aaec5f2` (branch `master`; delivery/bundle commits follow)
- **Model / revision:** `state-spaces/mamba2-1.3b` @ `c5b59d00ec85d313adea86a08cad2a43c962dd3b`, official
  `mamba_ssm` 2.2.4 fast path (fast path proven firing; `fallbackPathCalls=0`).
- **Bridge workload:** NoLiMa `amodaresi/NoLiMa` @ `378115b1…` — **SEMI_SYNTHETIC_CONTROLLED_BRIDGE**,
  never a natural-workload qualification. Provenance (source/revision/SHA-256) in
  `EXTERNAL_WORKLOAD_PROVENANCE.json`; datasets excluded from the bundle.
- **Budget:** total train GPU ≈ **~71 GPU min ≤ 3-hour ceiling** (parent ~58 + bridge short ~1.2 + bridge
  long/recovery ~11.5). MAX_CONFIDENCE frozen; no training/tournament; no Qwen/DART/StateX/SDM/GDN-2/INT8/
  ReplaySSM; no host-policy change.

## Reconciliation carried in (append-only, unchanged parent results)

`runs/rnn/RNN-07A/AUDIT_RECONCILIATION.md`: LongBench-v2 results and `REALISTIC_TASK_COMPETENCE =
INSUFFICIENT` / `OPERATING_POINT = BLOCKED` are preserved and NOT retroactively reinterpreted. The defect
was: BLOCKED triggers the NoLiMa bridge, which the parent minted-around instead of executing → RNN-07A was
incomplete. This corrective executes the bridge.

## Bridge verdicts

| mint | verdict |
|---|---|
| `BRIDGE_SHORT_CONTEXT_COMPETENCE` | **SUFFICIENT** (short 512-tok acc 0.866, Wilson LB 0.791 > 0.50; 97 eligible) |
| `BRIDGE_LONG_CONTEXT_DEGRADATION` | **QUALIFIED** (8K/16K 0.256, 32K 0.222; degradation 0.744–0.778, CI_lo > 0.65; 67–70 forgotten; recovery cell 32K) |
| `BRIDGE_HISTORICAL_RECOVERY_SIGNAL` | **NO_SIGNAL** (all snapshots ≤ chance and worse than FINAL; deltas negative) |
| `BRIDGE_ADAPTIVE_SELECTION_SIGNAL` | **NO_SIGNAL** (MAX_CONFIDENCE 0.208 < FINAL 0.333; delta −0.125) |

## Headline scientific result

On a semi-natural NoLiMa needle workload the 1.3B base Mamba-2 is **competent at short context and
genuinely forgets at long context** — a real forgetting operating point the natural LongBench-v2 workload
could not even reach (it lacked competence). But the frozen historical-state + `MAX_CONFIDENCE` recovery
machinery gives **no recovery benefit and is mildly harmful**. Two mechanisms:
1. **Forgetting outpaces snapshot granularity** — needle at 15% depth is forgotten before the earliest
   positional snapshot (SNAP_25 is already ~3.2K tokens past it at 32K), so no 25/50/75/90% snapshot
   retains it.
2. **Confidence is not calibrated to correctness** on natural needles — `ORACLE_BEST_GOLD = 0.604` shows
   a retaining snapshot often exists, but `MAX_CONFIDENCE` (0.208) can't identify it (it tracks context
   length, not needle retention; selector picks FINAL only 3/48).

⇒ RNN-06T2's synthetic-MQAR recovery/adaptive utility is **construction-specific** and does **not**
transport to this bridge.

## Executed source (paths)

`ops/rnn_07a_bridge_lib.py` (needle/haystack construction, stable seeds — no `hash()`), 
`ops/rnn_07a_bridge_short.py` (competence gate), `ops/rnn_07a_bridge_long.py` (degradation + gated
recovery), `ops/rnn_07a_bridge_bundle.py`. Reuses parent `ops/rnn_07a_lib.py` frozen readout +
`ops/rnn_06t_lib.py` state machinery.

## Committed diffs (append-only; nothing pushed)

`bc82a71` bridge prereg/provenance/audit/tools · `c5bab43` short competence · `0baa971` long+recovery ·
`aaec5f2` bridge decision + git evidence.

## Authority / effect status

Record + controlled-bridge discovery. Confirms forgetting on a semi-natural workload; shows the frozen
recovery machinery does not generalize there. Does not change the natural-workload negative. No
production/deploy effect. Nothing pushed.

## Exactly one next recommendation (NOT executed)

Freshly preregister a test of whether **finer/adaptive snapshot spacing** (matched to the natural
forgetting timescale) plus a **calibrated retention signal** (not raw option-likelihood confidence) can
recover natural NoLiMa needles — no base-model training, fresh set. Opens only after audit accepts
RNN-07A-BRIDGE.
