# RNN-05B-EXT — PRE-REGISTRATION (written before any outcome-bearing run)

Direct **H3** test: can a HISTORICAL DN/GDN recurrent-state snapshot recover an association the stable frozen
backbone has LOST from its final state, beyond what the final state alone yields? LA is the mechanistic control.
This file and `preregistration.json` are generated from ONE frozen config object (`ExtConfig`); the executed
constants, the JSON, and the human-readable selection rule are identity-checked
(`CALIBRATION_RULE_IDENTITY = PASS`).

## Hypotheses
- **H3**: historical DN/GDN states contain recoverable task information no longer sufficiently represented in
  the final recurrent state.
- **NULL**: historical snapshots add nothing recoverable beyond the final state under a stable regime.
- **LA control**: additive/collapsible final state -> should not benefit from redundant historical snapshots.

## Architecture (RNN-05B-qualified family, UNCHANGED)
`MQARDeltaModel` d_model=128, d_k=64, d_v=64, conv_k=4; MC/chunk segment
seg=64. No recurrence-equation edits, no deeper readers, no GDN-mechanism edits. Reader = the existing
`w_u` grm connector only.

## Memory-bound challenge (temporal pressure, NOT capacity overload)
- num_pairs = **12** (far below the RNN-05B capacity cliff ~40 at d_k=64 — no capacity overload).
- num_queries = 8; num_keys = 128; num_vals = 64.
- Writes spread across the EARLY 25% of the body; a long retention gap with distractor
  interference follows; queries are at the very end. Recall is hard because old associations decay / are
  interfered with, NOT because the recurrence is globally untrainable.

## Predeclared candidate grid (cheap-first; frozen numeric generator params)
seq_len axis = [512, 768, 1024] (retention gap); distractor axis =
[('low', 0.15), ('med', 0.35), ('high', 0.55)] (interference). 9 conditions, order:
mb_L512_low, mb_L512_med, mb_L512_high, mb_L768_low, mb_L768_med, mb_L768_high, mb_L1024_low, mb_L1024_med, mb_L1024_high.

## Headroom / calibration rule (single source of truth)
> iterate the predeclared grid cheap-first (seq_len asc [512, 768, 1024], then distractor asc ['low', 'med', 'high']); SELECT the first condition whose GDN seed-42 BASE holdout accuracy is within (0.4,0.8); QUALIFY it iff all 3 GDN seeds are within (0.2,0.9) and their mean is within (0.4,0.8); otherwise H3_TESTABILITY=BLOCKED with NO nearest-condition fallback

Required GDN BASE band **[0.4, 0.8]** (operator-design TESTABILITY window, not a scientific
noise floor); per-seed stability band [0.2, 0.9]. Calibration observes **BASE only** — no MC
or reader result may influence difficulty selection. If no condition qualifies: `H3_TESTABILITY = BLOCKED`
(`BLOCKED_BY_CEILING` if every condition > 0.8; `BLOCKED_BY_UNSTABLE_BASE` if the base collapses),
and STOP with **no nearest-condition fallback**.

## Training identity (predeclared seeds)
BASE backbones trained single-state at steps=2500, lr=0.003, batch=96,
pool_train=4096. Load-bearing: GDN seeds [42, 43, 44], DN seeds [42, 43, 44];
control: LA seeds [42]. Frozen reader trained 1800 steps (w_u only; backbone mutation
must be 0). Disjoint CALIBRATION / DEVELOPMENT / FINAL-HOLDOUT example ranges; pinned id hashes.

## Load-bearing comparison (frozen backbone)
A = BASE single final state · B = param-free historical snapshots (moving average) · C = trained `w_u` reader.
Primary quantity = **RECOVERY_RATE** (fraction of BASE-wrong target queries MC flips to correct) with
**HARM_RATE** (BASE-correct flipped to wrong) reported alongside (a net gain can hide large recovery+harm).
Then: snapshot attribution (DESCRIPTIVE), a small snapshot ablation, a fixed-position pure cache-count curve
(K=[1, 2, 4, 8]), the LA falsification control, DN vs GDN reported separately, and a small secondary 2x2.

## Effects
Raw paired deltas across the 3 training seeds; direction agreement; MARGIN=0.03
retained only as an OPERATOR_HEURISTIC (not a measured noise floor). No p-values manufactured from n=3.

## Decision
A strong positive H3 requires: `H3_TESTABILITY=QUALIFIED`; GDN positive net holdout delta; direction positive
across all 3 seeds; recovery exceeds harm by a meaningful margin; ablation supports the
snapshot mechanism; and LA does NOT show the same pattern. If BASE has headroom and MC still adds nothing ->
`H3 = NOT_DETECTED_IN_QUALIFIED_REGIME` (a much stronger negative than RNN-05B). Qwen: no weights used; gate
= `PASS_CANDIDATE` only on a defensible positive mechanism (authorizes DESIGN of a separate Qwen packet only),
else `DEFER`.

## Guardrails
No Qwen weights · no llama.cpp/serving/deploy · no TPTT · no RNN-05C · no FLA install · not pushed. Budget
target < 1 GPU-hr (hard 2). RNN-05B raw evidence immutable.
