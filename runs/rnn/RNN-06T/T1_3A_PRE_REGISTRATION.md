# RNN-06T — 3A PRE-REGISTRATION (exact-contract transportability replication)

Frozen BEFORE 3A outcomes. Executes because T0 both QUALIFIED. Purpose: isolate
checkpoint/backend/capture transport from any challenge redesign — keep the RNN-06D challenge
semantics; change only subject (official mamba_ssm) and capture (T0-qualified single-pass).

## Frozen configuration

Official `state-spaces/mamba2-1.3b` @ `c5b59d00`, mamba_ssm 2.2.4 fast path. Construction = RNN-06D v2
anti-oracle: M=192, target band [8,64], sentinel-pre / unique-DS-load-post, K=4 schedule
[38,76,115,153]; fixed 770-token sequence; constrained argmax over 256 scored values, chance 1/256.
**Fresh disjoint** qualification set `qualificationSetSha256_3A = 5e47408e723e10473c1072fd436699df477a2cbfe9f69bd9cf02e01e1abe09e7` (seed 20260970; example-level
overlap 0 with p0/06b/b2/b3/c06/d0_calib/d0_qual). N=192 (3 strata × 64). Capture via the
T0-qualified single-pass trajectory (in-run states at token boundaries [156,308,464,616]+FINAL 768),
**fixed batch size 16** throughout (T0: neighbor isolation bit-exact; batch-size numeric sensitivity
handled by fixing batch).

## Frozen selector

**MAX_CONFIDENCE** = select the pool snapshot with the highest constrained answer-space top-1
probability (softmax over the 256 scored-value logits, max), identical to RNN-06D. **Not retuned.**

## Arms

FINAL (same-run final state); ORACLE_BEST_GOLD, ORACLE_TARGET_PROXIMAL (diagnostic only);
**FIXED_SLOT_76** (mandatory non-adaptive control); MAX_CONFIDENCE (frozen adaptive primary);
RECENCY/FIXED_SLOT_153 (descriptive); MATCHED_NO_HISTORY (compute control). No seven-method selector
competition.

## Two distinct claims, separate SESOIs (frozen)

- **CLAIM 1 — historical recovery transport.** `FIXED_SLOT_76 − FINAL` and `MAX_CONFIDENCE − FINAL`.
  SESOI_RECOVERY = 0.15; require both ≥ 0.15, stratified-bootstrap 95% CI lower bound > 0.05, robust
  (Δ ≥ 0) in ≥ 2/3 strata.
- **CLAIM 2 — adaptive selector incremental value.** `MAX_CONFIDENCE − FIXED_SLOT_76`.
  SESOI_ADAPTIVE = 0.05; QUALIFIED iff Δ ≥ 0.05, paired CI lower bound > 0, robust ≥ 2/3;
  DIRECTIONAL iff Δ > 0 but below the bar; NOT_QUALIFIED iff Δ ≤ 0. (This is the contrast the RNN-06D
  audit flagged as NOT_QUALIFIED — its point estimate there was only +0.0625.)

The claims are NOT collapsed into one number.

## Gate

- `HISTORICAL_RECOVERY_TRANSPORT ∈ {QUALIFIED, NOT_REPLICATED}` from CLAIM 1.
- `ADAPTIVE_SELECTOR_ADVANTAGE ∈ {QUALIFIED, DIRECTIONAL, NOT_QUALIFIED}` from CLAIM 2.
- `OFFICIAL_MAMBA_SYNTHETIC_TRANSPORT`: QUALIFIED if recovery QUALIFIED **and** adaptive QUALIFIED;
  PARTIAL if recovery QUALIFIED but adaptive DIRECTIONAL/NOT_QUALIFIED; NOT_REPLICATED if recovery not
  QUALIFIED. It is explicitly valid for recovery to replicate while adaptive advantage remains
  unqualified.

3B executes only if HISTORICAL_RECOVERY_TRANSPORT = QUALIFIED. Also report whether MAX_CONFIDENCE's
06D result (~0.833) replicates on this fresh official-substrate data (exact equality not required).
Paired on identical examples; stratified bootstrap; MAX_CONFIDENCE frozen before these data; no new
best method selected from outcomes; multiple-comparison status recorded for descriptive arms.
