# RNN-06T — Official Mamba Transportability + Single-Pass Historical Recovery — TRAIN PROTOCOL

**Author:** Claude (Opus 4.8) via Claude Code. **Date:** 2026-08-12. **Branch:** `master` (no
upstream). **Pushed:** NO. Two backlog items (T0 lifecycle+capture; B transportability+selector) with
hard internal gates. Follows the RNN-06D audit reconciliation ([[AUDIT_RECONCILIATION.md]]).

## Frozen official subject (resolved LIVE)

`state-spaces/mamba2-1.3b` @ immutable revision `c5b59d00ec85d313adea86a08cad2a43c962dd3b`; loaded via
the **official** `mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel` (NOT transformers); d_model
2048, 48 layers, Mamba2, bf16, cuda. Fast-path stack (frozen): mamba_ssm 2.2.4, causal_conv1d
1.5.0.post8, triton 3.2.0, torch 2.6.0+cu124 (cxx11abi=False), CUDA 12.4, RTX 3090 (cc 8.6), driver
591.86. `OFFICIAL_MAMBA_FASTPATH = RUNNABLE` — kernel firing proven (see `OFFICIAL_MAMBA_ENV.json` /
`ENVIRONMENT_PROVENANCE.md`). State per sequence = 52,002,816 bytes (conv (4352,4) + ssm (64,64,128)
bf16, ×48 layers).

## Canonical execution path (single trajectory) — binds all outcome-bearing state work

The FINAL state and every historical snapshot come from **one** trajectory: a short prompt **prefill**
(`mamba_chunk_scan_combined`) of the first slot, then autoregressive **step** decode
(`selective_state_update` + `causal_conv1d_update`, the Triton/CUDA fast path) advancing one token at a
time under a single `InferenceParams`. Snapshots at token boundaries {156,308,464,616} (slots
{38,76,115,153}) and FINAL at token 768 (slot 191) are captured **in-run** from that continuing
trajectory — NOT by re-prefilling the prefix from token zero (that was the RNN-06D approximation).
FINAL is the step continuation endpoint, not a separate full-sequence prefill. This is the contract
2.1 demands: same-path continuation, not full-prefill-FINAL vs independent-prefix-prefill-snapshots.

## Permanent boundaries (this train)

No Qwen; no trained reader; no DART; no synthetic dense Memory Caching; no StateX / SDM / GDN-2 /
INT8 archive / ReplaySSM; no host-policy/driver mutation; no serving change; nothing pushed. Ordinary
local engineering/debugging autonomous. MAX_CONFIDENCE (constrained answer-space top-1 probability,
same definition as 06D) is FROZEN before this train's qualification data — never retuned.

## Item A — T0 (gates everything)

1. `OFFICIAL_MAMBA_STATE_CONTRACT.json` — every state field: owner, shape, dtype, device, sequence
   ownership, serialization, restore/branch/reset semantics.
2. Lifecycle qualification (held-out deterministic sequences, NOT from 06A; tolerances in
   `T0_PRE_REGISTRATION.md`): A same-path replay, B save/destroy/reload/continue vs uninterrupted,
   C branch/fork parent-unchanged, D neighbor isolation, E reset/reuse, F serialize roundtrip, G
   batch slice ownership, H snapshot temporal identity, I weights immutable, J backend frozen.
3. Single-pass capture (2.4): one run over a synthetic sequence, capture ACTUAL in-run states at
   {38,76,115,153}+FINAL with `runId`, monotonic boundaries, `snapshot.runId==final.runId`; each
   snapshot hash == the state at that exact boundary in an uninterrupted replay (proving it is the
   real in-run state, not a later recompute). Restore each into an independent branch and read out.
4. Mint `OFFICIAL_MAMBA_LIFECYCLE ∈ {QUALIFIED|NOT_QUALIFIED|NOT_RUNNABLE}` and
   `SINGLE_PASS_HISTORICAL_CAPTURE ∈ {QUALIFIED|NOT_QUALIFIED}`. **Both QUALIFIED required for Item B.**
   Else persist negative evidence, package, STOP (no recovery).

## Item B — transportability + selector (only if T0 both QUALIFIED)

**3A exact-contract replication.** Fresh disjoint qualification set, same 06D semantics (M=192, band
[8,64], K=4, schedule [38,76,115,153]); MAX_CONFIDENCE frozen. Arms: FINAL, ORACLE_BEST_GOLD (diag),
ORACLE_TARGET_PROXIMAL (diag), **FIXED_SLOT_76** (mandatory non-adaptive control), MAX_CONFIDENCE
(frozen adaptive), optional RECENCY/FIXED_SLOT_153, MATCHED_NO_HISTORY. Two distinct claims, separate
SESOIs: **CLAIM 1** historical recovery transport (FIXED_SLOT_76 vs FINAL; MAX_CONFIDENCE vs FINAL);
**CLAIM 2** adaptive selector incremental value (MAX_CONFIDENCE vs FIXED_SLOT_76). Mint
`OFFICIAL_MAMBA_SYNTHETIC_TRANSPORT ∈ {QUALIFIED|PARTIAL|NOT_REPLICATED}`, separating
`HISTORICAL_RECOVERY_TRANSPORT` from `ADAPTIVE_SELECTOR_ADVANTAGE`.

**3B wide-target generalization** (only if 3A recovery positive). Band [8,144], region strata
{[8,38],[39,76],[77,115],[116,144]}, fresh calibration selects `BEST_FIXED_SNAPSHOT`, fresh disjoint
qualification. Primary MAX_CONFIDENCE vs BEST_FIXED_SNAPSHOT. Mint `WIDE_TARGET_RECOVERY` and
`ADAPTIVE_SELECTION ∈ {QUALIFIED|DIRECTIONAL|NOT_QUALIFIED}`.

## Section 4 — true end-to-end economics

Measure FINAL-only run vs recovery-enabled run: capture overhead, transfer, restore, readout,
selection, total added latency, throughput; net recovery / MiB, / added ms; compile/cold/warm split.
Capture is INCLUDED in the end-to-end metric. Mint
`END_TO_END_RECOVERY_UTILITY ∈ {QUALIFIED|COST_FAIL|NOT_QUALIFIED}`.

## Section 5 — optional non-synthetic scout (only if 3A transport positive)

Bounded, exploratory, deterministic scoring, target position varies, intervention target-agnostic.
Mint `NON_SYNTHETIC_RECOVERY_SCOUT ∈ {POSITIVE_SIGNAL|NO_SIGNAL|BLOCKED}`. Does not authorize Qwen.

## Statistics / discipline

Paired on identical examples; stratified/cluster bootstrap; expose denominators; no new best method
from qualification outcomes; multiple-comparison status recorded for secondary arms. Append-only
commits; no amend/rebase of outcome history; no weights/caches/venv/.git in commits; nothing pushed.
Report all decision states separately (never "Mamba PASS").
