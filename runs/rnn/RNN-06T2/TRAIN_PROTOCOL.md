# RNN-06T2 — Official-Mamba Lifecycle Requalification + Recovery Confirmation

**Packet:** `RNN-06T2-MAMBA-REQUALIFICATION`
**Predecessor:** RNN-06T (superseded; see `runs/rnn/RNN-06T/AUDIT_RECONCILIATION.md`).
**Substrate:** `state-spaces/mamba2-1.3b` @ `c5b59d00ec85d313adea86a08cad2a43c962dd3b`, official
`mamba_ssm` fast path (mamba_ssm 2.2.4 / causal_conv1d 1.5.0.post8 / triton 3.2.0 /
torch 2.6.0+cu124 / CUDA 12.4 / bf16 / RTX 3090). No weight mutation, no training.

## Two backlog items with a HARD dependency gate

- **Item 1 — RNN-06T2-T0R:** fresh, prospective *fixed-batch* single-pass historical-recovery
  lifecycle requalification. Fixes every RNN-06T strict-preregistration defect (batch-shape
  property substitution, `or True` fork tautology, zero-only reset, hash-only roundtrip, gate
  ordering). Mints `OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE`, `BATCH_SHAPE_NUMERICAL_PORTABILITY`,
  `SINGLE_PASS_HISTORICAL_CAPTURE_T0R`.
- **Item 2 — RNN-06T2-T1R:** fresh recovery confirmation + corrected apples-to-apples economics.
  Executes **only if** `OFFICIAL_MAMBA_FIXED_BATCH_LIFECYCLE = QUALIFIED` **AND**
  `SINGLE_PASS_HISTORICAL_CAPTURE_T0R = QUALIFIED`. Otherwise `RNN-06T2-T1R = BLOCKED_BY_T0R`.

## Gate discipline (HARD)

Scientific contracts, model/backend/source identity, preregistered thresholds, qualification
identities, and the dependency gate are frozen before substantive outcomes. A failed upstream gate
blocks downstream execution. No post-outcome amendment to make a gate green. MAX_CONFIDENCE is a
frozen selector — no selector tournament in this packet. Nothing pushed.

## Out of scope for this train (explicit)

Realistic-workload discovery, Qwen, selector/reader training, DART, StateX, Sparse Delta Memory,
Gated-DeltaNet-2, INT8 historical-state archive, ReplaySSM, and any host-policy change are all
excluded. The old NL needle scout remains `EXPLORATORY_NO_SIGNAL` and is neither extended nor relied
upon here.

## Files

- `T0R_PRE_REGISTRATION.md` — frozen T0R contract (this precedes T0R outcomes).
- `T0R_RESULTS.json`, `T0R_DECISION.md`.
- If gated: `T1R_PRE_REGISTRATION.md`, calibration/qualification specs, `T1R_*_RESULTS.json`,
  `T1R_ECONOMICS.json`, `T1R_DECISION.md`.
- `ENVIRONMENT_PROVENANCE.json` — live-verified environment/source identity.
