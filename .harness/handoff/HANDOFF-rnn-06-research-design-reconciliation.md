# HANDOFF — RNN-06 Research/Design Reconciliation Pass (2026-08-11)

Reconciliation-only pass on the RNN-06 research/design packet. **No implementation, no GPU work, no large-weight
downloads, no Qwen/llama.cpp/SGLang/serving/deploy changes, no EXT3, nothing pushed.** Research direction ACCEPTED;
candidate decision UNCHANGED (PRIMARY Gated-DeltaNet-1.3B, FALLBACK Mamba-2-1.3B).

## Attached audit note
The reconciliation request referenced `AUDIT_RECONCILIATION_RNN-06_RESEARCH_DESIGN_2026-08-11.md`. That file was
**not present in the working tree** (searched repo-wide). Its seven requirements were, however, fully transcribed in
the reconciliation request itself, and are addressed below. No audit content was fabricated; if a copy of the audit
file should be tracked in-repo, provide it and it will be added append-only.

## Files changed (this pass)
- **Modified** `runs/rnn/RNN-06/RNN-06-RESEARCH-DESIGN-PACKET.md` — revision banner + 6 corrections (below).
- **Rewrote** `runs/rnn/RNN-06/RNN-06_candidate_matrix.json` — new lifecycle/complexity columns, stage model with
  RNN-06-P0, set-identity block, narrowed Mamba/Qwen wording (JSON is the machine-readable authority).
- **Regenerated** `runs/rnn/RNN-06/RNN-06_candidate_matrix.csv` — from JSON via `csv.writer` (proper quoting) +
  round-trip validated.
- **Added** `.harness/handoff/HANDOFF-rnn-06-research-design-reconciliation.md` (this file, untracked).
- No RNN-04/05A/05B/EXT/EXT2 artifact touched; all remain byte-identical.

## Exact research corrections (the 7 audit items)
1. **CSV repair.** Regenerated from JSON with a standard CSV writer so comma-containing fields (`S[H,dk,dv]`, brace
   lists) are quoted, not column-shifting. **JSON is authoritative** until/again after repair.
2. **Stage naming/ordering.** Added **`RNN-06-P0 — Frozen-Checkpoint BASE Regime Scout`** as an EXPLORATORY
   pre-packet (prove-runnable / freeze identity / cheap inference-only calibration sweep / judge band reachability /
   choose-or-falsify model+axis). Preserved RNN-06A (lifecycle) → 06B (confirmatory BASE) → 06C (info) → 06D
   (recovery). P0 statuses = `P0_GRADED_BAND = PLAUSIBLE | NOT_FOUND_WITHIN_BUDGET | MODEL_NOT_RUNNABLE`; P0 may
   **never** mint `FIXED_BACKBONE_GRADED_REGION = QUALIFIED` — **only 06B** may.
3. **Lifecycle risk upgrade.** Separated `state_externally_accessible` (observation API exists) from
   `state_semantically_checkpointable_restorable` (BIT_EXACT-or-bounded restore proven). Added
   `base_inference_complexity`, `state_introspection_complexity`, `lifecycle_qualification_risk`, `lifecycle_status`
   columns. **BIT_EXACT is not inferred from the Cache API**; GDN-1.3B (and all candidates) lifecycle status =
   `NOT_QUALIFIED` until RNN-06A proves it on the exact pinned backend/version/checkpoint (public GDN
   segmented/manual-cache discrepancies noted; risk = HIGH for GDN).
4. **Mamba wording narrowed.** "strongest published graded-forgetting evidence" → "strongest **directly-relevant
   state-capacity / long-context** evidence **among the readily runnable fallback candidates**." A smooth common
   graded band on our frozen `mamba2-1.3b` checkpoint = `NOT_QUALIFIED` until measured.
5. **Qwen backend wording narrowed.** "state introspectable **only** via transformers/fla" → "Transformers/FLA is
   the **currently identified** clean state-introspection path; equivalent access via vLLM/SGLang/llama.cpp is
   `NOT_QUALIFIED`" (not asserted impossible). Qwen stays `DEFER_VALIDATION_TARGET`.
6. **Calibration vs qualification separated.** Added `calibrationSetSha256` (P0-tunable, exploratory),
   `qualificationSetSha256` (06B confirmatory, **disjoint** from calibration), `stressGridSha256` (frozen ladder).
   Rules recorded: disjoint example sets; template families preferably disjoint (declared if shared); fixed
   process-stable integer seeds (not `hash()`-based); pinned generation version; P0 tunes only the pressure
   range/ladder; qualification set + stress grid + metrics frozen before any 06B outcome-bearing run.
7. **Candidate decision preserved.** PRIMARY Gated-DeltaNet-1.3B; FALLBACK/anchor Mamba-2-1.3B; controls GLA-1.3B,
   DeltaNet-1.3B, RecurrentGemma-2B; DEFER Qwen3.6-35B-A3B; WATCH DART/HOLA/SDM/GDN-2/StateX/Memory-Caching. Scope
   not broadened.

## CSV round-trip evidence (regenerate + validate)
Generator: Python `csv.writer(quoting=QUOTE_MINIMAL)` writing `columns` header then `[cand[k] for k in columns]`
rows from the JSON; validator re-parses the CSV with `csv.reader` and compares to the JSON.
```
candidates: json=14 csv=14
columns: 22
cells compared: 308
identities match: True
ROUND-TRIP: PASS (header, count, identities, and all shared fields match record-by-record)
```
Checks performed: header == columns; candidate count equal; candidate identities equal (order + value);
every shared field compared record-by-record (308/308 equal). Result: **PASS**.

## Source corrections
- No factual source was retracted. Wording corrections only: (a) observability≠checkpointability (state-semantics);
  (b) Mamba evidence scoped to "directly-relevant, among runnable fallbacks"; (c) Qwen serving-backend access marked
  `NOT_QUALIFIED` rather than impossible. Post-cutoff `[verify]` flags on 2602/2604/2605/2607/2608 papers retained.

## CURRENT × RESEARCH × PROPOSED
- **CURRENT (repo):** verified RNN-06 packet committed `3216229`; HEAD before this pass = `3216229`; EXT2 chain and
  all prior evidence immutable; not pushed.
- **RESEARCH (accepted):** real pretrained recurrent LM before Qwen; fla/`mamba_ssm` expose recurrent+conv state;
  graded knob = MQAR #KV-pairs vs fixed state (Based/Zoology/Stuffed-Mamba); on-thesis DART/HOLA/SDM lack runnable
  weights.
- **PROPOSED (after reconciliation):** exploratory **P0 scout** first (cheap, no confirmatory authority) → confirmatory
  06A lifecycle → 06B BASE → 06C info → 06D recovery; lifecycle `NOT_QUALIFIED` until proven; calibration/qualification
  contamination boundary enforced by three SHA identities; MC remains one 06D candidate.

## Confirmations
- **No GPU work:** no run invoked; no model weights downloaded; no inference executed.
- **No implementation:** research/design artifacts only; no experiment code authored/run.
- **No push:** `master` has no upstream; local only.
- **No historical modification:** RNN-04/05A/05B/EXT/EXT2 artifacts unchanged; only RNN-06 files edited + this handoff added.

## Exactly one next recommendation (NOT executed)
**Open a NEW implementation session for `RNN-06-P0 — Frozen-Checkpoint BASE Regime Scout`** (not RNN-06A): prove the
frozen `Gated-DeltaNet-1.3B` checkpoint runs on this box, freeze its identity, run a cheap inference-only MQAR
#KV-pair calibration sweep (Mamba-2-1.3B as parallel anchor), and report
`P0_GRADED_BAND = PLAUSIBLE | NOT_FOUND_WITHIN_BUDGET | MODEL_NOT_RUNNABLE` — minting no confirmatory qualification
(only RNN-06B may) and building no lifecycle/probe/recovery machinery.

STOP.
