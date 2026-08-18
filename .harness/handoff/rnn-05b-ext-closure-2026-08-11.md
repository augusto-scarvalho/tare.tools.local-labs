# HANDOFF — RNN-05B-EXT FINAL CLOSURE — 2026-08-11

Session-closure step for RNN-05B-EXT (direct H3 test). **No GPU, no rerun, no EXT2, not pushed.** The governing
audit reconciliation was already durably committed in the prior step; this closure adds **no code/evidence
changes** — it verifies Git, preserves historical truth, and packages the final bundle.

## Git
- **Starting HEAD (this closure): `df8317b`** (the audit-reconciliation commit) — already contained the
  governing interpretation.
- **Reconciliation commit created this step: NONE** (the reconciliation artifacts were already tracked +
  committed at `df8317b`; a new commit would be empty/redundant, so none was made).
- **FINAL HEAD: `df8317b5acc7c7d982663153f70bacf34892fdbf`** · branch **master** · **not pushed**.
- `trackedTreeClean = true`, `stagedTreeClean = true` (`git status --porcelain --untracked-files=no` empty).

### EXT commit chain (`aad66fc..HEAD`)
```
df8317b clarify(rnn): RNN-05B-EXT audit reconciliation — protocol ordering + causal scope
d8b91d9 results(rnn): RNN-05B-EXT amend1 -> H3_TESTABILITY=BLOCKED_BY_UNSTABLE_BASE
e7abcbd experiment(rnn): RNN-05B-EXT amendment 1 — pre-commit finer BASE-only distractor grid
9862e1b results(rnn): RNN-05B-EXT original pre-registered grid -> H3_TESTABILITY=BLOCKED
83ac6c5 test(rnn): RNN-05B-EXT memory-bound H3 challenge + pre-registration
```

### Reconciliation commit `df8317b` — diffstat (pure additions; nothing rewritten)
```
 ops/rnn_05b_ext.py                                      | 17 ++++   (doc-only comments; behavior UNCHANGED)
 runs/rnn/RNN-05B-EXT/AUDIT_RECONCILIATION.md            | 98 +++++++   (new governing interpretation)
 runs/rnn/RNN-05B-EXT/rnn05bext_audit_reconciliation.json| 95 +++++++   (machine-readable form)
 runs/rnn/RNN-05B-EXT/source_controlflow_excerpt.txt     | 33 ++++    (protocol-ordering evidence)
 4 files changed, 243 insertions(+), 0 deletions(-)
```

## Governing classifications (accepted)
- `RNN-05B-EXT = ACCEPTED_AS_BLOCKED_WITH_AUDIT_CLARIFICATIONS`
- `H3_TESTABILITY = BLOCKED_BY_UNSTABLE_BASE`
- `PROTOCOL_GATE_ORDERING = FAILED` · `POST_STABILITY_GATE_MC_RESULTS = EXPLORATORY_NON_LOAD_BEARING`
- `TRAIN_PER_CONDITION_STABILITY = FAILED_UNDER_TESTED_INTERFERENCE_SWEEP` · `FIXED_BACKBONE_GRADED_FORGETTING = NOT_TESTED`
- `TARGET_PROXIMAL_SNAPSHOT_CAUSALITY = NOT_QUALIFIED` · `RANDOM_ABLATION_CONTROL = INVALID_DUPLICATE_OF_EARLY`
  · `HISTORICAL_SNAPSHOT_CAUSAL_SIGNAL = INCONCLUSIVE / NOT_QUALIFIED`
- `SEED_SCREENING_FOR_QUALIFICATION = PROHIBITED`
- `QWEN_GDN_TRANSPLANT_GATE = DEFER`

## Source / control-flow evidence for the protocol-ordering defect
`ops/rnn_05b_ext.py` run(): the P2 frozen-H3 loop trains the `w_u` reader (**line 676**) and computes
recovery/harm (**line 681**) for **all** modes/seeds (loop 655–705) **before** the 3-seed stability gate that
sets `H3_TESTABILITY` (**lines 707–713**). Therefore MC/reader outcomes were computed before the BLOCK decision
→ they are exploratory, non-load-bearing. Full excerpt: `runs/rnn/RNN-05B-EXT/source_controlflow_excerpt.txt`.
Historical runner behavior was **not** altered (only doc comments were added).

## Historical truth preserved (NOT rewritten)
Unchanged/immutable: `PRE_REGISTRATION.md`, `AMENDMENT_1.md`, `amend1_grid.json`, historical
`rnn05bext_results.json` (root + `amend1/`), `rnn05bext_outcomes.json`, `rnn05bext_summary.csv`, `run.log`. The
reconciliation (`AUDIT_RECONCILIATION.md` + `rnn05bext_audit_reconciliation.json`) supersedes **interpretation
only**.

## External checkpoint disposition (durable, non-Git)
7 frozen+reader checkpoints in `.harness/artifacts_ext/` (LA×1, DN×3, GDN×3; ~377 KB each; all
`BACKBONE_WEIGHT_MUTATION=0`; SHA-256 in `rnn-05b-ext-2026-08-11.md`). Kept as durable non-Git artifacts and
included in the audit ZIP. Not committed (per policy: no `.harness/`, checkpoints, ZIPs, venvs, caches, RNN-08
adapter dirs, or raw stdout logs in Git).

## Intentional untracked paths (all policy-expected; nothing to commit)
`.harness/` (checkpoints + handoffs + ZIPs) · `runs/rnn/**/git_evidence*.txt` · `runs/rnn/RNN-05B-EXT/**/stdout.log`
· RNN-08/08b adapter dirs (`runs/rnn/RNN-08*/**/adapter_*`, `canary_lora_adapter`, `gate_tptt_adapter`).

## Next (exactly one)
**RNN-05B-EXT2 — fixed-backbone inference-stress, in a NEW session.** Train ONE stable backbone → freeze exact
weights → vary memory pressure **at inference only** → calibrate BASE into the headroom band → freeze challenge
→ only then evaluate historical snapshots / reader. Strong preference: **reuse the already-qualified stable
RNN-05B DN/GDN backbones** if identity permits (isolates forgetting-under-fixed-representation from training
instability). Hard requirements carried forward: BASE qualification persisted **before** any MC/reader outcome;
deterministic random ablation control that excludes proximal+irrelevant (plus per-target proximal ablation);
`challengeGridSha256` identical across prereg/config/metadata/result; preregistered seeds all count (no
screening). EXT2 is the **final synthetic H3 attempt** before deciding park-Memory-Caching vs real-Qwen packet.
Do NOT auto-start. No Qwen/serving/llama.cpp/deploy/TPTT/RNN-05C.
