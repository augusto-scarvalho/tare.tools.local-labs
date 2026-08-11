# RNN-05B-EXT — Audit Reconciliation (protocol ordering + causal scope + design correction, 2026-08-11)

**RNN-05B-EXT = ACCEPTED_AS_BLOCKED_WITH_AUDIT_CLARIFICATIONS.** The block stands and is accepted:
`H3_TESTABILITY = BLOCKED_BY_UNSTABLE_BASE`, `QWEN_GDN_TRANSPLANT_GATE = DEFER`. The independent audit accepts
the original BASE-only grid evidence, Amendment 1 being pre-committed before amended outcomes, the seed-42
selection of L512/d=0.50, the 3-seed instability, and all frozen/reader numbers **as real observations**. This
note **supersedes interpretation / protocol-scope / experiment-design only** — every RNN-05B-EXT raw artifact
(`PRE_REGISTRATION.md`, `AMENDMENT_1.md`, `amend1_grid.json`, `rnn05bext_results.json` root + `amend1/`,
`rnn05bext_outcomes.json`, `rnn05bext_summary.csv`, `run.log`) is **immutable and unchanged**. Machine-readable
form: `rnn05bext_audit_reconciliation.json`. HEAD verified `d8b91d9`. **No GPU, no rerun, no EXT2, not pushed.**

## 1. Protocol gate-ordering defect → post-block MC results are EXPLORATORY only
Pre-registration required: select seed-42 candidate → evaluate GDN BASE on **all** preregistered seeds →
QUALIFY or BLOCK → **only if QUALIFIED** run MC/reader outcome-bearing work. The runner instead executes **P2**
(BASE + param-free MC + `w_u` reader **training** + reader eval + recovery/harm) for all modes/seeds **before**
**P1c** computes the 3-seed stability qualification that sets `H3_TESTABILITY`. Control-flow proof
(`ops/rnn_05b_ext.py` run(), historical/unchanged; full excerpt in `source_controlflow_excerpt.txt`): reader
training at **line 676**, recovery/harm at **line 681**, inside the P2 loop (655–705); the stability gate at
**lines 707–713** runs afterward.
- `PROTOCOL_GATE_ORDERING = FAILED`
- `POST_STABILITY_GATE_MC_RESULTS = EXPLORATORY_NON_LOAD_BEARING` — the recovery/harm numbers are **post-stop
  exploratory observations**; they are **not deleted** and are **not** called invalid measurements.
- **EXT2 hard requirement:** all load-bearing BASE qualification (all preregistered seeds) must **complete and
  be persisted before any MC/reader outcome is computed**. (Historical runner behavior is **not** altered to
  pretend another order was used; only a doc note was added.)

## 2. Forgetting vs training-instability — do not over-claim the substrate
Calibration trains a **new** backbone independently per condition, so the retain/collapse cliff conflates
recurrent forgetting, optimization difficulty, basin/seed sensitivity, and training stability. **Retract**
"no stable graded-forgetting regime exists for the GDN substrate." Use:
- `TRAIN_PER_CONDITION_STABILITY = FAILED_UNDER_TESTED_INTERFERENCE_SWEEP`
- `FIXED_BACKBONE_GRADED_FORGETTING = NOT_TESTED`

Correct reading: no stable intermediate-performance regime was found when **independently training** this toy
DN/GDN family at each tested interference condition. The experiment does **not** establish that one
already-trained stable GDN cannot exhibit graded forgetting as inference-time retention pressure increases.

## 3. Snapshot-ablation scope defect → causal signal not qualified
The ablation used `proximal_idx=0`, `irrelevant_idx=n_snap-1` (a **global early-snapshot** ablation, not a
per-target nearest-write ablation). In the executed run `random_idx=0` (== `proximal_idx`) for **both** GDN and
DN (`n_snap=7`), so the random same-count control is a **duplicate** of the early ablation and is no
independent control.
- `EARLY_SNAPSHOT_ABLATION_SIGNAL = OBSERVED_DESCRIPTIVE`
- `TARGET_PROXIMAL_SNAPSHOT_CAUSALITY = NOT_QUALIFIED`
- `RANDOM_ABLATION_CONTROL = INVALID_DUPLICATE_OF_EARLY`
- `HISTORICAL_SNAPSHOT_CAUSAL_SIGNAL = INCONCLUSIVE / NOT_QUALIFIED` (supersedes the raw `SUPPORTED` label;
  raw JSON unchanged).
- **EXT2 requirement:** a deterministic random control that explicitly **excludes** the target/proximal and
  irrelevant indices; preferably a per-target or write-region-aware proximal ablation.

## 4. Collapsed-backbone wording
**Retract** "a collapsed final state has no useful history to read." Evidence supports only
`NO_USEFUL_HISTORICAL_RECOVERY_EXTRACTED_BY_TESTED_READER` under the collapsed-backbone seeds — historical
information content was **not** independently proven absent.

## 5. Calibration identity scope
`CALIBRATION_RULE_IDENTITY` checks only the numeric headroom/stability bounds, not the full Amendment-1 grid.
- `CALIBRATION_BAND_IDENTITY = PASS`
- `AMENDMENT_GRID_IDENTITY = NOT_MACHINE_QUALIFIED_IN_HISTORICAL_RUN` (still auditable: `AMENDMENT_1.md` +
  `amend1_grid.json` + executed result conditions exist and agree).
- **EXT2 requirement:** a stable `challengeGridSha256` recorded **identically** in PRE_REGISTRATION, machine
  config, run metadata, and final result, checked before outcomes.

## 6. Recovery/harm reporting — expose denominators; means of rates are not pooled rates
Per-seed denominators (raw `evaluate_paired` already recorded these; surfaced here):

| substrate·seed | n_base_wrong | n_recovered | RECOVERY | n_base_correct | n_harmed | HARM |
|---|---|---|---|---|---|---|
| la·42  | 9    | 1   | 0.111 | 4087 | 3   | 0.001 |
| dn·42  | 2747 | 339 | 0.123 | 1349 | 196 | 0.145 |
| dn·43  | 167  | 36  | 0.216 | 3929 | 16  | 0.004 |
| dn·44  | 37   | 8   | 0.216 | 4059 | 4   | 0.001 |
| gdn·42 | 873  | 142 | 0.163 | 3223 | 128 | 0.040 |
| gdn·43 | 3681 | 164 | 0.045 | 415  | 112 | 0.270 |
| gdn·44 | 3726 | 159 | 0.043 | 370  | 136 | 0.368 |

Do not treat a mean of per-seed rates as a pooled query-level rate (denominators vary by ~100×, e.g. LA 9 vs
GDN·42 873 BASE-wrong).

## 7. Seed screening is NOT a qualification strategy
**Retract** the handoff's "seed-screening to only stable seeds" tactic for load-bearing evidence.
`SEED_SCREENING_FOR_QUALIFICATION = PROHIBITED`. Final-evidence seeds must be preregistered and **all count**;
screening is debugging/development only. If a recipe is changed to improve stability, **preregister the recipe**
and rerun fixed/fresh seeds.

## 8. Next experiment design (record only; do NOT execute)
**RNN-05B-EXT2 = fixed-backbone inference-stress.** Train a stable backbone **once** → freeze exact weights →
vary memory pressure **at inference only** → calibrate BASE into the headroom band → freeze challenge → only
then evaluate historical snapshots / reader. Strong preference: **reuse the already-qualified stable RNN-05B
DN/GDN backbones** if model/task identity permits. This isolates retention/forgetting under **one fixed
representation** from training instability across conditions. EXT2 is the **final synthetic H3 attempt** before
deciding whether to park Memory Caching or design a real-Qwen packet.

## Net
No numerical result changed. RNN-05B-EXT is **ACCEPTED_AS_BLOCKED_WITH_AUDIT_CLARIFICATIONS**;
`H3_TESTABILITY = BLOCKED_BY_UNSTABLE_BASE`; post-block MC/reader numbers are **EXPLORATORY_ONLY**;
`QWEN_GDN_TRANSPLANT_GATE = DEFER`. Next: compact this session, then RNN-05B-EXT2 fixed-backbone inference-stress.
Do not start EXT2.
