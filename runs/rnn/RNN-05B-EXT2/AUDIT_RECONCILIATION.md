# RNN-05B-EXT2 — Audit Reconciliation (session closure, append-only, 2026-08-11)

**Append-only.** This note adds documentation reconciliation ONLY. It does **not** modify any measured result,
`PRE_REGISTRATION.md`, `machine_config.json`, `BASE_QUALIFICATION.json`, the original `HANDOFF.md`, checkpoints,
or any historical (RNN-05B / RNN-05B-EXT) evidence. No experiment was rerun. Nothing was pushed.

## 1. Scientific decision — ACCEPTED (unchanged)
The independent audit accepted the Case-A verdict, and the repository confirms it:

- `FIXED_BACKBONE_GRADED_REGION = BLOCKED`
- `H3_TESTABILITY = BLOCKED_FIXED_BACKBONE`
- `H3 = BLOCKED_FIXED_BACKBONE`
- `QWEN_GDN_TRANSPLANT_GATE = DEFER`
- `SYNTHETIC_DENSE_MC = PARK`
- `EXT3 = DO_NOT_OPEN`

No numerical result is changed by this reconciliation.

## 2. Git state independently verified against the current repository
- **Current HEAD:** `2b5e946d64f25d3cdad69bdfe9f7bfad51f1bf04` · **branch:** `master` · **not pushed** (no upstream).
- **Working tree:** no staged changes; the EXT2 evidence directory is committed/clean (unrelated untracked items
  from other packets — `.harness/`, prior `git_evidence.txt` files, RNN-08 artifacts — are pre-existing and not
  part of EXT2).
- **EXT2 commit chain (ordering intact):**
  `5abeab4` pre-registration → `12157b4` results → `2b5e946` git-evidence (append-only).
- **Commit-hash precision (reconciliation of the reported reference):** `12157b494b2cbbc32ff310c4c0e458bd80a27784`
  **is the RESULTS commit** (confirmed to exist and to carry the Case-A result), but it is **not** the current HEAD.
  HEAD is one commit ahead at `2b5e946` — the append-only `git_evidence.txt` commit made *after* results (this was
  already stated as "FINAL HEAD: 2b5e946" in the original delivery). This is a labeling precision, **not** a
  contradiction of the science or of the reported result commit.
- **PRE_REGISTRATION → result ordering:** intact (`5abeab4` precedes `12157b4`); `challengeGridSha256`
  `66ff24765d17c4fa95dcfcbaf4a7b374c66aa1ba52e507cd047b486ad512a9e5` is identical across PRE_REGISTRATION.md,
  machine_config.json, BASE_QUALIFICATION.json, results, and outcomes.
- **Historical evidence untouched:** every EXT2 commit (`5abeab4`, `12157b4`, `2b5e946`) touches only
  `runs/rnn/RNN-05B-EXT2/**` and `ops/rnn_05b_ext2.py`. No file under `runs/rnn/RNN-05B-delta-gdn/` or
  `runs/rnn/RNN-05B-EXT/` was modified. RNN-05B and RNN-05B-EXT evidence remain immutable.

CURRENT confirms the audit (with the commit-label precision above). No blocking discrepancy was found.

## 3. Finding 1 (LOW) — `rnn05bext2_curves.csv` is header-only
**Confirmed in repo:** `rnn05bext2_curves.csv` contains exactly one line (the header
`substrate,seed,method,dose_0.0,…,dose_0.64,AURC`) and **no data rows**.
- **Root cause (evidence-export defect, not missing data):** the CSV writer `_write_csv()` iterates
  `R["curves"]`, which is populated only inside the P3 MC phase (`ops/rnn_05b_ext2.py`). The Case-A stop returns
  from `run()` **before** P3, so `R["curves"]` is empty when `finalize()` writes the CSV. This is a downstream
  export artifact of the (correct) early stop — **not** evidence that BASE curves were absent.
- **Authoritative source of the executed BASE curves:** **`BASE_QUALIFICATION.json` → `per_seed_base_curves`**
  holds all 9 doses × all 7 backbones (GDN/DN seeds 42/43/44, LA seed 42). This is the machine-readable authority
  for this Case-A-blocked run.
- **Disposition:** documentation-only. The original CSV is **left unchanged** (append-only discipline). No rerun.

## 4. Finding 2 (LOW) — HANDOFF descriptive "AURC_RETENTION ≈ 0.99"
**Confirmed in repo:** the original `HANDOFF.md` says descriptively "`AURC_RETENTION ≈ 0.99` for all curves"
(≈ line 95) while a later section correctly states the AURC/D50/width/recovery/etc. were "**Not computed** — all
are downstream of the graded-region gate, which BLOCKED" (≈ line 105).
- **Reconciliation:** the "≈ 0.99" is a **descriptive eyeball characterization** of the flat-high BASE curves, not
  a packet-authoritative preregistered outcome metric. It does not contradict the "not computed" statement: no
  formal preregistered AURC analysis was executed, because Case A blocked before downstream outcome analysis.
- **Post-hoc descriptive AURC (audit calculation ONLY — normalized trapezoid over dose 0..0.64; NOT
  packet-authoritative):**
  gdn s42 0.9967 · gdn s43 0.9939 · gdn s44 0.9854 · dn s42 0.9826 · dn s43 0.9973 · dn s44 0.9986 · la s42 0.9994.
  These corroborate "≈ 0.99" but carry **no** decision weight; the preregistered metrics remain **not computed**
  by design of the Case-A stop.
- **Disposition:** documentation-only. The original `HANDOFF.md` is **left unchanged**.

## 5. Authoritative-interpretation clauses (preserved for closure)
- `BASE_QUALIFICATION.json` is the **authoritative machine-readable source** for the executed BASE curves in this
  Case-A-blocked run.
- The empty `rnn05bext2_curves.csv` is an **evidence-export defect** (the CSV writer reads `R["curves"]`, not
  populated before the Case-A stop). It is **not** evidence that BASE curves were absent.
- Any AURC values recomputed from the BASE curves after the fact are **descriptive audit calculations only**, not
  packet-authoritative preregistered outcome metrics.

## 6. Final scientific scope (preserved, unchanged)
RNN-05B-EXT2 does **not** prove that a fixed GDN can never forget, and does **not** prove historical recurrent
state is useless. It proves that the **preregistered fixed-backbone recipe** (train once, single-state, mixture
over the nested distractor-density ladder; freeze; vary distractor density at inference) **did not generate a
qualified graded-forgetting regime** and therefore **could not test H3**. Two independent preregistered synthetic
attempts have now failed to yield a testable graded regime: RNN-05B-EXT via an unstable base,
RNN-05B-EXT2 via a non-degrading fixed base.

The synthetic dense Memory-Caching line is **PARKED**. **No EXT3.** If research continues later, **RNN-06 is NEW
SCOPE on a real recurrent LM** and requires its own research/design/pre-registration packet in a new session.

## Net
No measured result changed. The Case-A verdict stands and is accepted. Two LOW-severity documentation findings are
reconciled here (append-only) without altering any original artifact. `BASE_QUALIFICATION.json` is authoritative
for the executed BASE curves; the header-only CSV is an export defect, not missing evidence; post-hoc AURC is
descriptive only. Nothing pushed; no rerun; historical evidence immutable.
