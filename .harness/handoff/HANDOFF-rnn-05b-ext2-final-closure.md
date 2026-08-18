# HANDOFF — RNN-05B-EXT2 FINAL CLOSURE (2026-08-11)

Session-closure micro-packet. Adds an append-only audit reconciliation and closes RNN-05B-EXT2. **No experiment
rerun. No historical outcome artifact modified. Nothing pushed. No RNN-06. No EXT3.**

## Before / after HEAD
- **Before:** `2b5e946d64f25d3cdad69bdfe9f7bfad51f1bf04`
- **After:** `03a863f8b27736b62885916ba98bada26d720fd3`
- Branch: `master` · Upstream: none · **Pushed: NO**

## Exact commit made (only one)
```
03a863f clarify(rnn): RNN-05B-EXT2 audit reconciliation — session closure (append-only)
 runs/rnn/RNN-05B-EXT2/AUDIT_RECONCILIATION.md | 89 +++++++++++++++++++++++++++
 1 file changed, 89 insertions(+)
```
`git show --stat 03a863f` → exactly one file added: `runs/rnn/RNN-05B-EXT2/AUDIT_RECONCILIATION.md`.
No amend, no rebase, no rewrite of the EXT2 result commits (`5abeab4` prereg, `12157b4` results, `2b5e946`
evidence remain byte-identical).

## Files changed
- Added (committed): `runs/rnn/RNN-05B-EXT2/AUDIT_RECONCILIATION.md`.
- Added (NOT committed, this closure handoff): `.harness/handoff/HANDOFF-rnn-05b-ext2-final-closure.md`.
- Modified: **none** (no original artifact touched).

## Final `git status`
```
## master            (no upstream; NOT pushed)
# staged: none
# EXT2 evidence dir: committed/clean
# untracked (pre-existing, unrelated to EXT2): .harness/, various runs/rnn/*/git_evidence.txt, RNN-08 artifacts
```

## Confirmations
- **Nothing pushed:** `@{u}` resolves to none; local `master` only.
- **No experiment rerun:** no `--run` invoked; no GPU work; no change to `run.log`, results, outcomes,
  checkpoints, or `BASE_QUALIFICATION.json`.
- **No historical outcome artifact modified:** `PRE_REGISTRATION.md`, `machine_config.json`,
  `BASE_QUALIFICATION.json`, original `HANDOFF.md`, `rnn05bext2_results.json`, `rnn05bext2_outcomes.json`,
  `rnn05bext2_curves.csv`, and `artifacts/*.pt` are all unchanged. RNN-05B and RNN-05B-EXT evidence untouched
  (every EXT2 commit touches only `runs/rnn/RNN-05B-EXT2/**` + `ops/rnn_05b_ext2.py`).
- **Ordering intact:** `5abeab4` (pre-registration) precedes `12157b4` (results); `challengeGridSha256`
  `66ff24765d17c4fa…` identical across PRE_REGISTRATION / machine_config / BASE_QUALIFICATION / results / outcomes.
- **Commit-reference precision:** `12157b494b2cbbc32ff310c4c0e458bd80a27784` is the **results** commit (confirmed),
  not the current HEAD; HEAD advanced to `2b5e946` (append-only git-evidence) and now `03a863f` (this closure).
  This matches the originally reported "FINAL HEAD: 2b5e946" and is a labeling precision, not a contradiction.

## The two audit findings and their disposition (both LOW; verdict unchanged)
1. **`rnn05bext2_curves.csv` header-only** — CONFIRMED (1 line, no data). Root cause: `_write_csv()` reads
   `R["curves"]`, populated only in the P3 MC phase, which the Case-A stop skips. **Authoritative BASE curves live
   in `BASE_QUALIFICATION.json` → `per_seed_base_curves`** (9 doses × 7 backbones). Disposition: documentation-only;
   original CSV left unchanged.
2. **HANDOFF "AURC_RETENTION ≈ 0.99"** — CONFIRMED alongside the correct "not computed (downstream of the blocked
   gate)". The "≈ 0.99" is a **descriptive** characterization, not a preregistered metric. Post-hoc descriptive
   AURC (audit calc only, non-authoritative): gdn 0.9967/0.9939/0.9854, dn 0.9826/0.9973/0.9986, la 0.9994.
   Disposition: documentation-only; original HANDOFF left unchanged.

## Accepted verdict (unchanged)
`FIXED_BACKBONE_GRADED_REGION = BLOCKED` · `H3_TESTABILITY = BLOCKED_FIXED_BACKBONE` ·
`H3 = BLOCKED_FIXED_BACKBONE` · `QWEN_GDN_TRANSPLANT_GATE = DEFER` · `SYNTHETIC_DENSE_MC = PARK` ·
`EXT3 = DO_NOT_OPEN`.

Scope: RNN-05B-EXT2 does NOT prove a fixed GDN can never forget, nor that historical recurrent state is useless.
It proves the preregistered fixed-backbone recipe did not generate a qualified graded-forgetting regime and so
could not test H3. Synthetic dense Memory-Caching is PARKED.

## Exactly one next recommendation
**OPEN RNN-06 AS A NEW RESEARCH/DESIGN PACKET IN A NEW SESSION** (H3 on a real recurrent LM; its own
pre-registration). Do not continue EXT.

STOP.
