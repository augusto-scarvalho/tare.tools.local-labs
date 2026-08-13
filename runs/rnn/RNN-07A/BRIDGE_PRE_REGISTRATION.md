# RNN-07A-BRIDGE — NoLiMa SEMI-SYNTHETIC CONTROLLED BRIDGE — PRE-REGISTRATION

Frozen BEFORE any bridge outcome. Corrective execution of the NoLiMa controlled bridge whose condition
was met (parent natural-workload `OPERATING_POINT = BLOCKED`) but not executed. NoLiMa results are
**`SEMI_SYNTHETIC_CONTROLLED_BRIDGE`** and never upgrade/overwrite the natural-workload negative. All
LongBench-v2 results and `REALISTIC_TASK_COMPETENCE = INSUFFICIENT` are preserved unchanged.

## Subject (unchanged, qualified)

`state-spaces/mamba2-1.3b` @ `c5b59d00ec85d313adea86a08cad2a43c962dd3b`, official `mamba_ssm` 2.2.4 fast
path (chunked prefill + selective_state_update decode), bf16, RTX 3090. `MAX_CONFIDENCE` frozen. No
selector tournament, no training/fine-tuning, no Qwen/DART/StateX/SDM/GDN-2/INT8/ReplaySSM, no
host-policy change. Append-only; nothing pushed.

## Workload (semi-synthetic controlled bridge)

NoLiMa (`amodaresi/NoLiMa` @ `378115b1…`; provenance in `EXTERNAL_WORKLOAD_PROVENANCE.json`).
- **Needles:** `needle_set_ONLYDirect.json` (the DIRECT-association subset — the easiest, most literal
  needles; best fair shot at a 1.3B base model). Needle template `"Actually, {CHAR} lives next to {1}."`;
  DIRECT question `"Which character lives next to {1}?"` where `{1}` = `test.input_args[0]` (a literal
  entity, e.g. "the Kiasma museum"). 10 needle templates × their `tests` = 28 (needle,test) combos.
- **MC scoring (deterministic):** 4-way option-likelihood over character names = {gold `{CHAR}`} + 3
  seeded distractors drawn from that needle's `character_set`. Prediction = argmax length-normalized
  option-likelihood (teacher-forced), confidence = max softmax over the 4 option scores. This is the
  SAME frozen readout family as the parent (`rnn_07a_lib.readout_from_state`). Chance = 0.25.
- **Haystack filler:** real book text `haystack/rand_shuffle/rand_book_1.txt` (natural prose, 103,691
  tokens). The needle sentence is planted into this filler.
- **Example pool:** for each (needle,test) combo, `N_CHAR = 4` seeded character assignments (gold + 3
  distractors), giving up to 112 examples (cap `MAX_EXAMPLES = 112`). Seed `20261400`.

## Context constructions (frozen; needle depth is a-priori, snapshots positional)

- **SHORT** context: needle sentence embedded in `SHORT_TOKENS ≈ 512` tokens of book filler (needle at
  ~15% depth of the short window). This is the competence probe.
- **LONG** cells: `8K / 16K / 32K` tokens of book filler with the needle planted at a **fixed a-priori
  depth `NEEDLE_DEPTH = 0.15`** (15% into the context), identical rule for every example — NOT tuned and
  NOT aligned to any snapshot boundary. `~8K` here is achievable because the bridge is constructed (unlike
  natural LongBench v2 whose native contexts are ≥13K).
- **Snapshots** (recovery): normalized context progress `25/50/75/90% + FINAL`, by token offset — chosen
  ONLY by position, never by needle location. (With needle at 15%, every snapshot ≥25% has read it; the
  recovery question is whether the FINAL state forgets a needle that earlier snapshots retained.)

## Deterministic scoring

Teacher-forced length-normalized option-likelihood over the 4 candidate names (content). Greedy/argmax,
no sampling. Same readout used from any state (short, long/FINAL, or a snapshot). No gold used to compute
the readout.

## Gated mints (frozen definitions)

- **`BRIDGE_SHORT_CONTEXT_COMPETENCE`** ∈ {SUFFICIENT, INSUFFICIENT}. SUFFICIENT iff the Wilson-95% lower
  bound of SHORT MC accuracy over the pool is `> 0.50` (clearly above the 0.25 four-way chance floor) AND
  the number of short-correct (competence-eligible) examples `≥ MIN_ELIGIBLE = 20`. Else INSUFFICIENT →
  **STOP**, long/recovery not run.
- **`BRIDGE_LONG_CONTEXT_DEGRADATION`** ∈ {QUALIFIED, NOT_QUALIFIED, NOT_RUN}. Runs only if short
  competence SUFFICIENT. On the competence-eligible population (short-correct; eligibility is independent
  of the long outcome — no example enters the forgotten population merely because LONG was wrong),
  QUALIFIED iff for at least one long cell, `SHORT_acc − LONG_acc ≥ DEGRADE_MARGIN = 0.15` with paired
  bootstrap CI lower bound `> 0`, AND the forgotten population {short-correct ∧ long-wrong} has size
  `≥ MIN_FORGOTTEN = 15`. The qualifying cell with the largest degradation is the recovery cell.
- **Only if `BRIDGE_LONG_CONTEXT_DEGRADATION = QUALIFIED`:**
  - **`BRIDGE_HISTORICAL_RECOVERY_SIGNAL`** ∈ {POSITIVE_SIGNAL, NO_SIGNAL, INCONCLUSIVE}. POSITIVE iff
    some fixed earlier snapshot recovers (answers correctly) a fraction of the forgotten population with
    paired bootstrap 95% CI LB `> 0` AND recovery-rate `≥ REC_MIN = 0.15`. INCONCLUSIVE if forgotten
    population `< MIN_FORGOTTEN`.
  - **`BRIDGE_ADAPTIVE_SELECTION_SIGNAL`** ∈ {POSITIVE_SIGNAL, NO_SIGNAL, INCONCLUSIVE}. POSITIVE iff
    frozen `MAX_CONFIDENCE` accuracy on the eligible population exceeds `FINAL` with paired bootstrap 95%
    CI LB `> 0` AND `Δ ≥ ADAPT_MIN = 0.05` (reported vs best fixed snapshot too). `ORACLE_BEST_GOLD` is
    diagnostic only.

All accuracies reported with raw denominators, Wilson intervals, paired stratified bootstrap
(`N_BOOT = 2000`), selector histograms, and reasoning-type strata. Fast-path firing verified via kernel
counters.

## Optional LongBench 64K descriptive cell

Permitted as a purely descriptive add-on (natural workload, native ≤64K). It MUST NOT rewrite the
existing natural-workload negative (`REALISTIC_TASK_COMPETENCE = INSUFFICIENT`,
`OPERATING_POINT = BLOCKED`); it only reports a descriptive accuracy point. Run only if budget allows.

## Budget & seeds

Remain within the original 3-hour train ceiling (parent used ~58 GPU min). Seeds disjoint: pool/char
assignment `20261400`, bootstrap `20261401`. Nothing pushed.
