# RNN-07A — REALISTIC LONG-CONTEXT OPERATING-POINT DISCOVERY — PRE-REGISTRATION

Frozen BEFORE any RNN-07A outcome. This is **DISCOVERY**, not confirmation. If a positive operating
point is found, RNN-07A STOPS; RNN-07B (a fresh, separately preregistered set) would confirm later. The
discovered best cell must NOT be turned into confirmatory evidence.

## Subject (unchanged, qualified)

- Model: `state-spaces/mamba2-1.3b` @ `c5b59d00ec85d313adea86a08cad2a43c962dd3b`.
- Backend: official `mamba_ssm` 2.2.4 fast path (chunked prefill + selective_state_update step), bf16,
  RTX 3090, same qualified fixed-batch lifecycle/state handling as RNN-06T2 (`ops/rnn_06t_lib.py`).
- `MAX_CONFIDENCE` selection remains **FROZEN** (per-example pick the snapshot with the highest model
  confidence). **No selector or reader is trained or fine-tuned.** No Qwen. No DART/StateX/SDM/GDN-2/
  INT8-archive/ReplaySSM. No host-policy change. Append-only. Nothing pushed.

## Workload (PRIMARY, natural)

LongBench v2 (`THUDM/LongBench-v2`, `data.json`, 503 examples, 4-way multiple choice A/B/C/D), bounded
subsets only. Local copy at `/home/augus/data/longbench_v2/data.json` (NOT bundled; excluded as data).
Candidate priority (spec): Single-Document QA → Long-dialogue History Understanding → Multi-Document QA
→ Long Structured Data Understanding. `Long In-context Learning` and `Code Repository Understanding` are
excluded from the primary pool (off the forgetting-of-natural-prose thesis) but their exclusion is
logged. No RULER/NIAH substitution. NoLiMa is a controlled-bridge fallback only (see §Fallback) and is
never called a natural-workload result. LongBench Pro is not used (bounded natural subset already
available here).

### Measured dataset reality (motivates the cells; observed during ingest, pre-outcome)

gpt-neox-20b tokenizer, chars/token ≈ 3.49. Even the `short` length class is 13.7K–99K tokens
(median ≈ 28K). Native token budget yield: **≤8K: 0 examples**, ≤16K: ~10, ≤32K: ~105, ≤64K: ~171.
Consequently the `~8K` grid cell is **INFEASIBLE on natural LongBench v2 context** and is recorded as a
boundary condition — it is NOT reached by prefix-truncation, because truncating a long context to 8K
would remove the answer evidence and confound "forgetting" with "evidence absent." Cells actually run:
**~16K (native ≤16K, underpowered), ~32K (native ≤32K, PRIMARY), ~64K (native ≤64K, if budget)**.
All contexts are used at their **native length ≤ cell budget** (no evidence-bearing truncation).

## Deterministic scoring contract (frozen)

Because the subject is a 1.3B **base** pretrained model, we do NOT rely on free-form instruction
following. Two deterministic readouts, computed from a given model state (a snapshot, the FINAL state,
or the RAG-control state):

- **PRIMARY — TASK_CONTENT_COMPETENCE (teacher-forced option-likelihood).** From the state, step the
  query `question + "\nAnswer:"`, then for each of the 4 choices compute the **length-normalized**
  sum-log-prob of the choice's tokens (teacher-forced). Prediction = argmax over the 4 normalized
  option scores. **Confidence** = max of softmax over the 4 normalized option scores (used by
  MAX_CONFIDENCE). Deterministic (argmax; no sampling). This does not require the model to emit a
  letter, so content competence is separated from output format.
- **SECONDARY — OUTPUT_FORMAT_ADHERENCE (letter-constrained).** From the state, step an **enumerated**
  MC prompt (`question` + `"A. <cA>\nB. <cB>\nC. <cC>\nD. <cD>\nAnswer:"`) and read the next-token logits
  over the four letter tokens `{" A"," B"," C"," D"}`; format-adherence = fraction of examples whose
  letter-argmax equals the option-likelihood (content) argmax (i.e. the model, if asked to emit a
  letter, would emit the same answer). Degradation of interest must be in the PRIMARY (content) metric,
  not merely a drop in format adherence.

Both readouts are functions of state only; no gold answer is used to compute either.

## Target-agnostic COMPRESSED_OR_RAG control (frozen; establishes competence)

Question-conditioned BM25 retrieval — **target-agnostic** (uses the QUESTION only; never the gold answer
or its position):
1. Split the native context into non-overlapping ~512-token chunks.
2. Score each chunk by BM25 (Okapi, k1=1.5, b=0.75) against the tokenized QUESTION.
3. Concatenate the top chunks **in original document order** up to `RAG_BUDGET = 2048` tokens.
This CONTROL context is the competence probe: if the model can answer when the question-relevant text is
concentrated, it is competent on the task content. Retrieval never consults the answer choices' gold
label or its location, so no target leakage.

## Competence eligibility (frozen BEFORE fresh qualification)

- An example is **COMPETENCE-ELIGIBLE** iff it is answered **correctly under the RAG control** (PRIMARY
  option-likelihood). This is defined **independently of the full-context outcome**.
- **Hard rule:** *No example may enter the forgetting population merely because FULL_CONTEXT was observed
  wrong.* The forgetting population = {COMPETENCE-ELIGIBLE examples} (control-correct), and within it we
  then observe whether FULL/FINAL is wrong. Eligibility is frozen by control competence, not by full
  failure.

## Snapshots (frozen; positional, never gold-aligned)

Historical snapshots at **normalized context progress 25%, 50%, 75%, 90%, and FINAL (100%)** by token
offset into the native context. Snapshots are chosen **only by normalized position**, never using
gold/evidence location. Captured via prefix prefill: for each p ∈ {0.25,0.5,0.75,0.9,1.0}·L, prefill
`context[:round(p·L)]` on the fast path and clone the (conv,ssm) state.

## Arms evaluated

- `FINAL` — state after the full native context.
- Each fixed snapshot `SNAP_25/50/75/90` — state at that normalized position.
- `MAX_CONFIDENCE` (frozen) — per example, the snapshot (among 25/50/75/90/FINAL) with the highest
  option-likelihood confidence; **non-oracle** (uses only model confidence).
- `ORACLE_BEST_GOLD` — **diagnostic only** (upper bound): per example, any snapshot whose answer matches
  gold counts as correct. Never used to mint recovery/adaptive signals.

## Mints (frozen definitions)

- **`REALISTIC_TASK_COMPETENCE`** ∈ {SUFFICIENT, INSUFFICIENT}. SUFFICIENT iff, over the candidate pool,
  the **Wilson-95% lower bound** of RAG-control PRIMARY accuracy `> 0.35` (robustly above the 0.25 MC
  chance floor by margin) AND the number of competence-eligible examples `≥ MIN_ELIGIBLE = 20` (enough
  to power a recovery test). Otherwise INSUFFICIENT.
- **`REALISTIC_FORGETTING_OPERATING_POINT`** ∈ {FOUND, NOT_FOUND_WITHIN_BUDGET, BLOCKED}.
  - BLOCKED if `REALISTIC_TASK_COMPETENCE = INSUFFICIENT` (or the workload cannot be run within budget).
  - FOUND iff competence SUFFICIENT AND, on the competence-eligible population, FINAL PRIMARY accuracy is
    degraded vs control by `≥ DEGRADE_MARGIN = 0.10` with a paired bootstrap CI lower bound `> 0`, AND
    the degradation is **content, not format** (format-adherence on the eligible population does not drop
    by more than `FORMAT_TOL = 0.10`, i.e. the model is not simply failing to point at a letter), AND the
    "forgotten" population {control-correct ∧ FINAL-wrong} has size `≥ MIN_FORGOTTEN = 15`.
  - NOT_FOUND_WITHIN_BUDGET if competent but the degradation/forgotten-population conditions are not met
    within the GPU budget.
- **Only if FORGOTTEN OPERATING POINT = FOUND:**
  - **`REALISTIC_HISTORICAL_RECOVERY_SIGNAL`** ∈ {POSITIVE_SIGNAL, NO_SIGNAL, INCONCLUSIVE}.
    POSITIVE_SIGNAL iff some fixed historical snapshot recovers (answers correctly) a fraction of the
    forgotten population with paired bootstrap 95% CI lower bound `> 0` AND point recovery-rate
    `≥ REC_MIN = 0.15`. NO_SIGNAL if recovery-rate CI includes 0 / below noise. INCONCLUSIVE if the
    eligible/forgotten population is too small (`< MIN_FORGOTTEN`) to resolve.
  - **`REALISTIC_ADAPTIVE_SELECTION_SIGNAL`** ∈ {POSITIVE_SIGNAL, NO_SIGNAL, INCONCLUSIVE}.
    POSITIVE_SIGNAL iff frozen `MAX_CONFIDENCE` accuracy on the eligible population exceeds `FINAL` with
    paired bootstrap 95% CI LB `> 0` AND `Δ ≥ ADAPT_MIN = 0.05`; strengthened (reported, not required) if
    it also beats the best single fixed snapshot. NO_SIGNAL / INCONCLUSIVE otherwise.

All accuracies reported with raw denominators, Wilson intervals, paired stratified bootstrap
(`N_BOOT = 2000`), selector histograms, and length/task strata (and position strata where naturally
available). `ORACLE_BEST_GOLD` is diagnostic only.

## GPU budget (frozen)

Scout (competence + degradation screen) target `< 90 GPU minutes`; hard total ceiling `3 GPU hours`
(shared with RNN-06T2-E1). If the budget is hit before the recovery eval completes, mint
`REALISTIC_FORGETTING_OPERATING_POINT = NOT_FOUND_WITHIN_BUDGET` (or report recovery as INCONCLUSIVE)
rather than overrun.

## Seeds (disjoint)

Candidate ordering / sampling seed `20261300`; bootstrap seed `20261301`. Deterministic argmax readout
(no sampling temperature).

## Fallback (controlled bridge only)

If LongBench v2 is unusable (it is not — it loaded), NoLiMa would be a *controlled bridge* and explicitly
NOT reported as a natural-workload result. Not invoked here.
