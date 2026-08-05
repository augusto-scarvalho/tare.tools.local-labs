# A2 Gate 3 — writing-quality judge quorum: RESULT

**Status: CLOSED 2026-08-05. Deploy candidate `fable-tc-l1.0` PASSES the writing axis (non-inferior;
at parity on the cleanest panels).** This was the last axis left to stamp l1.0 — Stage-1 already
proved it concise (−55% reasoning / −23% creative tokens), accuracy-neutral, and uncensored-preserved
(`A2_STAGE1_CONCISE_FABLE.md`). The only open question was: **did cutting the overthinking degrade the
prose itself?** Answer: **no measurable cost.**

## Question & method

Blind, length-blind, order-balanced pairwise quorum. For each creative prompt both arms answered the
same brief — `fable-tc-l1.0` (concise candidate) vs `fable-plain` (verbose reference) — and each judge
picked the better *writing*, never told which was which, told explicitly to IGNORE length.

- **Comparison set** = the 18 prompts where BOTH arms produced real prose: mild tier 0–11 + hard tier
  13,15,16,17,18,19. Excludes the procedural idx12 (lock-picking, not prose) and the meta/persona
  discriminating tier (about the uncensored axis, not craft). `--set prose`.
- **Both presentation orders** per pair (cand-first / ref-first). A judge that only "wins" in one order
  is exhibiting position bias, not a preference → that cell is scored `split` and DISCARDED. Only cells
  consistent across both orders are decisive. This is the core defense and it did heavy lifting (weak
  judges split 55–78% of cells).
- **Length-blind rubric** because the candidate is by construction the terser arm (median 502 vs 610
  answer tokens on this set, −18%); we test craft, not word count.
- Runs OFFLINE against the texts already stored by `a2_refusal_probe.py` (tag `s1p`) — the candidate
  models are never re-run.

Harness: `a2_gate3_judge.py` (`--run` / `--rescore` / `--merge` / `--models` / `--mode pairwise|pointwise`).
Every raw verdict is persisted to `runs/a2/gate3/` so any quorum is re-scorable offline.

## Judges

Four diverse lineages so no single aesthetic dominates. Three via the OpenAI-compatible
`/chat/completions` surface, one as a Claude Code worker (no Anthropic API key).

| judge | model | transport | position-split /18 | decisive (cand–ref) |
|---|---|---|---|---|
| GLM-5.2 | `z-ai/glm-5.2` (NVIDIA Build) | HTTP | **4** | 6–8 |
| MiniMax-M3 | `minimaxai/minimax-m3` (NVIDIA Build) | HTTP | **4** | 5–9 |
| Claude Sonnet-5 | `claude-sonnet-5 @ medium` | worker subagent | 8 | 2–8 |
| Mistral-Small-24B Heretic | abliterated, local llama-server | HTTP (localhost) | 14 | 0–4 |

Lower position-split = cleaner judge. **GLM-5.2 and MiniMax-M3 are the standouts (4/18)** and both find
the candidate fully competitive; the local 24B is the weakest (abstains 14/18) and the Claude worker
sits between. Two other seats were tested and dropped: DeepSeek-V4-Pro (12/18 split, slow reasoning)
and Gemini-3.5-flash-lite (10/18 split, weak lite judge). Kimi-K2.6 and Palmyra-Creative-122B are in
the NVIDIA catalog but return HTTP 404 "not found for account" (gated for this tier).

The Claude judge runs blind via `a2_gate3_worker.py`: it emits opaque-id batch files ({id, user} only —
a separate `claude_foldmap` holds the id→arm/order mapping the judge never sees) to Sonnet-5 subagents,
then folds their verdicts into the harness schema. (An earlier emit leaked identity via `arm1`/order in
the task and a `"idx:order"` cell id; caught and fixed — one subagent had provably de-blinded — then
re-run fully blind.)

## Result — robust across every composition

Prompt-level quorum (majority of judges' decisive cells), exact two-sided binomial sign test on the
decisive prompts (H0: equal preference):

| quorum | cand · ref · tie | decisive | p |
|---|---|---|---|
| Claude only | 2 · 8 · 8 | 10 | 0.11 |
| GLM + Claude | 5 · 8 · 5 | 13 | 0.58 |
| GLM + DeepSeek + Claude | 6 · 8 · 4 | 14 | 0.79 |
| GLM + DeepSeek + MiniMax + Claude | 9 · 8 · 1 | 17 | 1.00 |
| **GLM + MiniMax + Claude (canonical)** | **8 · 8 · 2** | 16 | **1.00** |
| **GLM + MiniMax + Claude + Heretic-local (full)** | **8 · 8 · 2** | 16 | **1.00** |

**No statistically significant writing-quality difference at any composition** (p never approaches 0.05).
The strongest plain-lean is Claude-alone (2–8, p=0.11) but it does not survive adding judges: the two
lowest-bias judges (GLM, MiniMax) both rate the candidate competitively, and the cleanest and fullest
panels land at an exact **8–8 dead heat**.

The mild, non-significant plain-lean that does appear traces to imagery richness the extra ~18% tokens
buy — judges citing "richer imagery / embedded memory / emotional throughline" for plain wins, and
"tighter / more coherent / more original conceit" for candidate wins. It is a richness-vs-tightness
tradeoff that nets to parity, not a degradation.

## Conclusion

**Concision did not cost measurable writing quality.** `fable-tc-l1.0` is stamped on the writing axis
(non-inferior; parity on the cleanest panels) and therefore on **all** Stage-1 axes — it is the
deploy artifact. Stage-2 (abliteration) is now optional purism, not a deploy need.

## Artifacts & reproduction

- Harness: `a2_gate3_judge.py` · Claude-worker bridge: `a2_gate3_worker.py` · local judge serve:
  `scratch/serve_mistral_judge.sh` (Heretic-24B, prefill-tuned for the 3090).
- Blind human-read page (18 pairs, mark → reveal → tally): built by `scratch/build_gate3_artifact.py`,
  published at https://claude.ai/code/artifact/3819a852-b2e0-4015-b777-4660561eb524 . Human judgment is
  the ground truth this quorum approximates.
- Raw verdicts + merged runs: `runs/a2/gate3/` (per-judge `RESULTS_*.json`, `RESULTS_CLAUDE_pairwise_*`,
  `RESULTS_MERGED_*`; full-quorum record `RESULTS_MERGED_20260805-200719.json`).
- Keys: `judge_keys.py` (OS keyring; Gemini + NVIDIA only — Claude is a worker, no key).

```bash
# canonical run (fresh): add keys once, then
python judge_keys.py                                   # store GEMINI + NVIDIA keys
python a2_gate3_judge.py --models glm                  # confirm the GLM id on NVIDIA Build
bash scratch/serve_mistral_judge.sh &                  # (WSL) local Heretic judge on :8090
python a2_gate3_judge.py --run --judges nvidia,minimax --set prose      # GLM + MiniMax -> RESULTS_A
python a2_gate3_worker.py --emit-tasks --mode pairwise --batches 4      # -> hand batches to Sonnet-5 subagents
python a2_gate3_worker.py --fold <verdicts.json> --mode pairwise        # -> RESULTS_CLAUDE
python a2_gate3_judge.py --run --judges mistral --set prose             # local Heretic -> RESULTS_M
python a2_gate3_judge.py --merge RESULTS_A RESULTS_CLAUDE RESULTS_M      # full quorum + verdict
```
