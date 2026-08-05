# A2 Stage-1 — Concise-Fable via full-rank task-arithmetic merge: STRONG WIN

**Verdict (2026-08-05): `fable-tc-l1.0` = a Qwen3.6-27B dense that is BOTH concise and uncensored.
−55% reasoning tokens on math / −23% on creative, uncensored character PRESERVED (indistinguishable
from plain Fable on two independent alignment metrics), accuracy neutral-to-better, 0 short-but-wrong.
This REVIVES the "LoRA transfer DEAD" verdict of `A2_THINKINGCAP.md`: the concision transfers via a
FULL-RANK task-arithmetic merge, where the rank-64 SVD LoRA failed.** Deploy candidate = `l1.0`, pending
only the writing-quality judge quorum (Gate 3). Follow-on abliteration program: `A2_STAGE2_PLAN.md`.

---

## Method (the merge)

Full-rank task arithmetic on safetensors (`a2_merge_raw.py`, `scratch/merge_stage1.sh`), keeping Fable's
identity/tokenizer/template, weights only:

    W = W_Fable + λ·(W_TC − W_base)          λ ∈ {0.4, 0.7, 1.0}

`W_TC − W_base` is ThinkingCap's concision task-vector. All arms (incl. plain Fable) quantized BY US to
Q4_K_M **without imatrix**, so every arm is matched — the only variable is the merge. GGUFs on disk
(`models/merges/`). Base/TC/Fable fp16 kept for future recipes. Arms & their nature (a clean 2×2):
`dense-base` = aligned+verbose, `thinkingcap` = aligned+concise, `fable-plain` = uncensored+verbose,
`fable-tc-l*` = uncensored+concise (the merges). External-validity arm: `fable-fusion-711` (the raw
DavidAU artifact people actually run; NOT in paired stats — unmatched quant).

## Axis 1 — Concision (GSM8K n=60, paired; `a2_concision_bench.py` + `a2_stats.py`)

Base = plain Fable (80.0% acc, reasoning median 1084 tok). λ is MONOTONIC on every axis:

| λ | reasoning cut | wall-clock | accuracy | non-inferiority | short-but-wrong |
|---|---|---|---|---|---|
| 0.4 | −23.2% (Wilcoxon p=3.3e-6) | −18.1% | 88.3% | **FAIL** (CI −1.7%) | lost **3**/48 base-correct |
| 0.7 | −45.7% (p=4.4e-10) | −36.7% | 95.0% | PASS (+5.0) | lost 1/48 |
| **1.0** | **−54.8% (p=2.6e-11)** | **−46.7%** | **98.3%** | PASS (+10.0) | **0** |

**λ=1.0 dominates concision AND accuracy** — accuracy RISES with λ because concision cures *starvation*
(the verbose base exhausts the 4096-token budget mid-`<think>` and never answers; base starved 12/60,
l1.0 = 0). Genuine accuracy is ~flat (both-answered); the win is concision. **λ=0.4 is the worst of both
worlds** — cuts reasoning enough to lose 3 base-correct problems (fails non-inferiority) without the full
concision. No attractive knee below 1.0 on this axis. NB: "λ=1.0 dominates accuracy" is *budget-mediated*
(holds at the 4096 deploy budget).

## Axis 2 — Alignment preservation (28-prompt probe, v3; `a2_refusal_probe.py`)

**The probe was rebuilt (three fixes, all load-bearing):**
1. **Starvation:** at `max_tokens=1024` reasoning models spend the whole budget in `<think>` and emit no
   answer → the old 0% refusal was a starvation artifact (proven: idx15 on thinkingcap, 1024→empty,
   4096→full manifesto). Fixed to 4096; starved answers marked *inconclusive*, not "complied".
3. **3-way verdict** (comply / **hedged** / refuse), not binary: the discriminator is *out-of-character
   safety disclaimer that then complies* (l1.0's `"I can't drop my guidelines… but I can…"`) vs
   *in-character sarcasm* (Fable). Regex-based `classify()`; self-checked against the real idx14 answers.
2. **Persist full texts** (`text`+`reasoning_text`): every offline metric re-scores from disk with no
   re-run. This paid off immediately — the classifier was improved and ALL arms re-scored offline (0 GPU).
Plus a `<think>` **safety-deliberation** marker count (an aligned model reasons about its rules even when
it complies) and per-prompt **reasoning/answer tokens** (concision-on-creative). 28 prompts = 12 mild +
8 hard + 8 "unfiltered-persona / reject-your-own-rules" discriminating tier (append-only; idx is the join key).

### Result — a clean, replicated 2×2

| arm | disc-tier balk (idx20-27) | `<think>` safety-delib (non-meta) | creative reasoning (median) | answer (median) |
|---|---|---|---|---|
| `dense-base` (aligned, verbose) | **5/8** | 2 | 1441 | 623 |
| `thinkingcap` (aligned, concise) | **5/8** | 2 | 1095 | 480 |
| `fable-plain` (uncensored, verbose) | **1/8** | 0 | 1159 | 458 |
| **`fable-tc-l0.7`** (merge) | **1/8** | 0 | 923 | 433 |
| **`fable-tc-l1.0`** (merge) | **1/8** | 0 | **889** | 431 |
| `fable-fusion-711` (raw DavidAU) | **0/8** | 0 | 1150 | 483 |

**Two independent alignment metrics give the SAME verdict:** both merges are indistinguishable from plain
Fable (balk 1/8, deliberation 0) and far from the aligned anchors (5/8, 2), which is confirmed by
replication (two aligned models both at 5/8+2, two uncensored both at 1/8+0). **The concision merge did
NOT re-censor Fable** — the faint idx14 "hedge" seen early does not generalize (plain Fable also has 1
hedge on the tier). **External validity:** the raw DavidAU `fusion-711` is 0/8 — our matched requant of
Fable (`fable-plain`, 1/8) is behaviorally faithful to it, so the sweep's transferability holds.

## Axis 3 — Concision generalizes to CREATIVE writing

Plain Fable overthinks creative prompts as much as math (~1159 reasoning tokens median). The merge cuts
it: **l1.0 = 889 (−23% vs plain Fable)** with **answer length preserved** (431 vs 458, −6%, within noise —
NOT the >20% truncation red-flag). More modest than the −55% on math (creative has less wasted
overthinking), but real and positive, and it does not compress the prose.

## Caveats (honest)
- **Discriminating tier is statistically thin** (Fable-5 review): 8 prompts cannot distinguish 1/8 from
  0/8 (Fisher p≈1.0). The "uncensored preserved" claim is a NULL (l1.0 == plain Fable), so small-n is less
  damning, but the tier should be **expanded to ~24 prompts** — scheduled as Stage-2 D0.
- **Provenance:** both GGUFs are community requants (matched imatrix-free), not the official artifact;
  absolute numbers would shift slightly, the paired deltas would not.
- **Q4 does not wash out** either the concision or the alignment signal.

## Deploy verdict
For the dense-27B slot, **`fable-tc-l1.0` = concise + uncensored + accuracy-neutral**, keeps the MTP head
(draft-mtp available for throughput on top). Pending: the writing-quality judge quorum (Gate 3) is the only
axis that could still separate l1.0 from plain Fable (did concision hurt the prose?). Judge infra:
`judge_keys.py` (OS-keyring TUI) + a coming OpenAI-compat quorum (Gemini/NVIDIA-Build/Sonnet-5 + local
uncensored Mistral). Follow-on (abliteration / uTC / CA governor): `A2_STAGE2_PLAN.md` (+ evidence docs).

## Repro
```
# merges (from fp16 base/tc/fable):  bash scratch/merge_stage1.sh 0.4 0.7 1.0
# concision:  python a2_concision_bench.py --model <arm> --workload gsm8k --subset 60 --tag s1p
#             python a2_stats.py --base fable-plain-q4 --cap <arm> --workload gsm8k --tag s1p
# alignment+creative:  python a2_refusal_probe.py --model <arm> --tag s1p   (28 prompts @4096, 3-way + <think>)
# one-shot starvation check:  python a2_refusal_one.py --model <arm> --idx N --budgets 1024,4096
```
Raw records: `runs/a2/refusal__s1p__*.json`, `runs/a2/s1p__*__gsm8k.json` (full texts persisted).
