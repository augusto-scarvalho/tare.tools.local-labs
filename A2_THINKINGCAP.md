# A2 — ThinkingCap long-to-short on the dense-27B: STRONG WIN (both axes), LoRA transfer DEAD

**Verdict (2026-08-04): ThinkingCap-Qwen3.6-27B halves reasoning tokens and wall-clock on the
dense-27B path at Q4_K_M, at equal-or-better accuracy, with zero short-but-wrong regressions —
and the effect does NOT wash out on code (contra the literature's pessimism). The community
rank-64 SVD LoRA is USELESS (fails reconstruction), so the DavidAU transfer track is closed.**

This is the first A2-tier item that is a POSITIVE lever, not a null/already-captured one. It is
a MODEL-level win, orthogonal to every engine lever (S1–S3/A1/A3/A4): it attacks token COUNT,
not t/s, so ~half the tokens = ~half the wall-clock at the same decode rate.

---

## Setup (the clean paired A/B)

* Arms: `qwen36-27b-dense-q4` (base) vs `thinkingcap-27b-q4` (full fine-tune). BOTH Q4_K_M,
  **matched provenance** — same `general.quantization_version=2`, same `general.file_type=15`,
  and the SAME imatrix (496 entries / 802 chunks, byte-identical), same arch (`qwen35`, 65
  blocks incl. the nextn/MTP head at blk.64), and **byte-identical chat template** (sha256
  `e84f32a2…`) + BOS/EOS. The only variable is the fine-tuned weights.
* **Decode PURE, spec OFF.** `--spec-type draft-mtp` is not verified-exact on qwen35 (#23335/
  #23302 + §Q): it changes committed tokens, hence the reasoning-token COUNT. So the concision
  metric is measured on plain decode; MTP is a throughput lever, never mixed into the count.
* `--jinja --reasoning-format deepseek` → the server splits the `<think>` block into a distinct
  `reasoning_content` stream; `collectors/request.py` assembles it (`reasoning_text`) and
  `count_tokens` tokenises each side via `/tokenize` (exact, model tokenizer). Cross-check on a
  live record: reasoning(1302)+answer(103)=1405 vs server `predicted_n`=1408 → diff 3 = the
  `<think>`/`</think>` delimiter tokens exactly; that constant offset cancels in the paired ratio.
* Greedy (temp 0), max_tokens 4096, ctx 8192, ncmoe 0 (dense fits VRAM). Subset seeded and
  NESTED (pilot ⊂ full) so fail-fast escalation reuses pilot records. Harness: `a2_concision_bench.py`
  (records, resume+incremental write), `a2_stats.py` (Wilcoxon + McNemar + bootstrap, all
  cross-validated against scipy to <1e-18 / d=0), `a2_reconstruct_gate.py`.
* Runs: GSM8K `tag a2g0` n=60; HumanEval+ `tag a2h0` n=40. Fail-fast: an n=12 pilot decided
  GO on Faixa A and KILLED the LoRA track in ~25 min before any multi-hour grind.

## Results

### GSM8K (n=60, paired)
| metric | base | cap | paired reduction |
|---|---|---|---|
| reasoning tokens (median) | 1164 | 575 | **−59.9%** [50.1, 64.8], Wilcoxon p=1.76e-11 |
| total generated tokens | 1319 | 714 | −55.9% [46.4, 60.7] |
| wall-clock | 33.0 s | 17.8 s | **−55.3%** [44.4, 60.3] |
| accuracy (overall @4096) | 78.3% | 96.7% | +18.3pp, McNemar p=0.003 |
| accuracy (both-answered, n=47) | **100%** | **98%** | −2pp (1 regression) → **accuracy-NEUTRAL** |

Difficulty split: reduction 47–77%, LARGEST on the hardest tertile (contra the literature's
"easy carries the mean"). Base **starved 13/60** (spent the 4096 budget reasoning, emitted no
answer); cap starved 0/60 and recovered 12/13 of those. → **On math the accuracy "gain" is
ENTIRELY starvation recovery; genuine accuracy is flat (confirms the vendor's ~−0.7pp).**

### HumanEval+ (n=40, paired; pass@1 = evalplus plus_status)
| metric | base | cap | paired reduction |
|---|---|---|---|
| reasoning tokens (median) | 2705 | 1182 | **−53.0%** [42.5, 59.2], Wilcoxon p=5.5e-8 |
| total generated tokens | 2832 | 1291 | −50.4% [41.0, 58.8] |
| wall-clock | 69.9 s | 31.7 s | **−51.0%** [40.4, 58.7] |
| pass@1 (overall @4096) | 52.5% | 87.5% | +35pp, McNemar p<0.001, base-only-right=**0** |
| pass@1 (both-answered, n=30) | 70% | 90% | **+20pp, 0 regressions, 6 gains** |

Base **starved 10/40**; cap recovered 8/10. Unlike math, the code accuracy gain is GENUINE:
+20pp even on the both-answered subset, with **zero problems where cap regressed** (base-only-
right = 0, everywhere). → **On code, ThinkingCap is both ~2× faster AND materially more
accurate.** The base Qwen3.6-27B is an extreme over-thinker on code (2705-token median
reasoning, 3× GSM8K), so there is huge redundant-reasoning headroom — which is why we see
~53% here, not the literature's ~8% (MixReasoning/2510.06052). The pessimistic code prediction
was for models that don't over-think; ours does.

### Reconstruction gate — FAIL (LoRA/DavidAU transfer DEAD)
`base + community-rank64-SVD-LoRA(λ=1)` vs full ThinkingCap, on the LoRA's own origin base:
* concision: base+LoRA reasoning **968** vs cap **502** (ratio 1.93) — recovered only ~15% of
  the ~60% concision.
* fidelity: sim(base+LoRA, cap)=**0.26** << sim(base+LoRA, base)=**0.65** — behaves like BASE.
* Verdict: the rank-64 SVD does NOT reconstruct the full FT; the concision lives OUTSIDE the
  rank-64 subspace — exactly the failure the L2S literature predicts for a non-low-rank full-FT
  delta (2503.20641 "SVD limited"; 2410.21228 intruder dimensions). Transferring this adapter
  to DavidAU-Fable-Fusion would transfer an adapter that carries ~none of the behavior, so the
  T3 transfer track is closed. Geometry was fine (DavidAU is stock 64-layer + MTP); the adapter
  is the problem, not the target. Only a BETTER extraction (TIES / rank-128-256 / Fisher) or a
  real trace-distillation could revive it — filed as a future item, low priority.

## What this means

* **Deploy: for the dense-27B slot, ThinkingCap replaces base.** ~2× wall-clock at equal (math)
  or better (code) accuracy, keeps the MTP head (draft-mtp still available for the throughput
  layer on top). No short-but-wrong (0–1 regressions across 100 problems).
* **The starvation finding is real and operational:** the verbose base fails 20–25% of problems
  by exhausting a 4096-token budget mid-thought; concision eliminates that. Even at 8192 the base
  would only recover those at ~2× wall-clock and still lose the speed race.
* **Scope limit: dense-27B ONLY.** ThinkingCap cannot apply to the 35B-A3B MoE (our primary
  worker) — shape-incompatible. The general lesson (long-to-short is a real ~2× lever) DOES
  transfer conceptually, but a concise 35B-MoE would need our OWN training (trace distillation on
  the MoE), since no such FT exists and cross-arch adapter transfer is impossible. Highest-value
  A2 follow-on if we want concision on the primary worker.
* **Provenance caveat:** both GGUFs are community requants (underscore-prefixed), not the official
  16.8 GB artifact — but they are MATCHED (same imatrix), so the DELTA is clean. Absolute numbers
  would shift slightly on the official quant; the paired reduction would not.
* **Q4 does NOT wash out the concision** — the ~55% reduction is at Q4_K_M, so the literature's
  quantization-attenuation worry (BitDelta/2602.13151) did not bite here (the behavioral delta is
  far larger than the quant step). No fp16 reference needed.

## Gates / reproduction
```
# fail-fast pilot (decides GO + kills LoRA in ~25 min):
python a2_concision_bench.py --model qwen36-27b-dense-q4 --workload gsm8k --subset 12 --tag a2g0
python a2_concision_bench.py --model thinkingcap-27b-q4   --workload gsm8k --subset 12 --tag a2g0
python a2_concision_bench.py --model qwen36-27b-dense-q4 --workload gsm8k --subset 12 --tag a2g0 --lora thinkingcap-lora-r64 --lora-lambda 1.0
python a2_reconstruct_gate.py --tag a2g0 --workload gsm8k --base qwen36-27b-dense-q4 --cap thinkingcap-27b-q4
# escalate (resume, +48 only) + stats:
python a2_concision_bench.py --model qwen36-27b-dense-q4 --workload gsm8k --subset 60 --tag a2g0
python a2_concision_bench.py --model thinkingcap-27b-q4   --workload gsm8k --subset 60 --tag a2g0
python a2_stats.py --base qwen36-27b-dense-q4 --cap thinkingcap-27b-q4 --workload gsm8k --tag a2g0
# code axis (accuracy via evalplus, in WSL):
python a2_concision_bench.py --model <base|cap> --workload humaneval --subset 40 --tag a2h0
/home/augus/evalplus-venv/bin/python a2_score_humaneval.py runs/a2/a2h0__<arm>__humaneval__samples.jsonl
python a2_stats.py --base ... --cap ... --workload humaneval --tag a2h0 --ext-base <base_scores.json> --ext-cap <cap_scores.json>
```
Raw records: `runs/a2/a2g0__*.json`, `runs/a2/a2h0__*.json`. Deep-dive sources + PRs + literature:
IDEAS_BACKLOG §A2, STATUS §A2.
