# M-A VLM visual-recognition QUALITY benchmark — MMStar — 2026-08-06

M0 judged description quality on 2 hand-made fixtures (vibes). This scores it OBJECTIVELY on
**MMStar** (`Lin-Chen/MMStar`, 1500 MCQ, 6 balanced ability categories, leakage-controlled). Metric
= exact-match accuracy on a single option letter — the visual analog of GSM8K/HumanEval for text.
Harness `vlm_vqa_bench.py` (runs in WSL sglang-venv, hits llama-server :8092). Subset = 25/category
= 150, seed 20260806 (same items for all three). Q4 builds, VRAM OC +350.

## Result — accuracy INVERTS the speed order

| model | MMStar acc | speed | unparsed | verdict |
|---|---|---|---|---|
| **gemma-4-12b-vision** | **0.573** (86/150) | **740 s** (thinking) | 0 | most accurate, slowest |
| **qwen3-vl-30b** (MoE) | 0.520 (78/150) | 47 s | 0 | 2nd, fast |
| **qwen3-vl-8b** (dense) | 0.487 (73/150) | 36 s | 3 | 3rd, fastest per-item |

Per-category (correct/25):

| category | gemma | 30b | 8b |
|---|---|---|---|
| coarse perception | 15 | 14 | 15 |
| fine-grained perception | 11 | 11 | 13 |
| instance reasoning | 19 | 19 | 15 |
| logical reasoning | 15 | 11 | 9 |
| math | 15 | 11 | 13 |
| science & technology | 11 | 12 | 8 |

## Findings
- **The M0 "Gemma is the weak one" impression is WRONG on quality.** Gemma-4's per-question thinking
  (the same thing that makes it 20× slower, 740 s vs 36 s) buys the **highest accuracy**, and it
  leads exactly where thinking should help: **logical reasoning (0.60 vs 8B's 0.36)** and math.
- **Real speed/accuracy tradeoff, now quantified:** Qwen3-VL-8B/30B are 15–20× faster per item but
  trail Gemma by 5–9 pp on MMStar. There is no free lunch — pick by workload.
- **Perception is a near-tie** (coarse ~all 15/25; 8B actually ties/leads fine-grained) → for the
  OCR/UI-reading a coding-agent-that-sees mostly does, **8B remains the right daily driver** (perception
  is what that needs, and it's fast + co-resident). Reserve **Gemma for hard visual REASONING**
  (charts/diagrams/logic) when you can eat the latency; **30B** is the fast middle.

## Honest caveats
- **n=150 → sampling SE ~4 pp.** Gemma > 8B (8.6 pp) is ~1.5σ (suggestive, fairly clear); Gemma vs
  30B (5.3 pp) is within noise. To firm the ranking, rerun at full 1500 or ≥100/cat.
- These are **Q4 quants + a simple MCQ parser**, so absolute numbers undershoot the models' official
  MMStar scores — but it's a fair apples-to-apples comparison of *our three deployed builds*, which is
  what the daily-driver decision needs.

Raw: `MMSTAR_{qwen3-vl-8b,qwen3-vl-30b,gemma-4-12b-vision}.json` (per-item gold/pred/raw persisted →
re-score offline without re-running).
