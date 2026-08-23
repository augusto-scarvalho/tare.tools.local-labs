# LAB-CTX-002 — official RULERv1 64k/128k bounded qualification

**Status:** COMPLETE / 64K PILOT FAIL / 128K BOUNDED PASS  
**Date:** 2026-08-22  
**Substrate:** historical Qwen3.8-27B Q4_K_XL, slop.cpp `b9863-5e7f6271c`, q4 KV, MTP n3, one RTX 3090

## Result

The official 13-task RULERv1 pilot produced a non-monotonic result:

| Target | Pilot | Preregistered 85.6% gate | Bounded follow-up panel | Operational result |
|---:|---:|---:|---:|---|
| 65,536 | 82.82% (13 tasks, n=1/task) | **FAIL** | 91.97% after VT/CWE/FWE became n=3 | fragile; 4/19 outputs hit length limit |
| 131,072 | 100.00% (13 tasks, n=1/task) | **PASS** | 100.00% after VT/CWE/FWE became n=3 | 19/19 `stop` |

The follow-up panel deliberately has unequal n: the ten pilot-perfect tasks remain n=1, while the three
non-monotonic tasks are n=3. It is a bounded localization result, not an upstream-comparable leaderboard score.

## Task localization

All eight NIAH variants and both QA tasks passed at both lengths in the pilot. The expanded tasks were:

| Task | 64k n=3 | 128k n=3 | Interpretation |
|---|---:|---:|---|
| Variable tracking | 66.67% | 100.00% | one 64k output exhausted 30 tokens before listing variables |
| Common-word extraction | 40.00% | 100.00% | all three 64k outputs exhausted 120 tokens; two were verbose before providing the full answer |
| Frequent-word extraction | 88.89% | 100.00% | one 64k prediction misspelled one of three identifiers |

The 128k instances are not the same text padded to a greater length: the official generator creates a new
length-conditioned instance. Therefore the reversal does **not** show that 128k is intrinsically easier or better.
It shows strong instance/position sensitivity at 64k and no observed degradation in this bounded 128k sample.

## Method and provenance

- NVIDIA/RULER `rulerv1-ns` `e8bbff677ca2c239640dc90f93310dcf32408c93`;
- NVIDIA-NeMo/Skills `chsieh/ruler-remove-prefix` `f4a3fd8e524acd9abd1fea4387e8f179f6d51cf3`;
- generator/config snapshot `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3`;
- all 13 official task configurations, prompts, answer prefixes, output budgets, and match semantics;
- deterministic endpoint sampling: temperature 0, top-p 1, seed 42, thinking off, prompt cache off;
- Qwen3.5-27B tokenizer matched the GGUF endpoint exactly on 8/8 fixed probes;
- prompts were preflighted through the live chat template and exact tokenizer before inference;
- actual prompts: 57,736–65,445 tokens at 64k and 123,364–130,994 at 128k;
- 38 total receipts; 32 generated dataset manifests and hashes retained in `summary.json`;
- HotpotQA source mirror revision and all corpus hashes are frozen in `PRE_REGISTRATION.md`.

The current NeMo Skills runner supports vLLM/SGLang/TRT-LLM, not llama.cpp. The local adapter changes only
transport, live template preflight, and resumable receipts. This preserves task/scoring semantics but must be
reported as an official-data bounded local run, not as a stock NeMo Skills engine result.

## Artifacts

- `PRE_REGISTRATION.md` — frozen pilot and gate-triggered replication amendment;
- `receipts.jsonl` — 38 raw predictions, references, usage, timings, and prompt hashes; SHA-256
  `d073679031679bf67026e2094a920fb7a993e5864eb00fe9c3ff263e64319dc8`;
- `summary.json` — metrics plus 32 dataset file manifests; SHA-256
  `9bfaee7bc83bd73c0b60a3bebB22ef556750e9c7a05e754297e67589cfbd0cb6`
  (hex is case-insensitive);
- `tools/benchmarks/ruler_local_eval.py` — local endpoint adapter;
- `tools/benchmarks/ruler_hotpot_from_parquet.py` — revision-pinned HotpotQA shape conversion;
- `tools/benchmarks/ruler_summarize.py` — receipt and dataset manifest summarizer;
- `tools/scripts_sh/prepare_ruler_pilot.sh` — official generator launcher with fail-closed file counts.

## Decision

Do not use single-needle success as evidence of broad 64k correctness. Keep 128k enabled for the incumbent:
this bounded panel found no 128k failure and the server remained healthy, but the small, selectively replicated
sample is not sufficient for a publication-grade effective-context claim. For agent use, constrain verbose
aggregation outputs and treat 64k CWE/VT behavior as a known robustness gap.
