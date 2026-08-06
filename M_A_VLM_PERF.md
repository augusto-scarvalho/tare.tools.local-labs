# M-A VLM performance/speed research + empirical validation — 2026-08-06

Question: how to make our vision models faster, ESPECIALLY Gemma-4-12b (the accurate-but-slow one).
Method: 4 parallel Sonnet-5/medium research workers (llama.cpp engine, SGLang/vLLM, acceleration
papers, thinking-budget + spec-decode) over web/GitHub/arXiv, then I VERIFIED the top lever on our
box. Workers' raw findings: workflow `wf_6a622b4a-733`.

## Root cause (re-confirmed): Gemma's cost is THINKING TOKENS, not decode rate
Gemma decodes fine (82 t/s); its 740s-for-150-MMStar wall-clock is reasoning-token VOLUME. So the
levers that matter are (a) control the thinking phase, (b) speed the decode of whatever remains. The
whole visual-token-pruning literature is NULL for us (see dead-ends).

## Key correction: our pin base is 2026-07-25, not "~Jul 2025"
`720d7fa40` = committer date **2026-07-25** ("vendor: update cpp-httplib #26067"); lifecycle HEAD
`068764d9` = 2026-08-04. So every VLM/thinking/MTP PR merged before Jul-2026 is ALREADY in our base +
built binary — VERIFIED via `llama-server --help`: `-rea/--reasoning [on|off|auto]`,
`--reasoning-budget N` (+`--reasoning-budget-message`), and `--spec-type ...,draft-mtp,draft-eagle3,
draft-dflash` all present. **These are config-only, no backport.** (Fix the stale date in the project
memory.)

## MEASURED — Gemma reasoning-budget sweep on MMStar-150 (our box, VRAM OC +350)

| Gemma config | MMStar acc | wall-clock (150) | speedup | verdict |
|---|---|---|---|---|
| baseline (unbounded thinking) | 0.573 | 740 s | 1.0× | the slow default |
| `--reasoning-budget 256` | 0.580 | 528 s | 1.4× | full accuracy, free |
| `--reasoning-budget 128` | 0.520 | 396 s | 1.9× | drops 6pp — below the knee |
| **`--reasoning-budget 256` + MTP draft** | **0.580** | **276 s** | **2.7×** | **STACKED — full accuracy, new profile default** |
| `--reasoning off` | 0.480 | 80 s | 9.3× | 8B-tier accuracy (Qwen-8b dominates here) |

**Reads:** 256 thinking tokens SATURATE MMStar accuracy (0.580 ≈ 0.573 unbounded, within n=150 noise)
while cutting 1.4× — a free win, now baked into the `gemma-4-12b-vision` serve profile. The knee is
between 128 and 256. Turning thinking OFF is 9.3× faster but collapses Gemma to 0.480 ≈ Qwen3-VL-8B's
0.487 — and there Qwen-8b is faster AND fully candid, so **reasoning-off Gemma is strictly dominated;
use Qwen-8b for the fast path and Gemma (budget-capped) only when you want the accuracy crown.**

## Ranked levers

### #1 — `--reasoning-budget 256` on Gemma [DONE, config-only, MEASURED]
Full accuracy at 1.4× faster, zero cost, already in the profile. Tune the budget higher for harder
visual-reasoning than MCQ. `--reasoning off` / `--reasoning-budget 0` available for a fast/dumb mode
(but Qwen-8b beats it). Evidence: llama.cpp `common/reasoning-budget.cpp`, PR #21697 (Gemma4 think-tag
wiring, merged 2026-04-10 → in base), discussion #21338/#21445; s1 "budget forcing" arXiv:2501.19393.

### #2 — Gemma-4's own MTP draft head via `--spec-type draft-mtp` [DONE, MEASURED, in profile]
Gemma-4 ships a trained MTP "assistant" drafter (the head our notes flagged as unwired). PRs #22673
(MTP infra) + #23398 (Gemma4 MTP) are in our base; #19493 removed the "spec-decode not supported with
multimodal" guard → runs WITH `--mmproj`. **Downloaded** the matching drafter (`Janvitos/gemma-4-12B-it
-qat-assistant-MTP-Q8_0-GGUF`, 465 MB, to the model dir) and added `--spec-type draft-mtp --model-draft
<assistant.gguf> --spec-draft-n-max 4` to the serve. **MEASURED: ~0.68 draft acceptance (mean len
~3.8), 0.580 @ 276 s = 1.9× over budget-256 alone (528 s), 2.7× over the 740 s baseline, at IDENTICAL
accuracy** (spec-decode is lossless at temp 0). The QAT drafter matches our Q4_0 QAT target (high accept
confirms it). Our gemma vision GGUF has NO built-in MTP layers (verified: draft-mtp alone errors "model
doesn't contain MTP layers") — the separate `--model-draft` assistant is required. KV is default **f16**
here so the old q8_0-KV+MTP 0%-accept bug does NOT apply. Do NOT stack ngram with draft-mtp (#24266:
40→4 t/s collapse). Now the `gemma-4-12b-vision` profile default. Evidence: PRs #19493/#22673/#23398,
`huggingface.co/Janvitos/gemma-4-12B-it-qat-assistant-MTP-Q8_0-GGUF`, `ai.google.dev/gemma/docs/mtp`.

### #3 — image-token budget hygiene on Gemma [minor, already safe]
`--image-min/max-tokens` control the vision-token budget; Gemma-4 needs all image tokens in one ubatch
or it SIGABRTs at high budgets — we're already safe (`--ubatch 2048`). For MCQ, 280–560 image-tokens
suffice; 1120 only for OCR/docs. Minor vs #1/#2 (image-encode tax already ~130 ms). Evidence: issues
#21550/#21461.

## Dead ends (verified, don't chase)
- **Visual-token pruning papers** (FastV 2403.06764, VTW 2405.05803, PyramidDrop 2410.17247, SparseVLM
  2410.04417, PruMerge 2403.15388, ToMe 2210.09461): built for 576–6000-token high-res/doc inputs; our
  460×420 screenshots emit only ~190–256 visual tokens → pruning saves tens of ms against a
  seconds-to-minutes request. NULL for our workload; none implemented in clip.cpp. Revisit only for 4K/
  multi-tile docs.
- **Switching to SGLang/vLLM** [worker verdict: DEAD END for us]: both now support Qwen3-VL + Gemma-4
  (vLLM ≥0.11 / PR #38826; SGLang PR #10323/#21952), but every feature that differentiates them —
  RadixAttention, continuous batching, chunked-prefill + vision-encoder cache — is a MULTI-REQUEST
  concurrency lever, worthless at our batch=1 single-user load; PagedAttention gather is a net COST at
  batch=1; prefix-cache we already have (`cache_prompt`). Their Gemma thinking-budget is WORSE (vLLM
  bolt-on w/ xgrammar bug #39130; SGLang budget unenforced #25536) than our first-class
  `--reasoning-budget`. **FP8 is dead on Ampere sm_86** (needs cc≥8.9 → Marlin W8A16 dequant fallback,
  adds latency). AWQ/GPTQ-int4 is Ampere-native but means re-quantizing every model + abandoning the
  tuned fork for an unproven batch=1 delta. Informal RTX-3090 batch=1 numbers put llama.cpp at parity/
  ahead. Flip only on Blackwell/Hopper HW or a real multi-user workload.
- **External-draft spec-decode for the Qwen3-VL-30B MoE**: a public RTX-3090 + A3B-MoE benchmark (19
  configs) found NO net speedup — same Ampere+MoE null we already know. MTP self-draft (dense Gemma) is
  architecturally different and IS worth it; external draft on the MoE is not.
- **EAGLE/Medusa/SpecVLM/ViSpec for VLMs**: paper-only, no llama.cpp/vLLM/SGLang production drafter for
  these models; would require training a head ourselves. `draft-eagle3` exists in our binary but needs
  an eagle3 draft checkpoint that doesn't exist for these VLMs.
- **`reasoning_effort` API field**: no-op beyond 'none' in llama-server; use `--reasoning-budget`/
  `thinking_budget_tokens` instead. `--no-mmproj-offload` = wrong direction (we have VRAM headroom).

## Bottom line
Gemma is now **2.7× faster at IDENTICAL accuracy** (0.580 @ 276 s vs 740 s), both levers measured and
shipped in the `gemma-4-12b-vision` profile: `--reasoning-budget 256` (1.4×) stacked with its own MTP
draft head (1.9×, ~0.68 accept, lossless). Everything else — engine switch to SGLang/vLLM,
visual-token-pruning papers, FP8, external drafters — is a verified dead end on a single Ampere 3090 at
batch=1. Qwen3-VL models are already fast and need nothing. Remaining optional headroom: sweep
`--spec-draft-n-max` (tried 4) and the reasoning budget per-workload; wire an MTP/EAGLE head for Qwen if
one ever ships (none exists today).
