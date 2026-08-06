# Track M-A · VLM (vision) — M0 baseline DONE (2026-08-06)

A coding-agent-that-sees: read screenshots, error dialogs, UI mockups. M0 = get a current-gen
VLM serving in our fork and prove image→text on a controlled accept test. **PASS on all three
models tried.** Evidence: `runs/m0-vlm/` (fixtures + RESULTS.md); reusable probe `vlm_probe.py`.

## What made it a short hop
`libmtmd` is built in the deploy fork and `llama-server` speaks vision natively (`--mmproj`,
`--image-min/max-tokens`, `--mmproj-offload`). Crucially, the mtmd in our build is **far newer
than the pinned base commit suggests** — `tools/mtmd/clip.cpp` lists `PROJECTOR_TYPE_QWEN3VL`,
`GEMMA4V`, `GLM4V`, `MINICPMV4_6`, plus OCR specialists (`DOTS_OCR`, `PADDLEOCR`, `DEEPSEEKOCR`,
`LIGHTONOCR`). So current-gen VLMs run without touching the pin. (First-pass picks were the stale
Qwen2.5-VL / Gemma-3; the user flagged it, we upgraded to Qwen3-VL + Gemma-4.)

## Models (all Q4, all fit 24 GB, served via `lmctl serve <name>` on :8092)
| profile | model | VRAM used | OCR (error dialog) | UI mockup | decode feel |
|---|---|---|---|---|---|
| `qwen3-vl-30b` | Qwen3-VL-30B-A3B UD-Q4_K_XL + mmproj-F16 | **20.5 GB** (3.8 GB free — *tight*, hair under the 4 GB reserve @8k) | perfect, 0.7 s | complete, 0.8 s | fastest (MoE ~3B active) |
| `qwen3-vl-8b` | Qwen3-VL-8B UD-Q4_K_XL + mmproj-F16 | **8.9 GB** (comfortable) | perfect, 0.6 s | most spatially detailed, 2.9 s | fast |
| `gemma-4-12b-vision` | Gemma-4-12B-it Q4_0 + mmproj-Q8_0 | **10.5 GB** | correct, 6.3 s | correct but **missed the headings**, 9.0 s | slower (thinking) |

Accept fixtures (known ground truth, so OCR is pass/fail not vibes): a Windows-style error dialog
(`Application Error` / `NullReferenceException at PaymentService.Charge(order=4471) line 132.` /
Retry·Cancel) and a login mockup (`Sign in to Acme`, Email `you@example.com`, Password, Forgot
password?, Continue, Sign up). Both Qwen models transcribed every line exactly; Gemma too once given
token headroom.

## Gotcha banked: thinking-VLM token budget
Gemma-4 is a **thinking model** — it spends the completion budget on `reasoning_content` FIRST, then
emits `content`. At `max_tokens=512` with an elaborate prompt, reasoning ran to the cap and `content`
came back EMPTY (finish_reason would be length). Fix in `vlm_probe.py`: default `max_tokens=1024` +
fall back to showing `reasoning_content` when `content` is empty. Same class as the Gate 3 Gemma-judge
gotcha (`A2` memory). Qwen3-VL does not think as hard, so it was unaffected.

## Verdict / recommendation on THIS box
- **Daily driver: `qwen3-vl-8b`.** Perfect OCR, richest spatial description, 8.9 GB leaves ~15 GB for a
  text worker or a big KV — you can co-resident it. Best ROI.
- **Quality ceiling: `qwen3-vl-30b`.** Fastest *and* accurate (MoE), but 3.8 GB free @8k is under the
  envelope reserve — drop `--ctx-size` or run it as the sole model. Not co-resident-friendly.
- **Cross-family: `gemma-4-12b-vision`.** Works, Google lineage, but slower (thinking) and needs the
  token-budget headroom; missed the mockup headings once.

## §71 next levers (only after a repeated-image workload exists — NOT now)
Visual-embedding cache by content-hash; separate mmproj VRAM budget from text KV (don't let the encoder
starve decode); offload the vision encoder when a session goes text-only; measure visual TTFT vs text
TTFT separately (vision-encode is compute-bound, our decode is bandwidth-bound → schedule apart).
