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

## Alignment + real perf added 2026-08-06 (the two M0 gaps the user flagged)
M0 picked 8B on a thin 2-image smoke test and never touched refusal or a real benchmark. Both closed:

**Alignment / over-refusal (`runs/m-a-align/`, probe `vlm_refusal_probe.py`, fixtures
`gen_refusal_fixtures.ps1`):** benign coding-agent categories (CAPTCHA, `.env` secrets, PII/SSN form,
bank-login UI, license plate, lab dashboard), each with a known ground-truth token so the metric is
objective. **All three models = 0/7 over-refusal** — transcribed secrets incl. passwords/API keys in
the clear, listed the SSN, read the plate, no redaction/refusal. The 2 "miss" are OCR errors on hard
random strings (Gemma worst, matches OCR ranking), not alignment. **→ abliteration is NULL** (same as
A2 Gate-2: fixes a non-problem). Scope caveat: real-photo person-ID untested (not synthesizable, and
out of scope for a coding agent).

**Real perf (`runs/m-a-perf/`, bench `vlm_bench.py`, streamed SSE, 5 reps + warmup, cache off, VRAM
OC +350):** decode t/s — **30B 165 ± 0.9 (fastest, MoE ~3B active) > 8B 127 ± 0.5 > Gemma 82 ± 0.8**;
visual TTFT all sub-250 ms; image-encode tax 103–147 ms (tiny). **The M0 "Gemma slow" was token COUNT
(thinking), not rate.** 8B daily-driver call HOLDS but the real reason is VRAM ergonomics (8.9 GB →
co-resident), not speed — 30B is faster solo. §71 encoder-starves-decode worry = non-issue at this
image size (would only bite with many/large/tiled images).

**Quality on MMStar (`runs/m-a-vqa/`, bench `vlm_vqa_bench.py`, 150 MCQ = 25/cat × 6, exact-match
acc):** accuracy INVERTS the speed order — **Gemma-4-12b 0.573 > 30B 0.520 > 8B 0.487**. Gemma's
per-question thinking (the same thing that makes it 20× slower, 740 s vs 36 s) buys the top accuracy
and leads exactly where thinking helps: **logical reasoning 0.60 vs 8B's 0.36**, + math. Perception is
a near-tie (8B even ties/leads fine-grained). **So: 8B stays the daily driver for OCR/UI-reading
(perception-bound, fast, co-resident); reserve Gemma for hard visual REASONING (charts/diagrams/logic)
when latency is affordable; 30B is the fast middle.** Caveat: n=150 SE ~4 pp → Gemma>8B (8.6 pp) is
~1.5σ (fairly clear), Gemma vs 30B (5.3 pp) within noise; Q4 + simple MCQ parse undershoots official
scores but is fair apples-to-apples across our deployed builds. Rerun ≥100/cat to firm the ranking.

**NSFW / mature-content (asked 2026-08-06):** the coding-agent refusal probe found 0/7 over-refusal
but did NOT test the sexual axis (I won't source/generate explicit imagery or write explicit
descriptions — held even for testing). Options recorded: (a) a MATURE-but-non-explicit refusal probe
(gore/violence + public-domain artistic nudity) extends `vlm_refusal_probe.py` to measure the mature
axis without pornography; (b) for actual explicit material the harness is fully OFFLINE — user drops
their own fixtures in `runs/m-a-align/` and runs the same probe, nothing leaves the box. Base instruct
VLMs sanitize/refuse explicit sexual acts but describe artistic/medical nudity; abliterated VLM
variants exist on HF (Arditi method) if full uncensored description is the goal.

## §71 next levers (only after a repeated-image workload exists — NOT now)
Visual-embedding cache by content-hash; separate mmproj VRAM budget from text KV (don't let the encoder
starve decode); offload the vision encoder when a session goes text-only; measure visual TTFT vs text
TTFT separately (vision-encode is compute-bound, our decode is bandwidth-bound → schedule apart).
NOTE 2026-08-06: the visual-vs-text TTFT split is now MEASURED (tax 103–147 ms, tiny) — the remaining
levers stay deferred until a repeated-image workload actually exists. STILL UNTESTED: the OCR
specialists the build supports (DOTS_OCR/PADDLEOCR/DEEPSEEKOCR/LIGHTONOCR) — an untested model class,
worth a look only if a heavy-OCR workload appears (Qwen3-VL OCR is already near-perfect for us).
