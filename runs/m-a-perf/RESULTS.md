# M-A VLM real perf benchmark — 2026-08-06

Fills the M0 gap: M0 had only single-request latency-feel on 2 images (no reps, no stats, never
separated vision-encode from decode). Harness `vlm_bench.py`: STREAMED SSE so TTFT and decode t/s
are timestamped directly; 5 reps + warmup discarded; unique nonce/rep (cache_prompt=False, A4
discipline). Image = `bank_login.png` (460×420, typical agent screenshot). Text mode = same prompt,
no image → TTFT delta = image-encode tax. VRAM OC at +350 (user set 2026-08-06). max_tokens=200.

| model | decode t/s | visual TTFT | image-encode tax | VRAM | note |
|---|---|---|---|---|---|
| **qwen3-vl-30b** (MoE ~3B active) | **165 ± 0.9** | 0.22 s | 147 ms | 20.5 GB (tight) | fastest decode + best OCR; not co-resident |
| **qwen3-vl-8b** (dense) | 127 ± 0.5 | 0.17 s | 103 ms | 8.9 GB | co-resident-friendly; **daily-driver** |
| **gemma-4-12b** (dense, thinking) | 82 ± 0.8 | 0.32 s | 134 ms | 10.5 GB | slowest/token AND emits many think tokens |

## Findings
- **The M0 "Gemma feels slow" was token COUNT, not token RATE.** Gemma decodes a healthy 82 t/s;
  its high total latency in M0 was the thinking model emitting far more tokens first. Real benchmark
  corrects the latency-feel impression.
- **30B is the fastest decoder (165 t/s)**, as the MoE ~3B-active geometry predicts — and it also had
  the best OCR in M0. Its only downside is VRAM (20.5 GB → 3.8 GB free @8k, not co-resident-friendly).
- **8B daily-driver call HOLDS, now data-backed:** 127 t/s + 8.9 GB leaves ~15 GB for a text worker
  or big KV. The pick is VRAM ergonomics (co-residence), not raw speed — 30B is faster solo.
- **Image-encode tax is tiny (103–147 ms) on a typical screenshot** → the §71 worry (vision encoder
  starving decode) is a non-issue at this image size; it would only bite with many/large images or a
  high-res tiling path. All visual TTFTs are sub-250 ms = snappy for interactive agent use.

Raw: `PERF_{qwen3-vl-8b,qwen3-vl-30b,gemma-4-12b-vision}.json`.
