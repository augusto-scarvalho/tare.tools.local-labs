# LAB-SERVE-001 — Realistic Serving Benchmark (bounded pilot)

**Status: PILOT_COMPLETE.** First qualified serving benchmark of the lab's `llama-server` endpoint
via its real OpenAI `/v1/chat/completions` API, driven by upstream `sglang.benchmark.serving`
(backend `vllm-chat`). Shape-discovery, not a significance study (1 repetition/cell). Full identity:
`../identity.json`. Raw upstream JSONL + stdout + argv: `../raw/`. Normalized: `../normalized/`.

## Instrument qualification (LAB-SERVE-QA-001): QUALIFIED
- transportCompatible ✓ (vllm-chat → llama-server /v1/chat/completions; live smoke + bench).
- streamingObserved ✓ (SSE; the model streams `reasoning_content`, which this bench version counts
  toward output tokens — verified in serving.py).
- tokenAccountingSane ✓ — reported vs retokenized output tokens ratio = **1.000** on every cell
  (forced length; exact `fp16/base` tokenizer for the Qwen3.6-27B merge).
- latencyMetricsPlausible ✓ (TTFT/TPOT/ITL consistent across N).
- instrumentQualifiedForPilot ✓.

## Configuration (see identity.json for the full binding)
- **Model:** `fable-tc-l1.0-q4` (Qwen3.6-27B **dense**, thinking), Q4_K_M, fully resident (`-ngl 99`).
  Chosen for an EXACT tokenizer match (`fp16/base`) — the tightest token-accounting validity. The
  original MTP-flip observation was on the 35B **MoE**; re-measuring on the MoE is the follow-up.
- **Engine:** lifecycle fork `068764d92` (fingerprint `b10159-068764d92`). MTP on = `--spec-type
  draft-mtp --spec-draft-n-max 4`; MTP off = identical flags minus those two.
- **Load gen:** `sglang.benchmark.serving` 0.5.16 (NOTE: `sglang.bench_serving` is a **deprecated
  shim** — a discrepancy vs the Wave-B packet, which named the old path). Backend `vllm-chat`.
- **Workload:** random, input 1024 / output 128, `apply-chat-template`, `ignore_eos=True` (FORCED
  length → controlled token budget that isolates the MTP effect). Matrix N∈{1,2,4,8} × MTP{on,off},
  1 rep, warmup 2, N-order shuffled per server config.

## Results — the concurrency × MTP surface
| N | thr on (tok/s) | thr off | **MTP thr gain** | TPOT on (ms) | TPOT off | **TPOT Δ (on−off)** | TTFT p95 on/off (ms) |
|---|---|---|---|---|---|---|---|
| 1 | 46.8 | 34.1 | **+37.2%** | 13.6 | 22.1 | **−8.5** (MTP better) | 1169 / 1038 |
| 2 | 56.9 | 47.5 | +19.8% | 23.2 | 30.5 | −7.3 (MTP better) | 2761 / 1749 |
| 4 | 61.5 | 54.3 | +13.3% | 47.6 | 44.8 | **+2.8** (MTP worse) | 4214 / 4212 |
| 8 | 59.2 | 49.6 | +19.4% | 52.4 | 45.3 | **+7.1** (MTP worse) | 12741 / 17333 |

Signatures: **ITL median ≈ 0.01 ms for MTP-on vs 22–40 ms for MTP-off** — the expected speculative
signature (accepted draft tokens arrive in one burst). VRAM peak on ≈ 21.0 GB vs off ≈ 18.3 GB
(draft costs ~2.7 GB). Mean GPU power 256–294 W (on) vs 307–323 W (off). All cells: success 100%,
token-sane. `accept_length` reads 0 — that field is sglang-specific; **llama-server does not expose
draft acceptance through the path bench reads** (a documented instrument limitation, not "no MTP").

## Interpretation (shape, not significance)
Re-measuring the prior hypothesis *"native MTP wins at low concurrency but flips around N≈4"*:
- **On aggregate output throughput, MTP wins in every measured regime** (N=1..8) — no throughput
  flip in this dense-27B pilot.
- **On per-token latency (TPOT), MTP FLIPS sign near N=4**: it is clearly better at N=1–2
  (−7 to −8.5 ms) and becomes worse at N=4–8 (+2.8 to +7.1 ms), as draft-verify compute competes
  with batching. So the "flip" is real but lives on the **latency axis**, around N=4.
- Interactive latency degrades steeply for both arms (TTFT p95 → 12–17 s at N=8); the interactive
  envelope for this model/box is roughly **N ≤ 2–4**.

MTP verdict (per §17 language): **wins in the measured low-concurrency regime (N≤2, throughput and
latency); on latency it loses in the measured high-concurrency regime (N≥4); on throughput no
detectable loss up to N=8.** Insufficient evidence for a universal threshold — 1 rep/cell, dense-27B,
forced length. Do NOT encode N=4 as a fixed threshold.

## Caveats / stop-condition notes
- 1 repetition per cell → variance unknown; the numbers are a shape, not a ranking. Next: repeat the
  N=1 and N=4 cells to establish per-cell variance before any promotion use.
- Dense 27B (not the MoE deploy); forced output length (not realistic-EOS). Both are deliberate,
  documented pilot choices (tokenizer-exactness + effect-isolation).
- No stop condition triggered: all requests succeeded, token accounting sane on every cell, VRAM
  within envelope (peak ≤ 21.2 GB < 24 GB), identities bound.

## Reproduce
```
python lmctl.py serve deploy-fable                      # MTP on   (or: serve fable-tc-l1.0-q4 -- -ngl 99  for MTP off)
/home/augus/sglang-venv/bin/python ops/lab_serve_bench.py --tag <t> --outdir runs/serving/LAB-SERVE-001/raw \
    --concurrency <N> --input-len 1024 --output-len 128 --num-prompts <8N> --warmup 2
python ops/lab_serve_normalize.py
python lmctl.py stop --port 8080
```
