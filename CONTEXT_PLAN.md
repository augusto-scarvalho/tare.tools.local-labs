# CONTEXT PLAN — pushing context length per model × lever (draft 2026-08-02)

The question: **how far can we push usable context on the 3090 (24 GB) + 64 GB RAM, per model, per
lever, and what does each extra token cost in speed and quality?** Context is fundamentally a **KV-cache
budget** problem: KV size (VRAM or RAM) is the wall. This plan deduces the frontier from real geometry,
then verifies it, then maps the speed/quality tradeoffs, then prospects external solutions.

## 0. The KV-budget model (deduced from GGUF geometry)

> **CRITICAL CORRECTION (research, 2026-08-02): both Qwen3.6 models are HYBRID Gated-DeltaNet +
> Gated-Attention** (~3:1 linear:full). Only ~**1 in 4 layers holds a GROWING KV cache**; the DeltaNet
> layers carry a **fixed-size recurrent state**, not KV. So the KV table below (computed as if all layers
> had KV) is a **~4× OVERESTIMATE** — real KV is far cheaper (community: only **~800 MB from 4k→64k** on
> Qwen3.5-35B-A3B). **The binding constraint for context is model weights + `--n-cpu-moe`, NOT KV.** Three
> consequences that rewrite this plan:
> 1. **Context is cheap** — likely reach 128k–256k in VRAM without heroics; §B2b (KV-in-RAM) may be
>    *unnecessary* for these models (its value is for large-KV *pure transformers*; revisit accordingly).
> 2. **q4 KV is likely near-lossless here** — hybrid linear layers "absorb" KV-quant noise (q4_0 measured
>    token-identical BLEU 1.000 on Qwen3.5-9B, #21385). The quality axis may show q4==q8; big if true.
> 3. **The 27B "dense" IS this hybrid** — the "fused Gated Delta Net disabled" warning we saw (§E4) is
>    exactly its DeltaNet kernel fallback. Needs recent llama.cpp for the GatedDeltaNet ops (we have it).
>
> Phase A now VERIFIES the real KV growth (load at 64k/131k/262k, measure KV VRAM) rather than trusting
> the arithmetic. The table stands as a *pessimistic upper bound*.

Both Qwen3.6 models are **natively 256k** (`n_ctx_train=262144`, `rope_freq_base=1e7`, YaRN-extendable to
**~1M at factor 4.0**) — so **no RoPE/YaRN needed ≤256k** (static YaRN degrades short-context quality;
enable only above 256k, and mind the server ctx-cap bug #22140 → `--override-kv qwen35moe.context_length`).
KV per token, AS IF every layer had KV (upper bound; real ≈ ¼ of this on the hybrid stack):

| Model | layers | n_head_kv | KV q8_0 | KV q4_0 | KV f16 |
|---|---:|---:|---:|---:|---:|
| **MoE 35B-A3B** | 40 | 2 (8:1 GQA) | 42.5 MB/1k | 22.5 MB/1k | 80 MB/1k |
| **Dense 27B** | 65 | 4 | 135 MB/1k | 71 MB/1k | 254 MB/1k |

**VRAM budget for KV** ≈ 23.2 GB (24 − DWM) − 4 GB guard − model-on-GPU − MTP-draft(~1.15 GB if on).
Model-on-GPU (MoE) ≈ 3.8 + 0.46·(40−ncmoe) GB [from §B3]. Dense (Q4 27B, ngl=65) ≈ 17 GB.

**MoE 35B-A3B — max context that fits in VRAM (q8 / q4), and the decode it costs:**

| ncmoe | model GPU | KV budget | max ctx q8 | max ctx q4 | decode (§B3) |
|---:|---:|---:|---:|---:|---:|
| 8 (deploy) | 18.5 GB | ~0.7 GB | ~16k | ~31k | ~98 t/s |
| 16 | 14.8 GB | ~3.3 GB | ~76k | ~144k | ~75 t/s |
| **24** | 11.2 GB | ~6.9 GB | ~161k | **~300k (>256k!)** | ~53 t/s |
| 32 | 7.5 GB | ~10.6 GB | ~248k | ~466k | ~40 t/s |
| 40 | 3.8 GB | ~14.3 GB | ~335k | ~632k | ~31 t/s |

**Headline (to verify):** the MoE's **full native 256k fits in VRAM at ncmoe=24 + q4 KV, ~53 t/s** — no
KV-in-RAM needed. Context on the MoE is bought with *placement* (ncmoe↑ frees VRAM, decode↓) and *KV
quant* (q8→q4 doubles ctx). KV-in-RAM (§B2b) is the fallback for >256k or to keep a low ncmoe.

**Dense 27B — KV is 6× heavier; the model alone is ~17 GB:**
- KV budget at ngl=65 ≈ 2 GB → only **~15k (q8) / ~28k (q4)** in VRAM. Beyond that needs **layer offload
  (ngl↓, dense decode tanks) or KV-in-RAM (§B2b)**. Dense 128k q8 = 17 GB KV alone → **KV-in-RAM is the
  main long-context path for the dense model.** This is exactly the regime §B2b was built for.

(Numbers are first-order; compute buffers + fragmentation eat extra — Phase A measures the true frontier.)

## Phase A — map the real VRAM frontier
For each (model, ncmoe/ngl, kv-format), find the TRUE max context that loads + serves (not just the KV
arithmetic — the compute/graph buffers and the guard's 4 GB reserve bite). Method: load at a target ctx,
check it serves a short decode without breaching the envelope; binary-search the ceiling. Deliver: the
corrected max-ctx table + where each config's wall actually is. Reuse the guard/envelope machinery.

## Phase B — context × decode-speed (the lever tradeoff)
At fixed target contexts {32k, 128k, 256k}, measure decode t/s across the placement/KV levers:
- **KV-on-GPU** (raise ncmoe until KV fits) vs **KV-in-RAM + §B2b** (low ncmoe, KV in 64 GB RAM). Which
  gives better decode at each context? §B2a showed KV-in-RAM is −70%+ and *grows with depth*; §B2b
  recovers up to +17%. This phase is where §B2b earns or loses its place for long context.
- **MTP at depth**: draft context costs ~1.15 GB VRAM (−max-ctx), and #23658 reports MTP accept collapses
  at KV-slot boundaries at high ctx — measure MTP accept vs context, decide if MTP stays on past ~32k.
- Decode-at-depth curve: even KV-on-GPU, decode slows as KV grows (more KV read per token) — quantify.

## Phase C — context × quality (does long context actually work?)
Two independent quality risks:
1. **KV-quant error at depth**: q4 KV may degrade *more* at long context (error accumulates). Test q8 vs
   q4 (vs f16) at {8k, 32k, 128k, 256k}.
2. **Effective vs advertised context**: 256k native (≈1M YaRN) ≠ usable. Research anchor (Qwen3 RULER):
   holds ~64–128k then declines — so **expect usable ≈ 64–128k, not 256k**. And the hybrid q4-lossless
   claim (#21385) predicts **q4 KV ≈ q8 quality even at depth** — a headline to confirm/refute.
- **Metric — two probes, because NIAH alone lies**: (a) **needle-in-a-haystack / passkey** (inject a key
  at depth d in filler of length L, exact-match over an (L,d) grid) — cheap, but SATURATES near-perfect
  and hides the real limit; (b) a **multi-hop / aggregation** probe (RULER-style: e.g. "find all values
  matching X and sum", or variable-tracking) — this is what actually degrades and reveals the *usable*
  ceiling. temp=0 deterministic. Build a small harness on the serve+request machinery. Sweep (L ∈ {8k,
  32k, 64k, 128k, 256k}) × (KV ∈ {q8_0, q4_0, iq4_nl}) → the **usable-context × KV-format** frontier.

## Phase D — external solutions (research done 2026-08-02)

**Available in llama.cpp today (use these first):**
- **Symmetric KV quant** `--cache-type-k/v` ∈ {f16,bf16,q8_0,q5_1,q5_0,q4_1,iq4_nl,q4_0}, `-fa` required.
  Rule: quantize **symmetrically** — asymmetric K/V falls back to a CPU KV buffer and *tanks* throughput
  (#20866). q8/q8 ≈ lossless ½; q4/q4 ≈ 0.28× (and likely near-lossless on this hybrid arch). `iq4_nl`
  is a better 4-bit than q4_0 at same size — worth a quality datapoint.
- **`--n-cpu-moe`** = the VRAM→context knob (raise it to buy KV room; costs decode). Tuning order: set
  target ctx first, then raise ncmoe until it fits.
- **YaRN** (`--rope-scaling yarn --rope-scale <t/262144> --yarn-orig-ctx 262144` + the `--override-kv
  <arch>.context_length` cap workaround) — ONLY for >256k; static YaRN hurts short prompts.
- **`--swa-full` OFF, `--kv-unified` default** (min VRAM); **`--context-shift`** (off by default) for
  rotating "infinite" generation — but it discards the middle, so wrong for long-doc reasoning.
- **`-nkvo` / §B2b** — keep as the fallback, but **re-evaluate necessity**: with hybrid-small KV, VRAM
  may hold 128k+ so we can spend the budget on a *lower* ncmoe (faster decode) instead of RAM-KV.

**Not merged but promising for THIS card (watch/test):**
- **TurboQuant** (disc. #20969): `tq3_0` ~3.25-bit / `tq4_0` KV, Hadamard+Lloyd-Max; ~4.9× vs f16, ±2%
  PPL, claims 262k fits in VRAM. **CUDA/3090 forks exist.** This is the KV codec we EXCLUDED from the
  fork for decode — reconsider it **strictly as a context lever** (2–3 bit KV → 2–4× more ctx). Recipe
  seen: TurboQuant-K + q8-V.
- **Per-head adaptive KV quant** (#21385, not planned): the source of the "q4 lossless on hybrid" datum.

**Strong papers NOT in llama.cpp (would need engine surgery — do not invest unless forking):** QUEST,
SnapKV, PyramidKV, Ada-KV, H2O, Scissorhands (eviction/sparsity); KIVI, KVQuant (per-channel-K 2-bit).
GGML's unified contiguous block-quant KV fights all of these — only uniform per-tensor quant + arch-native
SWA actually ship. PagedAttention/ring-attention = vLLM/multi-GPU, N/A here.

## The deliverable — a context operating-point table
A decision matrix: **want context X at quality ≥ Q → config C (model, ncmoe/ngl, kv-format, KV placement,
MTP on/off) → decode D t/s, load L**. So a target (e.g., "128k agentic at usable quality") maps to one
concrete serve command and its speed. This is the context-axis analogue of DEPLOY.md's decode stack.

## Sequencing
Runs AFTER the levers×quality matrix (in flight). Phase A is cheap (loads only). Phase C's NIAH needs a
harness but is cheap per-point. Phase B is the expensive one (decode at 128k/256k is slow). Prioritize:
A (frontier) → C-quant (does q4 hold at depth) → B (speed tradeoff) → D (only if native 256k is
insufficient). Everything measured on the `lifecycle` fork so the levers actually engage.
