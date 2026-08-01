# LANDSCAPE — our fork, upstream, forks, and rival engines (curated 2026-08-01)

What the outside world has already built or argued about for our exact problem — **fast decode
of a large MoE whose expert weights are offloaded to system RAM, on one consumer GPU (RTX 3090,
24 GB) + 64 GB RAM**. Curated from three research passes (llama.cpp upstream issues/PRs, SGLang,
and popular forks) plus direct inspection of our fork. Companion to `MECHANISMS.md` (theory),
`STATUS.md` (our results), `EXPERIMENTS.md` (our plans). Citations are live GitHub refs.

---

## 0. The framing that reorders everything: two philosophies for offloaded MoE

- **(a) Stream-to-GPU** — experts sit in RAM, the used experts are copied to the GPU per token
  and **computed on the GPU**. Bottleneck: PCIe transfer. **This is what our fork does**, and our
  entire §B1 / pinning line optimizes *the transfer*.
- **(b) Compute-on-CPU** — experts stay in RAM and are **computed on the CPU** with hand-tuned
  kernels; nothing crosses PCIe per token. Bottleneck: CPU matmul speed. **This is what
  `ik_llama.cpp` and `KTransformers` do.**

We have only explored (a). Which wins depends on the operating point (GPU speed vs PCIe vs CPU
GEMM throughput). **The single biggest untested speed lever is a head-to-head of the two
philosophies on our rig.** Everything below is judged against decode speed in our regime.

---

## 1. Our fork — `thecodacus/llama.cpp` ("Fable"), commit `5e7f6271c`

Exactly **three commits** on upstream `4fc4ec554`, all philosophy-(a), all opt-in, token-identical:

| commit | what | env var |
|---|---|---|
| `20f5994bf` | pin mmap-backed CPU expert weights (register_host) — DMA not bounce-buffer, ~6-7→~20 GB/s | `GGML_CUDA_REGISTER_HOST=1` |
| `1163cb349` | overlap expert uploads with compute on a 2nd CUDA stream | `GGML_SCHED_PREFETCH_EXPERTS=1` |
| `5f83fbbe7` | size prefetch slots per layer + fix a use-after-free | — |

**The fork's own benchmark (README):** RTX **3060**, Qwen3.6-35B-A3B, ncmoe=26, prefill
**~1143 → ~1880 t/s (+64%)** with both switches. That +64% **is the same number as upstream
issue #25859** — the issue is this same work. The fork only ever measured **prefill**; our
generation dose-response (STATUS §B1) and the §B2 KV idea are beyond it. Branches:
`fable5/host-register` (pin only), `fable5/prefetch-experts` (both).

**Why our 3090 numbers differ (reconciled):** pin-alone is ~+120% for us vs ~+21% for the 3060,
and prefetch is a *tax* for us vs the *majority of the win* on the 3060. The discriminator is
**GPU-idle%** during expert H2D: a fast 3090 is transfer-dominated (pinning is the big win, little
idle for overlap to reclaim → prefetch a tax); a slow 3060 leaves 42% GPU idle (overlap reclaims
it → prefetch wins). Both correct, different operating points. → instrument GPU-idle% (our §B3).

---

## 2. Upstream `ggml-org/llama.cpp`

- **Expert placement is already in mainline:** `--n-cpu-moe N` (PR #15077, merged) and
  `-ot/--override-tensor` regex placement. So "experts→CPU" is testable with **zero fork**.
- **[#25859]** offloaded-MoE prefill leaves GPU idle on serial H2D; page-locking alone ≈ +21%,
  second-stream overlap delivers the majority of +64% (RTX 3060). = our fork's work.
- **[#26110]** losing the "no-mmap + pin CPU tensors" combo caused swap thrash **25-27 → 7.55 t/s**
  — independent proof that pinning offloaded weights is load-bearing.
- **[#6206]** (slaren) introduced/gated `GGML_CUDA_REGISTER_HOST`, off by default, needs `--no-mmap`.
- **[#25932] `--pin-hot-experts`** — pin only the hottest experts, not the whole mmap. A cheaper
  generation-side variant of our blanket pin, directly A/B-able (our §B5).
- **MoE follow-ups:** `#20596` improve `--n-cpu-moe` TG; `#26003` `--lazy-experts` (MoE > RAM);
  `#25294` stream experts from disk; `#21609`/#24524/#20757 tiered expert residency caches.
- **MTP / speculative is real and merged:** `--spec-type draft-mtp` (#25980 merged), EAGLE3 for
  gpt-oss / Qwen3.5-3.6 (#25794/#24593 merged). Public target: **+30% tok/s, ~82% accept** (#25642).
- **KV-in-RAM pinning (our §B2): essentially no prior art in llama.cpp.** #21792 (mmap-KV) is about
  persistence, explicitly skips pinning/DMA. Our pinned-KV context-length dose-response looks
  **novel** for this engine (SGLang does a *form* of it, below, so the mechanism is validated).

---

## 3. Rival engines / forks (philosophy-(b) and beyond)

| project | stars | technique relevant to us | adoption cost |
|---|---|---|---|
| **`ikawrakow/ik_llama.cpp`** | ~2,984, active daily | Fast CPU-GEMM + repacked quants (`-rtr`, `_R4`/`_R8`, `IQ*_K`); **`-fmoe`** fused MoE; **`-ser`** smart-expert-reduction (skip low-prob experts); MLA/FlashMLA (DeepSeek only). GGUF-compatible. | **engine swap** (drop-in binary, same GGUF); `_R4`/`IQ_K` want a **re-quant** |
| **`kvcache-ai/ktransformers`** | ~19,134 | *The* consumer-GPU MoE-offload engine: selective expert offload, **Marlin GPU kernels + AMX CPU expert kernels, CUDA-graph decode**. Purpose-built for our exact rig. | **switch engines entirely** |
| `Nexesenex/croco.cpp` | ~177 | KoboldCPP powered by ik_llama.cpp — ik's quants+kernels in a friendlier runtime | engine swap |
| `LostRuins/koboldcpp` | ~11,301 | tracks upstream; quantized KV + context-shift; little decode edge over stock for us | — |

**Concrete ik numbers:** 27B 4bpw ≈ 1,312 t/s prompt on one 3090 (HN); most published ik headline
gains are *prompt-processing* — decode gains from `-fmoe`/`-rtr`/`-ser` on offloaded MoE are real
but more modest (tens of %), **worth measuring ourselves**.

**SGLang forks:** all low-star hardware/vendor ports (`antgroup` 35★, `bytedance-iaas` 31★,
`Ascend` NPU, `mingfeima` Intel-CPU). **None for the consumer-GPU offload regime.** The
SGLang-adjacent offload work lives in `kvcache-ai/ktransformers`, not an sglang fork — consistent
with treating SGLang as the fully-resident band.

---

## 4. SGLang (fully-resident / distributed band) — one idea worth stealing

SGLang is VRAM-resident + datacenter; its `--cpu-offload-gb` path is slow and **disables CUDA
graph** (#23664), and its offload investment is the **KV cache** (HiCache L1 GPU / L2 RAM / L3
storage) for prefix reuse, not weights. It *does* pin host memory (`cudaHostRegister`, incl. the
KV buffer #26301 — validates our §B2 mechanism). **The transferable insight:** in its Paged-Experts
draft (#29971, tested on a 16 GB 5070 Ti under WSL2) decode went **8.1 → 197 t/s**, attributed
*largely to CUDA-graph capture over the streamed decode path*, which needs stable pinned buffers.
→ pinning's larger payoff may be *enabling graph-style capture*, not just faster DMA (our §B4).

---

## 5. Priority — ranked by expected inference-speed gain (the deliverable)

**Tier 1 — biggest decode wins, test first:**
1. **Optimize expert placement in stock llama.cpp** (free, today): sweep `--n-cpu-moe` to fill
   24 GB VRAM without spilling; add `-ctk q4_0 -ctv q4_0` to keep Flash Attention on Ampere's fast
   path. Often the single biggest lever, no fork.
2. **ik_llama.cpp head-to-head** (engine swap, same GGUF): `-fmoe -rtr -ser` vs our stock build,
   same expert placement. Tests philosophy-(b) fast-CPU-compute vs our (a) pinned-stream.
3. **KTransformers head-to-head** (engine swap): the purpose-built rival; CUDA-graph decode +
   Marlin/AMX kernels. Decide which offload engine wins before more fork-tuning.

**Tier 2 — real, more setup:**
4. **MTP speculative decode** (`--spec-type draft-mtp`, merged; +30% upstream): amortizes weight
   *and* KV movement — the biggest single-stream lever, and central to the long-context agentic case.
5. **IQ*_K / `_R4` expert quant** (re-quant or download): better quality-per-byte + faster CPU GEMM.

**Tier 3 — our novel science (scientific value, modest speed):**
6. **§B2 pinned KV in RAM** — novel for llama.cpp; context-length dose-response.
7. **§B3 prefetch reconciliation via GPU-idle%** — cheap; explains the tax/win split.
8. **§B4 CUDA-graph × pinning** — does pinning enable graph capture (SGLang's real win)?
9. **§B5 `--pin-hot-experts`** — selective vs blanket pin on the generation side.

**Note on the current campaign (3 remaining dense controls):** low *speed* value (they are negative
controls, expected ~0), but they **clinch §B1's mechanism** (grade-3→4). They are cheap and already
running, so let them finish for scientific closure — but the *next investment* is Tier 1, not more
fork-tuning. **The honest headline: the largest untested speed gains are at the engine level
(ik_llama.cpp, KTransformers), not in tuning our fork further.**
