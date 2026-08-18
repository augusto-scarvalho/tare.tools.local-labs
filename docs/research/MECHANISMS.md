# MECHANISMS — how inference speed works here, and the levers

The theory behind the experiments: why a switch like pinning moves prefill 2× but generation
only ~2%, why generation is bandwidth-bound in the first place, and which levers actually
reduce the bottleneck. Companion to `STATUS.md` (measured results) and `EXPERIMENTS.md`
(methodology). Numbers are order-of-magnitude for this box (RTX 3090, 24 GB; 64 GB RAM;
PCIe 4.0 x16).

---

## 1. The memory hierarchy — what is placeable, what is not

| tier | size | bandwidth | can you *place* GBs of weights here? |
|---|---|---|---|
| registers | ~KB / thread | ~TB/s | no — hold the current instruction's operands |
| L1 / shared memory | ~100 KB / SM | ~TB/s | no — kernel-managed working set |
| L2 (GPU) | ~6 MB | very high | no — automatic (hardware) |
| **VRAM** | **24 GB** | **~936 GB/s** | **yes** — the fast placeable tier |
| **RAM (over PCIe)** | 64 GB | **~25 GB/s** (pinned) | **yes** — the slow placeable tier |
| NVMe | TB | ~GB/s | (fallback; disk-thrash if the model exceeds RAM) |

**Registers and cache are not a placement tier.** Cache is hardware-managed — it transparently
holds recently/soon-used data; you cannot say "put this weight tensor in L2". And the model is
far too big for cache anyway (L2 is 6 MB; one layer is hundreds of MB; the model is 16–40 GB).
So weights **live in VRAM/RAM and stream through** the caches and registers during compute — the
fast on-chip tiers are already used maximally by the kernels, one tile at a time. The only tiers
you place GBs of weights into are **VRAM (fast) and RAM (slow)** — the ~20–40× gap this whole
project tunes with `-ncmoe` / `-ngl`.

---

## 2. Pinning — the fork's real win (`GGML_CUDA_REGISTER_HOST` → `cudaHostRegister`)

Normal RAM (malloc/mmap) is **pageable**: the OS can move or swap its pages, so the physical
address is not fixed. **Pinned** (page-locked) RAM is locked to a fixed physical address.

The GPU's DMA engine can only read directly from RAM whose physical address is fixed:

- **From pageable RAM:** the driver cannot DMA directly. It copies the data to an internal
  pinned staging buffer, *then* DMAs from there — **two copies**, and the staging copy is
  synchronous (blocks). This is why "`cudaMemcpyAsync` out of pageable memory is not async".
- **From pinned RAM:** the DMA reads **directly** RAM→GPU — **one copy**, genuinely async
  (overlaps compute). Effectively ~**2× host→device bandwidth**.

The fork's `register_host()` calls `cudaHostRegister` on the model's mmap, so the offloaded
expert weights are pinned → each RAM→GPU expert upload is single-copy and async.

**Why prefill 2×, generation only ~2%.** Prefill uploads nearly all experts in one big batch →
transfer *is* the bottleneck → the bandwidth doubling shows up as ~2× (measured +105–137%).
Generation streams only the few active experts per token → transfer is a *smaller fraction* of
per-token time → the win is ~2%, and it **scales with active experts** (STATUS §B1 dose-response:
4→6→8 active = ~0→0.8→2.0%).

**The cost.** Pinned pages cannot be evicted or swapped → they occupy physical RAM permanently
(non-reclaimable). Pinning a large model locks GBs — which is *literally* why Nemotron/Laguna-S
did not fit the safety envelope: pinning the mmap locked more RAM than was free. **Pin trades
bandwidth for locked RAM.**

---

## 3. Why generation is bandwidth-bound — the roofline

In decode (batch = 1) you load a weight matrix and use it for **one** token: a matrix×**vector**
product, ~2 FLOP per weight byte — **arithmetic intensity ~2**. The 3090's roofline "knee" (where
compute would finally become the limit) is **tens of FLOP/byte**. At ~2, you are far to the left:
**deeply memory-bound** — the compute units sit idle waiting on bytes.

Prefill is the opposite: a large batch reuses each weight across many tokens (matrix×**matrix**),
intensity climbs past the knee → **compute-bound** → fast per token. Same hardware, same weights;
the batch size flips the regime. **This is the single most important fact about local inference
speed:** decode is slow because each byte moved does almost no math.

---

## 4. Moving less — the levers (and which reduce bytes vs merely hide them)

Since the weights must stream through the hierarchy regardless, the win is not *where* to place
them but **how many bytes cross per unit of useful work**. Two distinct families:

**A. Batching — amortize the movement over more work (reduces bytes/token).**
Process B tokens against one weight load → the load is amortized by B; at B ≈ 32–64 you cross
the knee into compute-bound. Getting B > 1 in decode:
- **Continuous batching (multi-request):** the server batches tokens from many concurrent users;
  one weight load serves all. Raises **throughput**, not single-stream latency (unless the agent
  runs parallel branches / tool calls that can batch).
- **Speculative / MTP decode (single-user, batch on the time axis):** a draft model or the MTP
  (`nextn`) head proposes K tokens; the big model **verifies all K in one batched pass**. Accepted
  tokens cost one weight load for K tokens → K× amortization. The lever for a lone agent.

**B. Pipelining / overlap — hide the movement you cannot avoid (does NOT reduce bytes).**
Split a big transfer into chunks and overlap each with the previous chunk's compute
(double-buffering). The fork's `prefetch` does exactly this. It hides latency but moves the same
bytes — and in our regime we measured it as a **tax**, not a win, because the overlap machinery
costs more than it saves when transfer is not the sole bottleneck.

**Also reducing bytes:**
- **Selectivity** — move only what's needed. The fork already copies only the *used* experts in
  decode (k of 256), not the whole tensor.
- **Quantization** — fewer bytes per weight → better arithmetic intensity *and* more fits in VRAM.
- **Flash attention** — keeps the KV working tile in **shared memory** (the fast on-chip tier)
  instead of round-tripping through VRAM. A concrete case of exploiting the fast tier by moving
  less.
- **Operator fusion** — more math per byte loaded. The fork's `turbo-mma-decode` was exactly this
  attempt (fused GQA decode) — measured **inert** on the 3090 (it did not beat upstream's kernel).

| technique | reduces bytes? | helps 1 user? | helps KV-bound? |
|---|---|---|---|
| continuous batching | amortizes weights | only with parallel branches | no |
| **speculative / MTP** | **amortizes weights + KV** | **yes** | **yes** |
| pipelining (prefetch) | no (hides) | marginal | maybe |
| quantization | yes | yes | yes (smaller KV too) |
| flash attention | moves KV less | yes | yes |

---

## 5. The KV nuance (why MTP matters most for long-context agentic)

Batching amortizes **weights** but **not the KV cache**: each sequence has its own KV, growing
with context. So continuous batching accelerates the weight-bound regime but **not** the KV-bound
regime (STATUS §B2, KV-in-RAM). **Speculative / MTP amortizes both** — K tokens per pass means K×
fewer weight loads *and* K× fewer KV reads. That is why, for a single long-context agent
(128k, KV in RAM), the ordered levers are **quantization** (fewer bytes) then **MTP** (amortize
weight + KV per K) — and why it matters whether llama.cpp actually *uses* the `nextn` head it
loads, which is a companion A/B of §B2.

---

## 6. How this maps to the project

- **§B1** — pinning the expert-weight stream (RAM→GPU). Settled: ~2× prefill; small,
  active-expert-scaled generation gain when transfer-bound.
- **§B2** — pinning the *KV* stream when KV is in RAM (`--no-kv-offload`); same mechanism, new
  buffer; dose-response on context length.
- **Agentic model** (`[[agentic-local-model-plan]]`) — the long-context substrate where
  quantization + MTP are the levers, and §B2 rides on the same setup.
- **Inert here, recorded:** `turbo-mma-decode` (fusion) and the `prefetch` (pipelining) — both
  real techniques, both failed to pay on this hardware/regime. Kept as measured negatives.
