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

## 1b. "Should we build a new fork cherry-picking what we liked?" — decided NO (for now), 2026-08-01

Asked directly after §E1. The answer is **not yet, and maybe not as a fork** — and the reasoning
is a guard against the exact trap this project already fell into (running three A/Bs against a fork
whose feature gates were all closed):

- **What we'd cherry-pick is thin.** Of the fork's 3 commits: **pinning is already upstream**
  (slaren #6206 — a flag, off by default) and, per §E1, is **null at the optimal placement**
  (helps only forced-heavy-offload); **prefetch is a −22% tax on the 3090** (drop it);
  prefetch-fix only exists to support prefetch. Nothing there is both ours and worth new code.
- **"First fork + our changes on today's upstream" already exists** — it is the `rebase` build
  (the patch replayed onto master with zero conflicts, identical diffstat). It is on disk.
- **Our actual changes are harness (Python), not inference C++** — the model registry, `robust.py`,
  the `--kv/--gguf/--tag/--dense` flags, `ab_isolate`. Those belong in `local-model-lifecycle`,
  not in a llama.cpp fork. We have not written new inference C++ to collect.
- **It is premature.** §E1 showed the fork's mechanisms are null at the placement we would actually
  run. The decode levers that *do* move at good placement are still **unmeasured**: §E2 (ik_llama,
  compute-on-CPU — could moot a llama.cpp fork entirely), §E3 (KTransformers), §E4 / the fork
  author's own **`turbo-mma-decode`** branch (a fused flash-attention *decode* path — the one that
  targets decode instead of prefill, still unmeasured, sitting behind `GGML_TURBO_MMA_FUSED` in the
  `stack` build). Assembling a fork now = curating parts before knowing which work.

**The version that IS worth doing, and when.** After §E2–§E4 identify which levers win decode at
good placement and are *not* upstream, consolidate a build we control: base = today's upstream
master (already has `--n-cpu-moe`, `--spec-type draft-mtp`, EAGLE3); + pinning rebased and gated
off (for the forced-offload regime); + whichever decode lever survives (candidate #1: turbo-mma-
decode); − prefetch. A fork that houses **only validated, non-upstream decode wins** — not an
anthology of everything tested. Until then, §E2–§E4 run against **current upstream at ncmoe=6** as
the philosophy-(a) baseline (`MASTER_BIN`/`REBASE_BIN` already built — no new fork needed to
proceed). See `[[placement-is-the-decode-lever]]`.

---

## 1c. Build consolidation — DECIDED YES (2026-08-02): the fork is upstream master + §B2b, nothing more

§1b said "not yet, and only once a non-upstream lever is proven." §E1–§E5 and §B1–§B5 now close that
condition, and a **due-diligence scan** (4 parallel research agents over ggml-org issues/PRs, rival
forks, the Fable fork's current state, and correctness footguns for our exact deploy) ran BEFORE building,
to make sure no lever was missed. What it found:

- **Every validated decode WINNER is upstream:** placement (`--n-cpu-moe`), CUDA graphs (default on),
  MTP (`--spec-type draft-mtp`). **EAGLE3** (#18039, merged) is *inferior* to MTP for offloaded MoE
  (parallel verify amplifies expert H2D) — stay on `draft-mtp`.
- **The one candidate non-upstream decode lever — the MoE expert cache** (Fable's post-snapshot
  `fable5/moe-expert-cache`; upstream #20757's PoC; an early copy already in our `stack` build) — was
  **tested (§E5) and is NULL/redundant** with static placement, because Qwen3.6's load-balanced routing
  isn't concentrated enough (top-8 = 28% hit). At every VRAM budget, static `--n-cpu-moe` ≥ the cache.
- **No other engine/fork beats stock decode here:** KTransformers still AMX-gated (even with its new
  AVX2 path), ik redundant, vLLM/#20757 caches are 8 GB-targeted, research papers ship no consumer path.
- **The only genuinely novel, validated, non-upstream lever the whole campaign produced is §B2b**
  (KV-host-pin, `patches/b2b-kv-host-pin.patch`) — and it only pays in the VRAM-starved long-context
  (`--no-kv-offload`) regime.

**DECISION: the consolidated fork = FRESH upstream master + the §B2b patch (env-gated), and nothing
else.** Exactly one non-upstream win, per §1b's rule ("only validated non-upstream decode wins, not an
anthology"). prefetch (−22% decode tax), turbo-mma-decode (null, STATUS §6), and the expert cache (§E5
null) are all excluded on evidence.

**Base — pinned to `720d7fa40`, NOT fresh master (the blessing gate earned its keep).** The plan was to
rebase §B2b onto fresh upstream master (`f5919bf45`, +102 commits, includes #26135). It built and §B2b
applied clean — but the **`draft-mtp` token-identity gate FAILED on fresh master**: base ≠ mtp
(929 vs 945 chars, deterministic mid-stream divergence at char 729), while `720d7fa40` is exact
(base == mtp). One of the 102 commits regressed spec-decode exactness for qwen3.6moe — a live instance
of open issue **#23335**. Diagnosis without a bisect: CUDA attention/softmax/argsort kernels were
**unchanged** in range (class ruled out); the base-decode shift under `-fa`+`--n-cpu-moe` pointed at
**#25832** (FLASH_ATTN_EXT op-offload scheduling) — but a **revert-and-rebuild test refuted it** (still
diverges), so the culprit is elsewhere and unbisected. Decision: **house the fork on the VALIDATED
`720d7fa40` + §B2b** (branch `lifecycle`, commit `0e4e2d897`); MTP exactness is load-bearing (it's what
makes the +27% lever "free"), and #26135 is immaterial for our mmap-default config. The fresh-master
worktree was dropped. Fresh master remains re-testable for a future upstream report on #23335.

**Hardening applied:**
- **Blessing gates (`bless_fork.sh`), all PASS on the 720d7fa40 binary:** G1 §B2b engages (KV on
  `CUDA_Host(B2b)`); G2 `draft-mtp` token-identity at `--spec-draft-n-max 4` (IDENTICAL, 80.5% accept);
  G3 coherence + `-nkvo` (both non-degenerate, KV-on-GPU == `-nkvo` output → #20140 not triggered).
- **`FA_ALL_QUANTS` left OFF** (matches the validated build): the whole campaign ran FA-on-GPU at that
  setting with q8_0 KV, so the kernel is present (#24485 fallback not a risk for our config); turn it ON
  only if other KV quant combos are ever needed.
- The 27B "fused Gated Delta Net disabled" warning is **cosmetic** (correctness-safe fallback), no guard.

**Future R&D, explicitly not now:** #20757's expert cache would only pay on a *concentrated-routing*
model (not Qwen3-family); §B2b currently pins ~25% of KV tensors and could pin all layers if long-context
becomes a priority. The 5 dead build trees (base/fork/rebase/stack/local) are retired **after** the fork
is built and blessed.

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
1. **Optimize expert placement in stock llama.cpp** (free) — **DONE 2026-08-01, the project's
   biggest decode win.** qwen36-35B-Q4: sweeping `--n-cpu-moe` from 40 (max offload, our whole
   campaign's placement, 21 GB VRAM idle) down to **6** took decode **27.6 → 101.7 t/s (+268%,
   3.7×)**, respecting the 4 GB VRAM reserve (ncmoe=4 breaks it). **KV `q4_0` vs `q8_0` was a null
   lever at 8 k ctx** (±2%, frees ~46 MB) — its payoff is long-context only. **New stock baseline
   for every later A/B: ~102 t/s at ncmoe=6, not 27 t/s.** See STATUS.md §E1. Pinning is null at
   this optimum — the fork only helps in forced-heavy-offload. Next Tier-1 is the engine swap.
2. **ik_llama.cpp head-to-head** — **DONE 2026-08-02.** At the operating point (ncmoe=6, matched
   GGUF/KV/`-fa`, n=4 clean): **decode TIE** (+0.29%, within noise), ik **+75% prefill**. Sweeping
   ncmoe shows ik's decode edge GROWS with offload (+2%→+5.3%→+9.6% at 6→16→28) — philosophy (b)
   degrades less than (a) — **but ik is unusable there on 64 GB**: `-rtr` OOMs at heavy offload,
   ik's RSS breaches the 16 GB reserve at ncmoe16+ (15.3/9.6 GB free), ncmoe40 generation crashes.
   **Engine swap is a decode tie where we run and RAM-unsafe where it would win.** Revisit at 128 GB
   + a model too big to place. See STATUS.md §E2. Needed a `--reset-between` (per-config WSL reset)
   and per-arm ports to measure cleanly.
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
