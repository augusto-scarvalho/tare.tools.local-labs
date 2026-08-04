# A3 — KV-cache quantization (asymmetric K/V + sub-/better-4-bit): CLOSED, negative — the KV axis is already at its optimum

**Status: CLOSED, negative / already-optimal. Double-checked 2026-08-04** (source + upstream issues/PRs on the same
GPU class + peer-reviewed literature all corroborate). Gate: `ops/kv-quant-bench.sh`. Raw: `runs/context/a3-kv-quant/`.

This file consolidates the full arc: the question, the first-pass measurement, the double-check (three independent
corroboration levels), the three corrections it produced, and the re-open trigger.

---

## The question (IDEAS_BACKLOG A3, research doc §62)

The doc proposed beating the deployed **symmetric q4_0 KV** by (a) **asymmetric K/V** ("preserve sensitive K, compress
V") or (b) a fancier codec — **SAW-INT4** (Hadamard rotation + token-wise INT4) now, TurboQuant/KVarN/OSCAR sub-4-bit
later — to *extend usable context / residency*. Accept if context headroom rises with no code/tool regression.

**Doc reconciliation (like A1, the doc was right — we measured its predicted failure).** §62.4's experiment table lists
the failure-to-detect for asymmetric K/V as **"incompatibilidade de kernel/offload"**, and the doc cites llama.cpp issue
**#20866** (ref-73, titled *"Asymmetric K/V Cache Quantization GPU Offload Limitation"*) as the operational signal. So
the doc did **not** claim asymmetric K/V would work — it flagged the exact risk. We measured that risk firing. The doc
also lists KVarN/OSCAR as methods that "still need to prove end-to-end gains on varied engines/models" — our null is a
concrete instance. And the doc's headline KV goal (§62.4 "Q8→Q4 KV: double context/residency") is **already achieved
for us**: q4 is lossless here and reaches native 262k in VRAM (Phase A/C, CONTEXT_PLAN).

---

## Measurement (deploy MoE Qwen3.6-35B-A3B Q4_K_M, ncmoe=8, `-fa on`, base 720d7fa40, undervolt clock-stable)

**Robust pass (6 reps, each arm isolated in its own process + 25 s cooldown, pre-arm GPU state logged; 2026-08-04):**

| type_k | type_v | tg64 @ d8192 (t/s) | CV | 95% CI | vs sym q4 | on-GPU? |
|---|---|---:|---:|---|---:|---|
| **q4_0** | **q4_0** | **88.55 ± 0.84** | 0.9% | [87.7, 89.4] | baseline (lossless, §Q) | ✅ fused FA |
| q8_0 | q8_0 | 89.80 ± 3.30 | 3.7% | [86.3, 93.3] | ≈ 0 (lossless, more VRAM) | ✅ fused FA |
| q8_0 | q4_0 | 38.42 ± 10.65 | 28% | [27.2, 49.6] | **−57%** | ❌ CPU offload |
| iq4_nl | iq4_nl | 16.11 ± 0.55 | 3.4% | [15.5, 16.7] | **−82%** | ❌ CPU offload |
| q4_0 | q4_0 @32k | 76.41 ± 2.09 | 2.7% | [74.0, 78.8] | graceful w/ depth | ✅ fused FA |

First-pass (3 reps, no cooldown) gave 87.4 / 33.2 / 33.6 / 18.5 — same signs; the robust pass tightened iq4_nl (CV
24%→3.4%; true mean ~16, the 18.5 was heat-soak-inflated) and confirmed the asymmetric collapse. **Even the noisy
asymmetric arm's 95%-CI ceiling (49.6) is far below the baseline floor (87.7)** → the penalty is statistically
unambiguous. The asym arm's high CV is itself a signature of the CPU-offload path (CPU attention timing is
contention-dependent), not measurement error.

---

## Mechanism — source-verified (`ggml/src/ggml-cuda/fattn.cu`, our build)

The penalty is **not** a slow GPU kernel — it is the whole flash-attention op being **offloaded to the CPU backend**
(a CPU KV buffer is allocated; #20866 shows ~156 MiB) because **no compiled CUDA FA kernel exists for the type combo**:

```c
// ggml_cuda_get_best_fattn_kernel()
#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) { return BEST_FATTN_KERNEL_NONE; }   // asymmetric -> no kernel
#endif
    if (!ggml_cuda_fattn_kv_type_supported(K->type) ||
        !ggml_cuda_fattn_kv_type_supported(V->type)) { return BEST_FATTN_KERNEL_NONE; }  // iq4_nl -> no kernel
```
`ggml_cuda_fattn_kv_type_supported` whitelists only **F16/BF16/Q4_0/Q8_0** by default (Q4_1/Q5_0/Q5_1 need
`GGML_CUDA_FA_ALL_QUANTS`); **`GGML_TYPE_IQ4_NL` hits `default: return false`** — it is *never* whitelisted. The default
build compiles only **4 symmetric FA KV combos**: f16/f16, q4_0/q4_0, q8_0/q8_0, bf16/bf16. `BEST_FATTN_KERNEL_NONE` →
`supports_op` false → the ggml scheduler places the attention op (and KV tensors) on CPU.

**Our build has `GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF`** (confirmed in `build/CMakeCache.txt`). Consequences:
- **Asymmetric is build-flag-gated, not impossible.** With `-DGGML_CUDA_FA_ALL_QUANTS=ON` the mixed kernels compile and
  asymmetric q8/q4 runs on-GPU (upstream #20866: mirkowodtke measured a **~25× prefill recovery** with the flag). We do
  **not** enable it because (i) it is still **dominated** — q4/q4 is already lossless, so any higher-precision asymmetric
  config is strictly worse (more VRAM, zero quality gain), and (ii) the flag balloons compile time / binary size (why it
  is off by default upstream too; the FR **#24485** to flip the default or add a runtime warning was closed `not_planned`).
- **iq4_nl is worse: no FA kernel on *any* CUDA arch, flag or not.** Not Ampere-specific. iq4_nl KV + FA always offloads.

**The CPU-offload penalty grows with context depth** (more KV to process on CPU per step). Our 8k decode numbers therefore
*understate* the deploy-scale (128k) penalty; #20866's prefill data on a 3090 Ti shows the extreme end — asymmetric
prefill **30.6 t/s vs symmetric 1340 t/s (−98%)**.

---

## Why there is no win to chase (physics + architecture)

- **Batch-1 decode is weight-bandwidth-bound; KV is ~3% of the bytes moved** (arXiv:2605.30571, tested A100/H100/L40S/L4:
  ~0.47 GB KV vs ~15.23 GB weights per step on a 7B). Shrinking KV below 4-bit is invisible in wall-clock, and Ampere has
  no native low-bit *attention* compute — same physics as our S2 (MMQ) negative. The real decode lever is the **weight**
  axis (Marlin, PPoPP'25: ~3.87× INT4-weight on Ampere, batch≤~16), which we already capture via MMQ (S2).
- **q4 is lossless on this hybrid by construction.** Gated-DeltaNet holds a growing KV in only ~1-in-4 layers (the rest
  carry a fixed recurrent state), and **QK-Norm suppresses the outlier channels that INT4 fights** — exactly the failure
  mode SAW-INT4/KIVI exist to fix. So there is no quality headroom for a "better 4-bit" to recover here.

---

## The three corrections the double-check produced (vs the first-pass writeup)

1. **SAW-INT4 is a 4-bit method, NOT sub-4-bit** (arXiv:2604.19157, Jia et al., Together AI incl. Tri Dao; code
   `togethercomputer/saw-int4`). It is block-diagonal-Hadamard + token-wise INT4, near-lossless (~1 pt GPQA), on
   **H100 + Triton + FA3 + a forked SGLang** — its own thesis is *"more sophisticated methods give only marginal gains
   once serving compatibility is considered."* Its value is quality-at-4-bit + low integration cost; both are **moot for
   us** (q4 already lossless; not stock-llama.cpp-portable; Hopper, not Ampere). The first pass wrongly lumped it with
   sub-4-bit TurboQuant/KVarN.
2. **Mechanism is CPU-offload of the FA op, build-flag-gated** (above) — sharper than the first pass's "CPU-KV fallback".
   iq4_nl's exclusion is universal (any arch), stronger than the first pass's "sm_86".
3. **Statistical robustness** — the first pass used 3 reps, no cooldown, and iq4_nl at 24% CV (below our S2/S3 standard).
   The robust pass (6 reps, isolation, cooldown, CIs) fixed it and shifted iq4_nl 18.5→16.1.

---

## External corroboration

**llama.cpp issues/PRs (same GPU class):**
- **#20866** (RTX 3090 Ti, Qwen3-32B-Q4_K_M) — asymmetric K/V can't be GPU-offloaded; CPU KV buffer allocated. theo77186:
  *"The kernels do exist, but they aren't built by default … rebuild with `-DGGML_CUDA_FA_ALL_QUANTS=ON`."*
- **PR #7527** (JohannesGaessler, quantized-KV FA vec) — *"performance stays mostly the same with a quantized KV cache;
  quantizing K makes it slightly better, quantizing V slightly worse"* → **KV quant is a memory feature, not a speed
  feature** on NVIDIA.
- **#22411** (am17an): *"the fused FA kernels only support matching K/V quantization types; if they don't match it
  silently falls back to the slower non-fused implementation."*
- **#24485** — FR to make `FA_ALL_QUANTS` default / add a runtime warning → **closed `not_planned`** (silent-fallback
  footgun is live in stock builds).
- **#20969** — TurboQuant KV: community discussion, **not upstream**. A real 3090 CUDA fork exists
  (`spiritbuun/llama-cpp-turboquant-cuda`, turbo3=3.25-bit) claiming ~q8 quality at 3.5× compression and ~97.5% decode on
  128k — **unverified fork marketing**; even if true it's a *compression* (memory) win, which for us is dominated (q4
  already reaches native 262k; freeing more VRAM buys < 1 ncmoe step and the free monitor→iGPU replug gives ~1.4 GB ≈ 3
  steps at zero quality risk).

**Papers:**
- **SAW-INT4** (2604.19157) — see correction #1; its thesis matches ours.
- **Batch-1 decode roofline** (2605.30571) — KV ~3% of bytes at batch-1 → sub-4-bit KV wall-clock-invisible on Ampere.
- **KIVI** (2402.02750) / **KVQuant** (2401.18079) — asymmetric/per-channel KV **require custom CUDA/Triton kernels**;
  per-channel-K breaks the contiguous single-buffer + fused-FA layout llama.cpp uses → not portable to our stock path.
  Their wins are all measured *with their own kernels* on A100.
- **Marlin** (PPoPP'25) — the Ampere low-bit win lives on the **weight** axis, not KV; converges to fp16 outside batch≤~16.
- Field consensus (2025-26): **4-bit KV = near-lossless default; below 4-bit is a cliff** (3-bit degrades, 2-bit −10..−15 pts);
  the frontier moved OFF "fewer bits" toward mixed-precision/importance-aware/eviction/architectural KV reduction.

---

## Adjacent check (ref-74 / #15039 "Draft Model with Q4 KV Cache Acceptance Regression")

Not a concern on our config: A1 already measured **MTP draft accept = 99.17% (byte-identical) across 6 depths to native
256k with q4 KV** — the regression #15039 reports does not appear on our base/model with our q4 KV. No action.

---

## Deploy takeaway & re-open trigger

**Keep symmetric q4_0 KV for long context (fast + lossless); q8_0 symmetric if ever wanted (≈ same speed, more VRAM).
Never run asymmetric KV on a default build (−57%). Never use iq4_nl KV (−82%, no CUDA FA kernel anywhere).** The KV axis
is at its optimum for this hardware/model — same pattern as S1/S2/S3/A1: already-captured or physically dominated.

**Re-open only if:** a sub-4-bit **fused-FA** KV kernel lands **upstream** for sm_86 (watch llama.cpp KV-quant PRs / the
TurboQuant discussion #20969), **or** a future served model becomes genuinely KV-bound rather than weight/ncmoe-bound
(our GDN hybrids are not; a dense non-hybrid pure-transformer with huge KV could be). If we ever target ≥256k context
(YaRN, §D) the calculus is unchanged — q4 already reaches native, and the next VRAM lever is the monitor replug, not KV bits.
