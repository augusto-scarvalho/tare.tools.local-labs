# IDEAS_BACKLOG — panorama → ROI-ranked validation backlog (2026-08-04)

Source: `harness_modelos_locais_..._panorama_campos_emergentes.html` (75 sections, 6 parts).
Method: 4 Sonnet-5 (high) readers extracted **120 idea-cards**; this doc is the filtered,
ROI-ranked synthesis against **our** hardware and fork — not the raw dump.

**Our lens.** RTX 3090 24GB (Ampere sm_86: INT8 Tensor Cores yes, TF32==FP32, **no FP4/bf16 edge, not
Hopper/Blackwell**), i7-13700K (**no AMX/AVX512**), 64GB DDR5-5600, WSL2. Fork `lifecycle` already has:
`--n-cpu-moe` placement, CUDA graphs, MTP draft-mtp, q4_0 KV (validated lossless + 128k), chunked GDN TF32
kernel, expert-prefetch + CPU-weight-pin + MoE-expert-cache (gated OFF), quality axis on HumanEval+.
**Our measured bottleneck: decode is transfer-bound = CPU-offloaded MoE expert streaming over PCIe** (core-clock
and mem-OC both ~0%). ROI = (benefit × applicability) ÷ (effort × risk); the highest-ROI ideas attack expert
streaming or are HW-matched (Ampere-measured) wins we can build like we built the GDN kernel.

---

## ⚠️ Scoping first: two axes in this document

The doc is 70% about building a **code-veracity / agentic-coding-quality harness** (§7-19, §46-58) and 30%
about **engine/inference** (§30-40, §59-75). Our project today is the **engine** (decode-optimization fork +
A/B lab), not an agentic coding product. So:

- **Engine axis** (Tiers S/A/B below) = direct extensions of our fork. This is "our backlog."
- **Harness-quality axis** (§7-19 TaskContract/mutation/critics/anti-slop; §46-58 experiment-lab) = a **new
  product direction**. Much of it we partly have (our A/B lab, quality axis); the rest is a strategic build
  decision, not an engine experiment. Flagged as Track H at the end.

---

## Tier S — do first (highest ROI: hits our bottleneck or is an Ampere-measured win, buildable in-fork)

### S1. Expert-access profiler → learned hot/warm/cold placement (`ExpertResidencyController`) — §66
- **What:** profile per-layer/per-expert access frequency + bytes-transferred + PCIe stall; classify experts
  hot/warm/cold; keep hot resident, async-prefetch next, per-expert quant — replacing static `--n-cpu-moe`.
- **Why top ROI:** attacks THE bottleneck (expert streaming). Doc names Qwen3.6-35B-A3B and calls it *"mais
  transformador que alterações globais de batch/KV."* We already have the prefetch/pin/cache scaffolding gated off.
- **Split to de-risk:** **S1a (cheap, do now):** build the profiler/simulator only — instrument expert-access
  histogram + bytes-moved + PCIe-stall + CPU/GPU co-util during decode (§69 recommends this exact instrumentation
  and cites our own `--no-mmap` result). **S1b (big):** learned placement controller, gated on S1a showing
  concentrated/predictable routing (note §E5: Qwen3 routing was load-balanced → may cap the win — S1a settles it).
- **Validate:** does per-expert access concentrate enough that resident-hot beats static ncmoe=8 on decode t/s at
  equal VRAM? Accept if ≥ +10% decode or ≥ same t/s at lower VRAM, quality-neutral.
- Effort: S1a low, S1b high (C++/CUDA). Maturity: speculative but our-exact-target.

### S2. INT8 Tensor-Core fused dequant+GEMM kernel — §61
- **What:** fuse dequant+bias+act+GEMM so low-bit weights actually use the 3090's INT8 Tensor Cores instead of
  dequant-to-FP16-then-GEMM; autotune per shape; explicit FP16 fallback when it loses.
- **Why:** the one **measured-on-RTX-3090** number in the whole doc: **GEMMs 2.8-4.2× / ~9-10% end-to-end**.
- **Caveat for us:** that win was a compute/GEMM-bound path; our MoE batch-1 decode is bandwidth-bound, so expect
  most of it on **prefill** and on the **dense-27B**, not MoE decode — measure both.
- **Validate:** microbench real GEMM shapes (prefill ubatch=2048) INT8-fused vs current; then E2E prefill t/s.
  Accept if E2E prefill ≥ +5% quality-neutral. Adopt the doc's "quality_card + kernel_card" dual report.
- Effort: high CUDA (our wheelhouse — we did GDN). Maturity: consolidated.

### S3. N-gram speculative decoding + drafter-selection policy — §35 / §63.3
- **What:** add n-gram spec-decode (no extra model, wins on repetitive code) alongside MTP; formalize drafter
  priority MTP > n-gram > small drafter; **regression-test** it (doc + community: a mismatched drafter *reduces*
  Qwen3.6 throughput).
- **Why:** best ROI-per-effort on the engine axis — llama.cpp already ships n-gram, so it's config + benchmarking.
- **Validate:** on a code-edit corpus, n-gram vs MTP vs MTP+ngram: accepted-draft ratio + wall-clock. Accept the
  drafter only where acceptance beats verification cost; keep the loser off.
- Effort: low. Maturity: consolidated.

---

## Tier A — high value, moderate effort, our stack

### A1. Windowed-MTP / adaptive speculation — §22 / §35
- Targets our **exact** combo (MTP draft + hybrid GDN + long context), where MTP's edge is known to drop because
  the draft pays full-context KV. Bleeding-edge, kernel work. Prototype on the GDN path; validate acceptance-rate
  and decode t/s at 32k/64k/128k vs plain draft-mtp. High potential, uncertain.

### A2. ThinkingCap long-to-short LoRA on the **dense 27B** — §23.6 / §56 / §64
- Halving reasoning tokens is a big wall-clock win on our decode-bound setup. ThinkingCap full-FT claims **45.8%
  reasoning-token cut at ~-0.7pp accuracy**; a community rank-64 SVD LoRA exists (**unvalidated**). **Restrict to
  the dense 27B — do NOT apply to the 35B-A3B MoE (shape-incompatible).**
- Cost: BF16 27B (~54GB) SVD extraction → CPU-RAM offload (our 64GB) or cloud; runtime LoRA applies to our GGUF.
- **Validate (doc's §56 protocol, right-sized):** reconstruction gate (base+LoRA ≈ full FT) → λ/rank sweep →
  accept if ≥25% reasoning-token + ≥15% wall-clock reduction on target, quality within ±1pp (ROPE), zero
  tool/JSON/MTP breakage. This is the doc's flagship P1 and the best "combined model+engine" experiment for us.

### A3. Asymmetric K/V quant + SAW-INT4 KV — §62
- We're already ahead (q4_0 KV lossless); next headroom = quantize K and V asymmetrically, or **SAW-INT4**
  (Hadamard rotation + token-wise INT4, explicitly designed for low serving-integration cost). Extends usable
  context/residency. **Validate:** needle + code-suite at 8k/32k/64k/128k vs q4_0; accept if context headroom up
  and no code/tool regression. (Sub-2-bit KVarN/OSCAR = Tier C watchlist; TurboQuant/UltraQuant = cut, AMD-only.)

### A4. Spec-decode + benchmark instrumentation discipline — §22 / §40 / §63.4
- Extend our A/B to log per-config **accepted-draft-tokens, acceptance rate, drafter-time-vs-saved, draft VRAM,
  TTFT/TPOT, wall-clock, and downstream code quality** — never tokens/s on one long deterministic output. Adopt
  the fixed fractional-factorial matrix (§40): our levers map ~1:1. Cheap, hardens every future engine claim.

---

## Tier B — worthwhile, secondary or higher-uncertainty

- **B1. Heterogeneous memory tiering + transfer instrumentation — §69.** Formalize VRAM/RAM-pinned/RAM/(NVMe)
  tiers over our existing CPU-weight-pin; the instrumentation half **merges into S1a**. NVMe tier = new, low near-term ROI.
- **B2. Auto-search engine flags with Optuna+ASHA (Pareto) — §70 / §51.** Automate what our lab does by hand over
  {ncmoe, KV type, spec/draft-len, batch/ubatch, mmap, cache eviction}; multi-objective {quality, TTFT, TPOT,
  VRAM, cost_per_accepted_task}. We already tune manually well → medium ROI. Optuna (not Ray/Ax — single GPU).
- **B3. Robust-stats + broaden the golden set — §50.2 / §53 / §54.** Adopt concrete ROPE/non-inferiority margins
  (±1pp quality, ≥15% wall-clock, ≥20% reasoning-tokens, 0.5pp crash ceiling) as promotion gates; add a **second
  benchmark axis** (BigCodeBench / SWE-bench Verified) since HumanEval+ alone is saturated/contaminatable. Cheap-ish.
- **B4. Hybrid-model cache-correctness probes — §21 / §67.** Explicit probes that slot save/restore, partial-
  sequence-removal, and speculative rollback are correct on our GDN hybrid (not just plain-KV). Cheap correctness insurance.
- **B5. Task-oriented quant gate + layer-sensitivity map (Q3/mixed) — §28 / §60.** Push below Q4_K_M with
  sensitivity-guided mixed precision, gated on task success (not perplexity). Research-y; only if we want more VRAM headroom.

---

## Tier C — watchlist (low near-term ROI / gated on other work)

- Multi-LoRA library + router (§73) — only after we have ≥2 useful adapters (gate on A2).
- Workflow-aware KV planner / KVFlow / HiCache (§18/§33) — multi-agent serving; low value for single-user.
- Sub-2-bit KV (KVarN/OSCAR), sub-4-bit weight frontier (NanoQuant/Bielik-Q2) — doc says unproven; track only.
- Full Experiment-Lab build-out (§52/§57/§58) — we already have a lab; adopt the schemas (`TrialResult`,
  `PromotionDecision`) incrementally, not wholesale.

---

## Track H — the harness-quality / agentic-coding axis (strategic, not an engine experiment)

Only relevant **if** we expand from "engine fork" to "local coding-agent product." Highest-value pieces if so,
all HW-free / pure orchestration: TaskContract + versioned deltas (§7), RepositoryEvidencePack structural
retrieval (§8, target ≥30% fewer prefill tokens — synergizes with our decode-bound setup), test-baseline
non-weakening gate (§9), mutation testing with an **independent** test-writer (§10), anti-slop maintainability
gate (§16), independent-critic / process-reward small model (§15/§25, a 4-8B QLoRA critic fits our 3090
sequentially), and the model-portfolio-by-role idea (§20/§29/§65 — our lineup already matches). This is a big,
separate initiative — decide the direction before pulling any of it into the backlog proper.

---

## Cut (out of scope / HW-blocked / redundant — recorded so we don't re-litigate)

- **HW-blocked:** FP4/NVFP4 (§61.2, Blackwell); FA3 (§37, Hopper — *useful negative signal: don't chase it*);
  TurboQuant/UltraQuant KV (§34/§62, AMD CDNA4 numbers, non-transferable); disaggregated prefill/decode &
  tensor-parallel (§36/§74, multi-GPU); CPU-FP8 expert kernels (§66, needs AMX we lack).
- **Out of scope:** multimodal modality caching (§71), diffusion/video (§72, real-time explicitly non-Ampere),
  distributed/home-cluster inference (§74, needs a 2nd machine — only agent-level parallelism if a laptop appears).
- **Needs cloud / disproportionate:** full RL training of the 35B (§24); formal-verification tiers F3/F4 (§13).
- **Redundant with what we have:** §30 phase framework, §38 MoE-pinning checklist, §40 matrix (we have the lab),
  E5 KV-q4 (done, lossless), E6 MTP (done), §67 hybrid direction (we already run one).

## Doc's independent validation of our choices (confidence signals)
§29 model portfolio ≈ our exact lineup · §37 confirms CUDA graphs + skip FA3 · §61.2 confirms our Ampere/TF32
limits · §42/§75 put "expert pinning / custom KV kernels" at P2-P3 — **we already built them**, i.e. our fork is
ahead of the doc's assumed baseline on the engine axis.
