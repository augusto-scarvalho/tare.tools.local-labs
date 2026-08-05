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
- **Multimodal axis** (§71 VLM, §72 image gen) = **opted in 2026-08-04** — see Track M. M-A (VLM) extends our
  own llama.cpp fork (`libmtmd` already built); M-B (image gen) is a separate ComfyUI/diffusers engine.

---

## Tier S — do first (highest ROI: hits our bottleneck or is an Ampere-measured win, buildable in-fork)

### S1. Expert-access profiler → learned MoE placement (`ExpertResidencyController`) — §66
- **What:** profile per-expert access; classify hot/warm/cold; keep hot resident, async-prefetch next — replacing
  static `--n-cpu-moe` with learned placement. Doc names Qwen3.6-35B-A3B, calls it *"mais transformador que
  batch/KV."*
- **✅ S1a DONE 2026-08-04 — verdict NEGATIVE for Qwen3.6 (re-confirms §E5).** The tooling already existed
  (`tools/moe-trace/simulate.py` + `qwen36-35b-moe-trace.csv`); no new code needed. Fresh sim: routing is
  **load-balanced, not concentrated** — top-10% of (layer,expert) pairs carry only **17.6%** of decode traffic;
  top-64 (25%) needed for ~79%. Static top-S placement equals/beats every dynamic policy at matched VRAM because
  dynamic churn's PCIe **upload dwarfs the miss it saves** (LRU@16 slots: 47% hit but 26 ms upload vs 7 ms miss =
  net negative). **→ S1b (learned/dynamic controller) is DEAD for our deploy MoE** — nothing concentrated to learn.
- **✅ S1 CLOSED 2026-08-04 — all 5 on-disk MoEs are load-balanced, none concentrated.** Screened Qwen3.6-35B,
  GPT-OSS-20B, Gemma-4-26B, Ernie-4.5-21B, Granite-4.0-H (5 labs) via `ops/moe-routing-screen.sh` (1 command:
  moe-trace + simulate.py). Top-10% of (layer,expert) pairs carries only **12-22%** of decode routing everywhere
  (uniform = 10%). This is architectural convergence — modern MoE training universally load-balances (aux-loss /
  aux-loss-free) so no expert is wasted. → **S1b (learned placement) is null across our whole fleet, not just Qwen.**
- **Posture for future / new-architecture models:** do NOT pre-build S1b — EV is poor (the industry trend is toward
  *more* balancing). The durable asset is the **screen**, now banked as a standing gate: run
  `ops/moe-routing-screen.sh <new-model>` and build S1b only if a model **fails** it (top-10% >> 10%). Watch for
  **shared-expert** designs (→ a cheap static "shared always resident" rule, not a learned controller) or
  non-load-balanced / hash-routed architectures.
- Effort spent: ~nil. Outcome: the top-ROI de-risk paid off as a negative that saves the whole S1b CUDA build,
  and produced a reusable gate for every future MoE.

### S2. INT8 Tensor-Core fused dequant+GEMM kernel — §61
- **What:** fuse dequant+bias+act+GEMM so low-bit weights actually use the 3090's INT8 Tensor Cores instead of
  dequant-to-FP16-then-GEMM; autotune per shape; explicit FP16 fallback when it loses.
- **Why:** the one **measured-on-RTX-3090** number in the whole doc: **GEMMs 2.8-4.2× / ~9-10% end-to-end**.
- **✅ S2 CLOSED 2026-08-04 — the kernel S2 asks for ALREADY EXISTS and is the DEFAULT (llama.cpp MMQ). Nothing
  to build; same pattern as S1/GDN (premise false on our HW).** Confirmed on three axes:
  - **Source (definitive):** on Ampere, `ggml_cuda_should_use_mmq(Q4_K, sm_86, any ne11)` returns `true`
    UNCONDITIONALLY (`mmq.cu:310` `turing_mma_available` early-return; the `ne11<MMQ_DP4A_MAX_BATCH_SIZE` cutoff
    only fires on non-tensor-core / Pascal cards). MMQ uses `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32` =
    INT8×INT8→INT32 Tensor Core, loads weights quantized + dequants in-register into int8 tiles (never
    materializes an FP16 weight matrix). It IS the "fused dequant + INT8-TC GEMM" the doc describes.
  - **Empirical (default MMQ vs a FORCE_CUBLAS build — the S2 delta, in reverse; prefill pp512/pp2048, -ub 2048,
    r5, undervolt clock-stable):** dense-27B pp512 MMQ **+11%**; dense-27B pp2048 cuBLAS **+5%**; **MoE ncmoe=0
    MMQ +420%; MoE ncmoe=8 (deploy) MMQ +268%.** For the deploy MoE, MMQ int8-TC CRUSHES cuBLAS (grouped
    per-expert GEMMs have tiny per-expert batch → cuBLAS overhead is catastrophic). The doc's "2.8-4.2× GEMM"
    win is real and **we already have it** via MMQ.
  - **Upstream corroboration:** PR **#8075** made MMQ default at all batch sizes on tensor-core cards, motivated
    by **VRAM savings** (explicitly accepting a large-batch speed hit); PR **#8062** maintainer bench (3090,
    Q4_K_S, pp2048: MMQ = 0.82× cuBLAS — same direction as our dense +5%); PR **#7921** (int8-TC k-quant kernels;
    Q8_1 activation-quant precision loss "negligible"); `docs/build.md` (MMQ int32-accumulates → the **more**
    numerically robust path; FORCE_CUBLAS risks FP16 overflow + costs VRAM). **No open/rejected PR proposes a
    fused int8-TC GEMM beyond MMQ** — MMQ *is* that kernel and is treated as mature.
- **The one residual (recorded, NOT adopted):** FP16 cuBLAS beats MMQ by ~5% on **large-ubatch DENSE-27B prefill**
  only. It's the OPPOSITE of what S2 proposed, dense-only (not the deploy MoE, where it's −70%), VRAM-costly (FP16
  dequant buffers on our VRAM-tight box), and off the decode/transfer-bound critical path → net not worth a
  per-shape heuristic. Available as a knob if a dense-heavy, VRAM-loose, long-prompt workload ever appears.
- **Gate banked:** `ops/mmq-vs-cublas-bench.sh` (builds the FORCE_CUBLAS A/B binary on demand; re-run if the MMQ
  heuristic or a new quant/arch changes). Quality: the deploy path already uses MMQ and was blessed on HumanEval+ (§Q).
- Effort spent: low (source read + one compile-flag A/B build + benchmark + web verification). Outcome: the top
  remaining Tier-S engine item closes NEGATIVE (already-captured), with a reusable GEMM-path gate.
- **✅ DOUBLE-CHECKED 2026-08-04 (per user request) — verdict UNCHANGED, materially strengthened, 3 corrections:**
  - **(implementation verified)** FORCE_CUBLAS build has the define active (CMakeCache `=ON` + flags.make + a
    behavioral proof: two binaries from identical source/arch diverge 3-6× on MoE). Arch 86 matches deploy; deploy
    uses `--ubatch-size 2048` so our ub2048 test IS the deploy prefill regime; the FORCE_CUBLAS MoE path is the
    real llama.cpp fallback (a fair comparison, not a strawman).
  - **(generality validated across 5 labs + 4 quants — S1-level breadth; gate `ops/mmq-vs-cublas-generality.sh`)**
    the first close tested only 2 Qwen3.6 Q4_K_M models; the double-check swept the axes that could actually move a
    GEMM-path decision (quant type; dense-vs-MoE; family). Verdict UNANIMOUS: **every MoE keeps MMQ** — Qwen3.6-35B
    Q4_K_M +329% / Q5_K_M +169% / Q6_K +1223%, Gemma-4-26B (q4_0) +67%, gpt-oss-20B +44%, Granite-4.0-H +80%. And
    the dense residual **REPRODUCES on a different family** — Mistral-Small-24B dense shows cuBLAS +4.7% at ub2048,
    same as Qwen dense-27B (+5-11%), proving the "cuBLAS edges large-batch dense" residual is a general GEMM-shape
    property, not a Qwen quirk. cuBLAS-for-MoE gets relatively worse at higher-bit k-quants (Q6_K collapses). Since
    the MMQ↔cuBLAS choice is model-weight-agnostic (keyed on quant/arch/batch/n_experts, NOT routing), this breadth
    fully generalizes the "keep MMQ default" verdict.
  - **(methodology flaw found + FIXED)** the first A/B ran arms back-to-back → GPU heat-soak inflated variance to
    35% CV (dense ub2048 read 1296±456), so my earlier "cuBLAS +5% at 4σ, clean" was OVERCONFIDENT — a lucky
    low-variance run. Isolated arms + cooldown + clock-guard collapse it to 1-3% CV. Honest number: dense
    large-batch cuBLAS wins ~**+5-11%** (wobbles; direction solid). The gate now enforces isolation+cooldown.
  - **(MoE cuBLAS is BROKEN, not just slow — correctness correction)** forcing cuBLAS for MoE OVERFLOWS TO NaN /
    corrupts output / asserts on the RTX 3090 (upstream **#19659**, reproduced on sm_86) and breaks CUDA graphs —
    it's a host-synced per-expert GEMM loop. So "MoE MMQ +284%" understates it: cuBLAS-for-MoE is not a valid
    option at all. Never force cuBLAS for the deploy model.
  - **PR archaeology (incl. rejected, as asked):** the int8-TC branch was NEVER batch-gated (unconditional since
    #8075; the cutoff was always dp4a/Pascal-only). Re-introducing cuBLAS was **REJECTED on precision (#23043)**;
    the original int8-TC prototype was killed for precision too (#4801); maintainers **TRIED and FAILED** to beat
    cuBLAS at large batch (#16512, "things that didn't work…"); NO runtime toggle (compile-time only, #15378
    declined); NO per-shape MMQ↔cuBLAS autotune PR exists; recent low-bit work is all Blackwell-NVFP4/Hopper.
  - **Physics (why no efficacy):** GA102 int8 is only ~**2×** the fp16/fp16-accumulate rate cuBLAS uses (NOT 4×);
    on-the-fly W8A8 quant/dequant overhead eats that 2×; prefill at ub2048 is compute-bound past the roofline
    ridge (batch >~32) where even the best Ampere low-bit kernel (Marlin, PPoPP'25) converges to fp16 parity →
    a negative S2 is the *physically expected* outcome, not a missed opportunity.
  - **WATCH (pin safety):** upstream **#26141** (2026-07-29) added a `smpbo < 48 KiB` guard atop `should_use_mmq`
    that REGRESSES the RTX 3090 → prefill ~1200→~40 t/s (open issue **#26285**). We're pinned to `720d7fa40`
    (pre-#26141; confirmed absent, and our ~1400 t/s prefill proves it). Any future pin bump MUST re-check #26285.
- **→ Tier S fully swept: S1 ✅ (null), S2 ✅ (already-captured), S3 ✅ (mtp-alone optimal). Tier A: A1 ✅
  (windowed-MTP — right paper, unreachable regime; CUT — edge grows to native 256k +176%; 2026-08-04,
  double-checked). A3 ✅ (asymmetric K/V −62% + iq4_nl −79% + sub-4-bit paper-null; KV axis already optimal;
  2026-08-04). A4 ✅ (instrumentation — already-captured; draft acceptance α/τ now in the standing harness,
  probe-validated; 2026-08-04). A2 ✅ (ThinkingCap long-to-short on dense-27B — **STRONG WIN, the first positive
  A-tier lever**: reasoning/wall −53–60% at equal(math)/better(+20pp code) accuracy, Q4 no washout, code did not
  collapse; community LoRA transfer DEAD via fail-fast reconstruction gate; 2026-08-04, `A2_THINKINGCAP.md`).
  **Tier A FULLY SWEPT.** Next: Track H (harness product) or Track M (multimodal), or the A2 follow-on (our-own
  concise 35B-MoE via trace-distillation — the only way to bring the ~2× concision lever to the primary worker).**

### S3. N-gram speculative decoding + drafter-selection policy — §35 / §63.3
- **What:** add n-gram spec-decode (no extra model, wins on repetitive code) alongside MTP; formalize drafter
  priority MTP > n-gram > small drafter; **regression-test** it (doc + community: a mismatched drafter *reduces*
  Qwen3.6 throughput).
- **Why:** best ROI-per-effort on the engine axis — llama.cpp already ships n-gram, so it's config + benchmarking.
- **✅ S3 CLOSED 2026-08-04, then DOUBLE-CHECKED & re-run rigorously 2026-08-04. Keep `--spec-type draft-mtp` alone.**
  The fork ships the full multi-drafter machinery (`--spec-type a,b` → priority fallback chain; hardcoded
  "cheap-first" order = ngram BEFORE mtp; `common/speculative.cpp:2357`). This mirrors **upstream's documented rule**
  (`docs/speculative.md`: *"if a draft model is combined with a draftless decoding the draftless decoding has higher
  precedence"*) — so the ngram-preempts-MTP behavior is upstream-by-design, not a fork bug. No code needed; config only.
- **Rigorous re-test** (deploy model, temp 0 / top_k 1, `enable_thinking:false`, 6 reps/cell, 95% CI, GPU clock 1845–1860
  MHz stable, temp 38–46 °C ~no drift). **Now includes the NO-SPEC FLOOR** (the original test lacked it — it only
  compared drafters to each other, so it could not state each drafter's *sign*). Decode t/s (mean, +/- vs no-spec):
  - **no-spec floor: ~87 t/s** (GEN 87.1, EDIT 88.1, pure-copy 86.6).
  - **`draft-mtp` (deploy default): 132–151 t/s = +53% to +73% over floor** (GEN 150.8, EDIT 149.8, copy 132.2),
    ~92–96% accept, mean accepted run ~4.7 — matches Leviathan E[tok]=(1−α^{γ+1})/(1−α)≈4.4 at α=.94,γ=4. **Winner in
    every regime, incl. pure copy.**
  - **`ngram-simple` alone: net-NEGATIVE except pure copy.** GEN 82.4 (−5%), EDIT 49.8 (−44%, long-but-wrong drafts on
    rename-edits), pure-copy 107.3 (**+24%** — its real §35/PLD niche, confirmed even on our MoE). So n-gram is not
    fundamentally broken here; it just needs ~verbatim copy to pay, which our code workload isn't.
  - **`draft-mtp,ngram-simple` (stack): ALWAYS worse than mtp-alone** — GEN 150.8→131.7, EDIT 149.8→65.3 (below floor!),
    copy 132.2→122.9. The cheap-first ngram draft preempts MTP's better draft → wasted verification. **Zero upside.**
- **CORRECTION to the first pass (exactness was stated backwards):** empirically (sha256, deterministic across reps),
  **`ngram-simple` is greedy-EXACT** (byte-identical to no-spec) while **`draft-mtp` deterministically DIVERGES from
  greedy** (different but stable output; quality-neutral on HumanEval+ separately, but NOT bit-exact). The FORK.md
  "BLESSED token-exact" for draft-mtp means fork==base parity, **not** spec==greedy. Corrected in DEPLOY.md + memory.
- **External corroboration** (all point the same way): upstream **Issue #23184** (closed *not-planned*) independently
  reports *"draft-mtp alone ~78% accept; adding ngram-mod on top = no speedup, only verification overhead"*;
  **thc1006** benchmarked our exact Qwen3.6-35B-A3B on an RTX 3090 (19 configs, *no* spec config beat their no-spec
  baseline); **Spec-Bench** (RTX 3090) puts n-gram/PLD at ~1.6 accepted tok/step vs model drafters ~3.5–4.5; the
  **Leviathan/Chen cost model** predicts a low-α high-γ drafter is strictly wasteful. Papers: PLD, REST, Lookahead,
  SuffixDecoding, CopySpec, "When/What/How" (2511.01282) — n-gram niche = copy-heavy; naive stacking loses.
- **Takeaway:** `--spec-type draft-mtp` alone is optimal for our code workload (MTP is a real ~1.7× win, not just
  "least-bad"). A copy-drafter could only help via a **gated** design (CopySpec / SuffixDecoding: fire only on verbatim
  spans, don't preempt MTP) — the fork's naive cheap-first stacking can't express that, and our workload isn't
  copy-heavy, so it's not worth building. Filed the gated-copy-drafter idea under Tier B (see B-copy). Rigorous
  benchmark banked as the gate: `ops/spec-drafter-bench.sh` (now includes the no-spec floor + CI + 3 regimes).
- Effort spent: low (config + benchmark + verification). Outcome: MTP-alone confirmed optimal AND MTP itself validated
  as a real +70% win; exactness claim corrected; a repeatable drafter-regression gate with proper statistics.

---

## Tier A — high value, moderate effort, our stack

### A1. Windowed-MTP / adaptive speculation — §22 / §35 — **✅ ANSWERED 2026-08-04 (double-checked): right paper, UNREACHABLE regime — CUT.**
- Premise (doc §22/§35): MTP's edge *drops* at long context because the draft pays full-context KV → window the
  draft attention to recover it. **Measured: the edge GROWS with depth to the model's NATIVE 262k ceiling** —
  MoE **+75% @8k → +134% @128k → +176% @256k** (256k: no-spec 32.7 → mtp 90.4 t/s), dense-27B 8k→48k
  **+122%→+133%** (`ops/a1_mtp_depth_bench.py`). Accept stable: T1 (context-indep) byte-identical **99.17% across
  6 depths 8k→256k**; T2 (realistic reasoning, ~50% accept) holds ±2pp with edge **+12%→+41%** at 128k. So the
  growth isn't a high-accept-task artifact, and the llama.cpp accept bugs (#23658 slot-boundary, #23322
  SWA/hybrid-collapse) are ABSENT on our base (720d7fa40, Jul-25, postdates them; we're not SWA:
  `full_attention_interval=4`).
- **CORRECTED (supersedes the first-pass "reversed/null"):** the doc's `windowed-MTP` cite [ref-105] is a **real**
  paper — **arXiv:2607.21535 "Windowed-MTP: Removing the Full-Context Draft-KV Tax _at Million-Token Context_"**
  (NVIDIA, single-author preprint). The draft-KV tax it removes only surfaces at **≥256k** (+27%@261k, +43%@1M),
  "vanishes at short context by construction", **worst on hybrids** (cheap verify exposes the draft's full read).
  We stay ≤128k, where term (A) target-forward amortization dominates → edge grows (Leviathan c-ratio; MagicDec
  1.02×@4k→2.0×@32k; EAGLE-3.1 flat accept; DeepSeek-V3 MTP 85–90%). The doc's error was **regime-misattribution**
  (dropped the "million-token" scope), not fabrication. Also corrected: **windowing the draft is LOSSLESS** (the
  full-attn target verifies every token → window changes only proposals, top-1 unchanged 86–94%), a cost-saver
  not an accept-killer.
- **Disposition: CUT.** The draft-KV tax is +27% on the *draft phase* @261k (a ~1-of-41-layer slice) rising to
  net-negative near ~1M — but we MEASURED native 256k and the edge is at its MAX (+176%), accept pristine, so
  term (A) target-forward amortization swamps the tax everywhere we can serve. The tax only dominates near ~1M,
  **unreachable** (native ceiling 262k; 1M needs YaRN past training where quality is gone) and unusable. So there
  is no context we can serve where windowing the draft helps → moved from "Tier-B watch" to **Cut** (revisit only
  if a future model ships a ≥512k *native* window with a real use case). Corrected mechanism (recorded): windowing
  the draft is **lossless** (target verifies every token, top-1 unchanged 86–94%), not accept-lowering. Verified:
  nextn = full-attn over KV (qwen35moe.cpp); accept metric correct; no upstream windowed-MTP PR (clean negative);
  adaptive-draft-length only an open unanswered discussion (#23738). Gate banked: `ops/a1_mtp_depth_bench.py`.
  Detail: STATUS §A1, EXPERIMENTS §A1.

### A2. ThinkingCap long-to-short LoRA on the **dense 27B** — §23.6 / §56 / §64
- Halving reasoning tokens is a big wall-clock win on our decode-bound setup. ThinkingCap full-FT claims **45.8%
  reasoning-token cut at ~-0.7pp accuracy**; a community rank-64 SVD LoRA exists (**unvalidated**). **Restrict to
  the dense 27B — do NOT apply to the 35B-A3B MoE (shape-incompatible).**
- Cost: BF16 27B (~54GB) SVD extraction → CPU-RAM offload (our 64GB) or cloud; runtime LoRA applies to our GGUF.
- **Validate (doc's §56 protocol, right-sized):** reconstruction gate (base+LoRA ≈ full FT) → λ/rank sweep →
  accept if ≥25% reasoning-token + ≥15% wall-clock reduction on target, quality within ±1pp (ROPE), zero
  tool/JSON/MTP breakage. This is the doc's flagship P1 and the best "combined model+engine" experiment for us.
- **✅ A2 CLOSED 2026-08-04 — STRONG WIN (the first positive A-tier lever); full record `A2_THINKINGCAP.md`,
  STATUS §A2.** Direct paired A/B of the FULL ThinkingCap GGUF (we already had it on disk — no LoRA/SVD needed
  for the win; §23.6's "Transfer Lab" machinery is only for transferring to OTHER fine-tunes) vs base
  `qwen36-27b-dense`, both Q4_K_M matched-imatrix, decode-pure (spec OFF — draft-mtp is not bit-exact on qwen35,
  would corrupt the token count), `--reasoning-format deepseek` + `/tokenize` for exact reasoning-token counts.
  **GSM8K n=60: reasoning −59.9% [50.1,64.8] (p=1.8e-11), wall −55.3% (33→17.8s); accuracy-NEUTRAL** (both-answered
  100→98%, 1 reg — the +18pp overall is ALL starvation recovery; base starved 13/60 at the 4096 budget, cap 0).
  **HumanEval+ n=40: reasoning −53.0% (p=5.5e-8), wall −51.0% (70→32s); pass@1 GENUINELY +20pp** (both-answered
  70→90%, 0 regressions, base-only-right=0 everywhere) + recovers 8/10 base starvations. **Q4 does NOT wash out
  the concision; code did NOT collapse to ~8%** — the base is an extreme over-thinker (2705-tok code reasoning,
  starves 20-25%) so headroom is huge. Difficulty split: cut GROWS with hardness (up to 77%). Zero short-but-wrong
  (≤1 regression / 100 problems). **Deploy: ThinkingCap replaces base in the dense-27B slot** (~2× faster at
  equal/better accuracy, keeps MTP head). **Scope: dense-27B ONLY** (shape-incompatible with the 35B-A3B MoE worker;
  a concise MoE would need our OWN trace-distillation training — the top A2 follow-on).
- **✅ COMMUNITY LoRA / DavidAU transfer (T2/T3) CLOSED-NEGATIVE 2026-08-04 via fail-fast (n=12 pilot, ~25 min).**
  The rank-64 SVD LoRA FAILS the reconstruction gate on its OWN origin base: base+LoRA@1.0 reasoning 968 vs cap 502
  (ratio 1.93 — recovers ~15% of the concision), fidelity sim(B,cap)=0.26 << sim(B,base)=0.65 (behaves like base,
  not ThinkingCap). Concision lives OUTSIDE the rank-64 subspace (2503.20641 "SVD limited" / 2410.21228 intruder
  dims). DavidAU-Fable-Fusion geometry was fine (stock 64-layer + MTP, GGUF on disk) — the ADAPTER is the problem,
  not the target, so transfer is meaningless. Only a better extraction (TIES / rank-128-256 / Fisher-weighted) or
  real trace-distillation could revive it → **Tier C, low priority** (gate multi-LoRA §73 on it). Fail-fast killed
  a Frágil multi-hour experiment with a 25-min pilot.
- **A2 FOLLOW-ON — `uTC` via ABLITERATION then task-arith merge (Tier C, FILED 2026-08-05, user idea).**
  Instead of the "fable-delta boost" `W=fable+λ(TC−base)+μ(fable−base)` (algebraically IDENTICAL to a *task-arith* uTC, and
  it over-drives the DavidAU delta → prose-degeneration risk), produce an ABLITERATED uncensored TC and merge
  `W=fable+λ(uTC−base)`. **Only the abliteration path is genuinely different** — non-linear/orthogonal projection, amplifies
  nothing; a *linear* uTC collapses back to the boost. **Method (Arditi et al. 2024, "refusal is a single direction"):** ~256
  refused + ~256 complied prompts → capture residual-stream activations (load TC in 4-bit, forward-only) → per-layer
  difference-of-means refusal direction r̂ → sweep/select best layer → orthogonalize every residual-WRITING matrix
  `W'=W−r̂(r̂ᵀW)` (attn `o_proj` + mlp `down_proj`, streamed shard-wise on fp16 like `a2_merge_raw.py`) → requant Q4_K_M →
  **fail-fast: GSM8K n=12 concision must survive + refusal drops + coherence.** **Cost on THIS box: ~1–1.5h COMPUTE
  (forward-only, ZERO training); deps ALL present in `sglang-venv` (torch 2.11+cu130, transformers 5.12, bitsandbytes 0.50,
  accelerate 1.14); fp16 TC on disk (52G); ~70G transient disk; 4-bit fits 24G VRAM; fp16 weight-edit streams on 64G RAM.
  Real cost = ~half-day writing the extraction+ortho harness (~250 LoC), NOT GPU.** **CAVEAT: TC is only mildly aligned
  (1–5/8 on the meta tier) → weak refusal signal to contrast → derive the direction from the BASE (dense, aligned, same arch
  — direction transfers) or a broader refusal set, else uTC comes out half-abliterated.** Cheaper variant: abliterate `l1.0`
  DIRECTLY (kills its residual 1/8 hedge in one op; non-modular, non-reusable). **TRIGGER: only for a purist/reusable
  uncensored-concision artifact, or to scrub l1.0's residual hedge — Gate-2 (2026-08-05) showed the plain merge ALREADY
  preserves uncensored (l1.0 balk 1/8 == fable-plain, think-deliberation 0), so this fixes a problem that did not manifest.**
  Not a deploy blocker.

### A3. Asymmetric K/V quant + SAW-INT4 KV — §62
- We're already ahead (q4_0 KV lossless); next headroom = quantize K and V asymmetrically, or **SAW-INT4**
  (Hadamard rotation + token-wise INT4, explicitly designed for low serving-integration cost). Extends usable
  context/residency. **Validate:** needle + code-suite at 8k/32k/64k/128k vs q4_0; accept if context headroom up
  and no code/tool regression. (Sub-2-bit KVarN/OSCAR = Tier C watchlist; TurboQuant/UltraQuant = cut, AMD-only.)
- **✅ A3 CLOSED 2026-08-04, DOUBLE-CHECKED — NEGATIVE / already-optimal (full record `A3_KV_QUANT.md`; STATUS §A3;
  gate `ops/kv-quant-bench.sh`).** Robust decode @8k (deploy MoE, ncmoe=8, -fa on, base 720d7fa40; 6 reps/arm,
  isolated + cooldown, 95% CI): **q4_0/q4_0 = 88.6 [87.7,89.4]** (lossless); **q8_0/q8_0 = 89.8** (≈, lossless, more
  VRAM); **asym q8/q4 = 38.4 [27.2,49.6] (−57%)**; **iq4_nl = 16.1 [15.5,16.7] (−82%)**. **Mechanism (source-verified
  `ggml-cuda/fattn.cu`):** the default build compiles only 4 SYMMETRIC FA KV combos; K≠V or an unwhitelisted type →
  `BEST_FATTN_KERNEL_NONE` → attention op offloaded to CPU (CPU KV buffer; #20866 ~156 MiB). Build-flag-gated
  (`GGML_CUDA_FA_ALL_QUANTS=OFF`); ON puts asymmetric on-GPU (#20866 ~25× recovery) but still dominated (q4 lossless →
  more VRAM, zero quality). **iq4_nl has no FA kernel on ANY arch, flag or not.** Corroboration: Gäßler #7527 (*"KV
  quant is a memory feature, not a speed feature"*), am17an #22411, FR #24485 closed not_planned. **CORRECTION:
  SAW-INT4 (arXiv:2604.19157) is a 4-BIT method (H100/Triton/FA3/SGLang), NOT sub-4-bit** — value is quality-at-4-bit,
  null since q4 lossless here (QK-Norm kills the outliers INT4 fights). **Sub-4-bit (TurboQuant #20969/KVarN/OSCAR) not
  upstream** and dominated anyway: frees < 1 ncmoe step @128k, buys zero context (q4 reaches native 262k), and batch-1
  KV is ~3% of bytes moved (arXiv:2605.30571) → wall-clock-invisible on Ampere. Free monitor→iGPU replug (~1.4 GB)
  dominates. **Dense-27B** blocked by the full-precision GDN recurrent state (Phase A #3), untouched by `--cache-type`.
  Same pattern as S1/S2/S3/A1.

### A4. Spec-decode + benchmark instrumentation discipline — §22 / §40 / §63.4 ✅ DONE 2026-08-04 (deep-dived pre-implementation)
- **Verdict: an INSTRUMENTATION item, LARGELY ALREADY-CAPTURED (S2 pattern).** Full record `A4_INSTRUMENTATION.md`;
  STATUS §A4; gate `ops/a4_spec_metrics_probe.py` (**A4 PROBE OK**). The one machine-readable §63.4 metric missing
  from the *standing* A/B harness was **draft acceptance** — now wired in (`request.py` α=`draft_n_accepted/draft_n`
  + `tpot_ms`; `throughput.py` aggregates; `ab_isolate` SPEC-DECODE block prints α, τ=1+γα, TPOT, gen), source- and
  empirically-validated (α==server `draft_ratio` to 5dp; τ==logged `mean_len` 3.79). TTFT/wall-clock/VRAM/quality
  (§Q)/distribution-divergence (§Q,A1,S3) already existed.
- **CORRECTION: the "our levers map ~1:1 [to §40]" claim above was FALSE.** Our L18 (`taguchi_screen.py`) screens
  BUILD knobs (pin/prefetch/ubatch/ncmoe/kv/cache); §40's matrix is WORKLOAD-facing
  (speculation/context/deterministic/engine/concurrency) — overlap is only kv+ncmoe. We have *a* fractional-factorial
  screen with correct discipline, but it is NOT the §40 matrix and has no `speculation` factor (the spec lever is
  measured by the dedicated paired `e4mtp` A/B + S3 instead).
- **Caveats banked:** batch-1 α/τ/speedup are an UPPER BOUND (batch-collapse arXiv:2601.11580 = our §CC); `/metrics`
  has no spec metric (#25327 closed-no-merge) so we read per-request `timings`; ⚠ #26320 (post-base) fixes
  checkpoint-restore acceptance inflation (+1/restore) → pin-watch; #26100 repeated-prompt inflation defended by
  our `cache_prompt=False`. Deliberately NOT built: drafter-time-vs-saved + per-pos acceptance (stderr-log-only, low
  batch-1 ROI), ITL p50/p95/p99 (needs `timings_per_token`; mean TPOT suffices at CV~0.006).

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

- Multi-LoRA library + router (§73) — DORMANT: A2 showed the only community adapter (rank-64 SVD) doesn't even
  reconstruct its own FT, so we have 0 useful adapters, not ≥2. Revisit only if we train our own (TIES/distill).
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

## Track M — Multimodal (in scope as of 2026-08-04: **both** VLM + image generation)

**Grounding:** no multimodal models on disk yet, but **`libmtmd.so` is already built in our fork**
(`tools/mtmd/` present) — so VLM serving is a short hop, not a new engine.

### M-A · VLM (vision understanding) — in our llama.cpp engine [do first, higher ROI]
- **Goal:** a coding-agent-that-sees — reads screenshots, error dialogs, diagrams, UI mockups.
- **Candidate:** a Qwen2.5-VL / Qwen3-VL-class model (same family as our text worker) — 7B for headroom or a
  32B quantized to share 24GB with a text KV budget; Gemma-3-vision / InternVL / MiniCPM-V as alternates
  (verify current llama.cpp mmproj support per model).
- **M0 (baseline, cheap):** build the `mtmd` target, fetch VLM GGUF + mmproj, serve via llama-server, test
  image→text. Accept: correct OCR/description of an error dialog + a UI mockup; VRAM within envelope.
- **§71 efficiency levers (only after M0, once a repeated-image workload exists):** visual-embedding cache by
  content-hash + encoder config; separate mmproj VRAM budget from text KV (don't let the encoder starve decode);
  offload the vision encoder when a session goes text-only; **measure visual TTFT vs text TTFT separately**
  (vision-encode is compute-bound, our decode is bandwidth-bound → different resource profiles → schedule apart).
- Effort: M0 low-moderate (library already built); levers moderate. Stays in-fork.

### M-B · Image generation (diffusion) — separate engine [new front]
- **Engine:** ComfyUI / diffusers (torch), **NOT llama.cpp** — a genuinely separate stack + maintenance surface.
- **Candidates on 24GB:** SDXL (comfortable); Flux.1-schnell/dev (12B → needs fp8/GGUF quant + CPU offload to fit).
- **§72 levers, ROI-ordered:** step reduction first (LCM / Turbo / schnell) → feature caching (DeepCache / TeaCache
  reuse across denoising steps) → quant (rotation-aware 4-bit DiT; **same §61 INT8-Tensor-Core lesson — W8A8 only
  helps if the kernel actually uses INT8 TC**, a direct conceptual synergy with S2) → mixed precision (HyperQuant).
- **Gates:** PickScore/CLIP + blind human eval + latency + VRAM. Video real-time stays cut (non-Ampere).
- Effort: higher (new stack, new deps, model downloads).

> **Synergy:** the INT8-Tensor-Core kernel discipline (S2, §61) is the *same* principle that governs diffusion
> quant speed on the 3090 — one kernel lesson, two domains. But M-A and M-B are separate engines: M-A extends
> our fork, M-B is a parallel ComfyUI/diffusers track.

---

## Cut (out of scope / HW-blocked / redundant — recorded so we don't re-litigate)

- **HW-blocked:** FP4/NVFP4 (§61.2, Blackwell); FA3 (§37, Hopper — *useful negative signal: don't chase it*);
  TurboQuant/UltraQuant KV (§34/§62, AMD CDNA4 numbers, non-transferable); disaggregated prefill/decode &
  tensor-parallel (§36/§74, multi-GPU); CPU-FP8 expert kernels (§66, needs AMX we lack).
- **Out of scope:** real-time **video** generation (§72, explicitly non-Ampere — offline image gen is now in
  Track M-B); distributed/home-cluster inference (§74, needs a 2nd machine — only agent-level parallelism if a
  laptop appears). *(Multimodal understanding §71 + image generation §72 moved INTO Track M above.)*
- **Needs cloud / disproportionate:** full RL training of the 35B (§24); formal-verification tiers F3/F4 (§13).
- **Redundant with what we have:** §30 phase framework, §38 MoE-pinning checklist, §40 matrix (we have the lab),
  E5 KV-q4 (done, lossless), E6 MTP (done), §67 hybrid direction (we already run one).

## Doc's independent validation of our choices (confidence signals)
§29 model portfolio ≈ our exact lineup · §37 confirms CUDA graphs + skip FA3 · §61.2 confirms our Ampere/TF32
limits · §42/§75 put "expert pinning / custom KV kernels" at P2-P3 — **we already built them**, i.e. our fork is
ahead of the doc's assumed baseline on the engine axis.
