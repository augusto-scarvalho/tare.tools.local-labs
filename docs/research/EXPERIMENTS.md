# EXPERIMENTS — methods we use, and the pre-registered plans

This project runs **paired 2-arm performance A/Bs on GPU inference** (metric: tokens/s,
a deterministic oracle). That is a different world from the LLM-multi-agent harness whose
methods library seeded this note (`EXPERIMENT_METHODS.md` / `EXPERIMENT_METHODOLOGY.md`,
owner's other project). Most of that library — judge bias, oracle recall, matched-**budget**
(compute) controls, contextual bandits, route churn, the experiment-lifecycle state machine —
is for stochastic LLM-in-the-loop evaluation and **does not apply here**. This note keeps only
what does, maps it to code we already have, and pre-registers the open experiments.

---

## 1. What we already have (mapped to the reference library)

Our `src/model_lifecycle/analysis/robust.py` + `reports/ab.py` + the noise-floor discipline
already implement most of what a clean perf-A/B needs:

| Reference card | Our implementation | Status |
|---|---|---|
| **Noise floor** | `ab-null` A/B → median\|%\| of a true-zero delta; "below the floor = not evidence" | ✓ core to the project |
| Distribution-free inference | `sign_test_p` — exact two-sided sign test (floor 0.031 at n=6) | ✓ |
| Bootstrap CIs | `bootstrap_ci` — seeded percentile bootstrap | ✓ |
| Robust location / spread / effect size | `hodges_lehmann`, `mad` (1.4826-scaled), `cliffs_delta` | ✓ |
| MDE / power (min detectable effect) | `min_rounds_for(delta, cv, α, power)` | ✓ |
| Taguchi **companion** (variance attribution) | `anova_contributions` | ✓ — but see §2 |
| Randomization / blocks / replicates | arm-order flips each round; warm-up discarded; pair by `(round, dose)` | ✓ in `ab_isolate` |
| Matched control (the *analog* that fits us) | same binary in both arms; only the switch differs (`genpin`) | ✓ |
| Split-plot | batch all rounds of one model (hard-to-change = the build) before switching | ✓ implicit |
| Confirmatory: discover then confirm | effect discovered on qwen36; now replicated across architectures | ✓ in practice |
| **Spec-decode metrics (A4/§63.4)** | `request.py` α=`draft_n_accepted/draft_n` + `tpot_ms`; `ab_isolate` spec block adds τ=1+γα; gate `ops/a4_spec_metrics_probe.py` | ✓ 2026-08-04 (see `A4_INSTRUMENTATION.md`) |

So for clean paired 2-arm contrasts we are well-covered. The gaps are only for the **new
question shapes** the KV/agentic work raises (dose-response, quality non-inferiority).

**Explicitly NOT adopted** (their domain, not ours): judge-bias audits, oracle-recall / `Δ_m`,
matched-**budget** controls, contextual bandits, route churn / Π-lite / recovery probes, and the
`proposed→active→shipped/shelved` CLI state machine. We take the *discipline* (pre-registration,
noise floor, evidence grades) without the multi-agent machinery.

---

## 2. Taguchi — why we are careful (the L18 burn)

We have a **negative history** with orthogonal-array Taguchi here. The L18 screen once measured
a fork whose three feature gates were **all closed by construction**; its `cache` factor was
inert (disabled unless a profile it never had was supplied) and scored 0.1% of variance. The
array ran clean and told us nothing true.

**Lesson, kept:** an orthogonal array is only valid if **every factor actually moves the
response**. A default-closed or inert factor does not average out — it pollutes the whole array,
and the tidy marginal-means table launders noise into a result. For the pinning experiments the
right tool is **not** an L18/Plackett-Burman screen — they are clean **2-arm paired contrasts**,
and the paired `sign_test_p` + `bootstrap_ci` we already use is the honest instrument.

**What we keep from Taguchi:** `anova_contributions` for *post-hoc* variance attribution — but
only on a design whose factors are **verified live** (the switch actually toggled the code path),
never as a screening front-end for an unverified switch. Robust-loss `R = E[L] + κ·√Var` and
signal-to-noise ratios are **not** used; our objective is a paired delta clearing the floor, not
flatness across noise regimes.

---

## 3. Worth implementing (small, high-value, GPU-free)

Four additions, in priority order. All live in `robust.py` and are unit-testable offline.

1. **Dose-response / monotonic-trend test — the important one.** §B1 and §B2 are both
   dose-responses, not single points:
   - §B1: pinning's *generation* delta vs **active experts per token** (4→6→8→10 across models).
   - §B2: pinning's *KV* delta vs **context length** (8k→32k→128k).
   A single positive point is weak; a delta that **rises monotonically with the mechanism's
   driver** is strong evidence the mechanism is real. Implement a trend test on the ordered
   per-dose paired deltas: bootstrap CI on the regression slope of `delta ~ dose`, plus a
   distribution-free increments sign test (Jonckheere-style). This is the genuinely new stat.

2. **Non-inferiority test** for the **quality** axis. The KV-quant A/B (q4 vs q8 KV) trades
   quality for speed/RAM; we must show q4 is *not meaningfully worse* on quality within a
   pre-declared margin `δ_Q`, one-sided — not that it is better. Small function; margin fixed
   before data (default `δ_Q = 0.2 SD` or 5 pp, per the reference §6.6 constants).

3. **Evidence grades 1–4** as a tag in STATUS: exploratory → attributive → confirmatory →
   promotion-qualified. §B1 is **grade 3** today (confirmatory: paired, floor-checked,
   pre-registered) and reaches **grade 4** only when the cross-architecture replication holds.
   A grade is **demoted** by a build/model change — evidence for the old build is not automatic
   for the new one.

4. *(Optional)* **Confidence sequence** for anytime-valid early stopping — settle at n=6 when the
   sequence already excludes the floor, spend n=12 only when it does not. Saves GPU across an
   8-model campaign. Nice-to-have, not required.

---

## 4. Pre-registration discipline (adopted, lightweight)

Every experiment below is a card with six fields written **before** any measurement, committed
to git (our version of their pre-registration — the commit timestamp is the pre-registration):
**hypothesis · metric · baseline · successCriteria · abandonCriteria · reversalPlan**. Factor
typing (control / noise_context / hard_to_change / nuisance / prohibited) is named per card.
Decision constants inherited from the reference §6.6 unless a card justifies otherwise:
α = 0.05, power ≥ 0.80, `δ_Q` = 0.2 SD / 5 pp. **Tightening is free; loosening needs a
pre-registered reason.**

---

## 5. §B1 — synthesis of the running campaign (dose-response across architectures)

**hypothesis** — Pinning (`GGML_CUDA_REGISTER_HOST`) speeds *generation* only when generation is
transfer-bound, and the effect **scales with active experts per token** (the per-token PCIe
transfer volume). Near-resident → null; more active experts → larger delta.
**metric** — `gen_tps`, paired `pin − base` by `(round, model)`, `genpin` arm-set, ncmoe = max
offload, n=12, ctx 8192.
**baseline** — the `base` (unpinned) arm; and the gen noise floor from the near-resident regime
(~0.2%, well under 1%).
**successCriteria** — a **positive monotonic trend** of the paired delta vs active-experts
(trend-test slope CI > 0), with the high-active models clearing the floor.
**abandonCriteria** — deltas flat/within-floor across all active-expert counts → pinning does not
help generation; §B1 stands only for prefill.
**factors** — control: pinning switch. noise_context: **architecture** (5 distinct arches — a
*blocking* factor, never pooled raw; pair within arch). hard_to_change: the model/build
(whole-plot; one clean baseline per model via `wsl --shutdown`). nuisance: cold-start round
(handled by warm-up discard + arm-order flip).
**reversalPlan** — measurement-only (env var toggled); nothing to revert.
**analysis** — per-arch paired `sign_test_p` + `bootstrap_ci` + `cliffs_delta` (have it), then
the new **trend test** of delta vs active-experts. Data so far: gpt-oss (4 active) **≈ 0**
(−0.32%, p=1.0); qwen36 (8) **+2.14%** (p=0.039); Granite (10), Gemma-4 (8), Ernie (6) **now DONE**
(`runs/ab-genpin-*`). Dense controls (Mistral, Qwen3.6-27B, ThinkingCap: 0 experts) read **null** — the
negative control proving the mechanism is expert-streaming, not something generic to offload. **RESOLVED:
transfer-bound-only, active-count-scaled, ~2% ceiling on this hardware (STATUS).**
**evidenceGrade** — **4 (DONE):** the trend held across all 5 arches + the 3 dense nulls read null.

---

## 6. §B2 — pre-registered card: pinning the KV cache in the KV-in-RAM regime

**hypothesis** — When the KV cache lives in system RAM (`--no-kv-offload`), each generated token
transfers the KV history over PCIe — a **second transfer-bound regime**, twin of §B1's expert
streaming. Pinning the *KV host buffer* (extend `register_host` to it; it is pageable today)
speeds per-token generation, and the delta **grows with context length** (more KV to move).
**metric** — `gen_tps`, paired `kvpin − base` by `(round, context_length)`.
**baseline** — `base` (pageable KV) at each context; and a **KV-null** A/B (same config both arms)
for the gen noise floor at each context.
**successCriteria** — the paired delta clears the floor AND rises monotonically with context
(trend-test slope CI > 0) — the dose-response that mirrors §B1.
**abandonCriteria** — delta within floor at all contexts → the §B1 pinning mechanism does **not**
generalize to KV streaming; record and stop.
**factors** — control: KV-pin switch. noise_context: context length {8k, 32k, 128k} (the **dose**).
hard_to_change: the patched binary (whole-plot). prohibited: the safety envelope (unchanged).
**reversalPlan** — the KV-pin is behind an env flag on a patched build; unset the flag / use the
stock binary. Patch is measurement-only (a `cudaHostRegister` on an existing buffer), no behavior
change to the served model.
**dependencies** — (a) verify the KV CPU buffer is pageable and locate its allocation; (b) small
patch extending host-register to it, gated by an env var (so both arms use one binary, per §1's
matched-control idiom); (c) the agentic 128k setup from `[[agentic-local-model-plan]]` as the
substrate. Runs AFTER the §B1 campaign (a rebuild + loads use the GPU).
**companion A/Bs on the same substrate** (cheap, flags only): KV-quant **q8 vs q4** (speed × RAM
× quality, judged by non-inferiority §3.2); flash-attn on/off; `--no-kv-offload` vs partial-layer
offload. And the wildcard: **MTP / nextn** speculative decode (the build loads the tensors; does
it *use* them?) — if yes, it amortizes KV reads and is the biggest lever in this regime.
**evidenceGrade** — target 3 (confirmatory) on first clean run; 4 if replicated on a second model.

---

---

## 7. Tier-1 — the speed pivot: engine & placement experiments (pre-registered)

The highest-payoff decode-speed levers (see `LANDSCAPE.md` §5) — run AFTER the §B1 campaign
closes. The headline reframe: our fork optimizes philosophy-(a) *stream experts to GPU*; the rival
engines use (b) *compute experts on CPU*. The largest untested gains are at the engine level, not
in tuning our fork. Each card's **engine/build is a hard-to-change whole-plot factor**; hold the
GGUF, expert placement, context, and prompt fixed across arms. Metric everywhere is `gen_tps`
(decode), floor-checked; `prompt_tps` reported alongside.

### §E1 — expert placement + KV format (stock llama.cpp, FREE, do first) — **ANSWERED 2026-08-01**
- **result** — CONFIRMED and then some. qwen36-35B-Q4 decode **27.6 t/s (ncmoe=40, our campaign
  placement) → 101.7 t/s (ncmoe=6) = +268% (3.7×)**, respecting the 4 GB VRAM reserve; ncmoe=4
  (~113 t/s) breaks it. **KV q4_0 vs q8_0 = null at 8 k ctx** (±2%, frees ~46 MB) — long-context
  lever only. Pin is null at the optimum (win is placement, not the fork). New stock baseline for
  §E2–§E4: **~102 t/s at ncmoe=6**. Full write-up: STATUS.md §E1; `[[placement-is-the-decode-lever]]`.
- **hypothesis** — decode t/s is maximized by keeping as many expert layers on the GPU as fit
  before VRAM spills (our runs used *max* offload, likely past the optimum), and by a KV format
  that stays on Ampere's flash-attention fast path.
- **metric/baseline** — `gen_tps` vs `ncmoe` (dose) at fixed model/ctx; baseline = our current
  max-offload number per model. **success** — an `ncmoe` below max with higher decode t/s, VRAM
  not spilling. **abandon** — max already optimal.
- **factors** — control: `ncmoe`, and `-ctk/-ctv` {q8_0, q4_0}. nuisance: cold start (warm-up).
- **how** — reuse `ab_isolate`'s dose axis: sweep `ncmoe` from smallest-that-fits up to max, plus
  a KV-format arm. No fork, no build — the cheapest Tier-1 win, and it recalibrates every later A/B.

### §E2 — ik_llama.cpp vs our stock (philosophy (b) vs (a)) — **ANSWERED 2026-08-02**
- **result** — at the operating point (ncmoe=6, n=4 clean): decode **TIE** (+0.29%, within noise),
  ik **+75% prefill**. Swept ncmoe: ik's decode edge grows with offload (+2%→+5.3%→+9.6% at
  6→16→28) — (b) degrades less than (a) — but ik is unusable at that offload on 64 GB: `-rtr` OOMs,
  RSS breaches the 16 GB reserve at ncmoe16+, ncmoe40 gen crashes. **Verdict: no decode win where we
  run; RAM-unsafe where it would win. Revisit at 128 GB + a too-big-to-place model.** Full write-up:
  STATUS.md §E2; `[[ik-ties-stock-decode]]`.
- **hypothesis** — for offloaded-MoE decode on the 3090, `ik_llama.cpp` (`-fmoe` fused MoE, `-rtr`
  run-time-repack, `-ser` smart-expert-reduction, fast CPU GEMM) beats our stream-to-GPU path by a
  margin worth the engine swap.
- **metric/baseline** — `gen_tps`, matched GGUF + expert placement; baseline = best §E1 stock
  config; the engine is the arm. **success** — ik decode clears the floor above stock by a swap-
  worthy margin. **abandon** — ik ≤ stock (our transfer-optimized path already wins here).
- **factors** — control: engine (whole-plot). held: GGUF, placement, ctx, prompt.
- **dependencies** — build `ikawrakow/ik_llama.cpp` (CPU compile — do it while the GPU is idle,
  NOT during a running measurement); optional `_R4`/`IQ*_K` re-quant as a second arm for ik's full
  edge. **Single most important Tier-1 result.**

### §E3 — KTransformers vs the §E2 winner
- **hypothesis** — `kvcache-ai/ktransformers` (Marlin GPU-expert + AMX CPU kernels + CUDA-graph
  decode; purpose-built for consumer-GPU MoE offload) beats both llama.cpp variants.
- **metric/baseline** — `gen_tps`, matched model/quant; baseline = §E2 winner; engine is the arm.
- **caveat** — KTransformers has its own quant/format conventions; "matched" is approximate —
  report the config gap honestly (a lane gap, never smoothed over). **dependency** — install the
  framework (Python, heavier); confirm it serves our model class.

### §E4 — MTP / speculative decode (amortize the movement) — **ANSWERED (MoE) 2026-08-02**
- **result** — CONFIRMED on qwen36-35B-A3B MoE. `draft-mtp` is EXACT (token-identical, verified) at
  **80.5% accept** (194/241) — matches upstream's ~82%. Decode **+26.75%** at matched placement
  (ncmoe=8, n=4, Cliff +1.0), **+54%** on structured code (greedy 139.6 vs 90.4 t/s), +16% deployable
  at the optimum (mtp holds ~116 t/s at ncmoe 6 AND 8; at ncmoe=6 it breaks the 4 GB VRAM reserve —
  draft context ~1.15 GB). It **decouples decode from placement** and stacks on §E1. Turn ON for the
  agentic deployment. **27B Gated Delta Net hybrid (dense) also DONE: +49.4% (bench) / +83% (code),
  73.6% accept — the biggest uplift, because a dense forward pass is costliest per token (payoff scales
  with forward-pass cost, not accept rate). Fused delta-net kernel disabled by this build (non-fatal).**
  STATUS §E4.
- **hypothesis** — running the MTP/`nextn` head as a draft (`--spec-type draft-mtp`) amortizes
  weight *and* KV movement → higher decode t/s in the memory-bound regime, scaling with accept rate.
- **metric/baseline** — `gen_tps` AND accept rate, draft-mtp on vs off. **success** — decode t/s
  up beyond the floor at an accept rate near upstream's ~82%; spec-decode is exact by construction,
  so **verify token-identical** output (quality-neutral). **abandon** — low accept / no gain here.
- **factors** — control: spec on/off. held: model, ctx.
- **dependency** — a model whose MTP head llama.cpp actually *uses* (the DavidAU Fable-Fusion has
  MTP; gpt-oss has EAGLE3). Ties directly to `[[agentic-local-model-plan]]`. **Target to beat:
  upstream #25642 — +30% t/s, ~82% accept.** Biggest single-stream lever for the long-context agent.

### §A2 — ThinkingCap long-to-short on the dense-27B — **ANSWERED 2026-08-04: STRONG WIN (both axes); LoRA transfer DEAD**
- **hypothesis** — a long-to-short concision fine-tune (ThinkingCap-Qwen3.6-27B) cuts reasoning tokens ~46% at
  ~flat accuracy, which on our decode-bound box is a ~2× wall-clock win. Sub-question: does the community rank-64
  SVD LoRA reconstruct the full FT (→ transferable to other 27B fine-tunes like DavidAU-Fable-Fusion)?
- **design** — paired A/B, full ThinkingCap GGUF vs base `qwen36-27b-dense`, both Q4_K_M matched-imatrix, decode
  PURE (spec OFF — draft-mtp isn't bit-exact on qwen35, would corrupt the token count), `--reasoning-format deepseek`
  + `/tokenize` for EXACT reasoning-token counts, greedy temp 0, seeded NESTED subset (pilot ⊂ full). Metrics:
  paired reasoning-token reduction (Wilcoxon + bootstrap CI), total tokens, wall-clock, accuracy (McNemar; GSM8K
  numeric, HumanEval+ evalplus pass@1), difficulty split, starvation/short-but-wrong guards. **Fail-fast**: n=12
  pilot before any escalation.
- **result** — **GSM8K n=60:** reasoning −59.9% [50.1,64.8] p=1.8e-11, wall −55.3% (33→17.8s), accuracy-NEUTRAL
  (both-answered 100→98%; +18pp overall is all starvation recovery). **HumanEval+ n=40:** reasoning −53.0% p=5.5e-8,
  wall −51.0% (70→32s), pass@1 GENUINELY +20pp (both-answered 70→90%, 0 regressions). Q4 no washout; code did not
  collapse to ~8% (base over-thinks → huge headroom); cut grows with difficulty; ≤1 short-but-wrong / 100.
  **Reconstruction gate FAILED** (rank-64 SVD: len ratio 1.93, fidelity 0.26 vs 0.65 to base) → LoRA/DavidAU
  transfer closed-negative. **evidenceGrade 4** (n=60/40 paired, exact token counts, evalplus pass@1, both-answered
  starvation control, stats cross-validated vs scipy). Full record `A2_THINKINGCAP.md`; STATUS §A2; raw `runs/a2/`.
- **disposition** — DEPLOY ThinkingCap in the dense-27B slot (~2× faster, equal/better acc, keeps MTP). Follow-on:
  our-own concise 35B-MoE via trace-distillation (the only way to bring the lever to the primary worker).

### §A3 — asymmetric K/V + better/sub-4-bit KV quant — **ANSWERED 2026-08-04 (double-checked): CLOSED, negative — KV axis already optimal**
- **hypothesis** — beat the deployed symmetric q4_0 KV via (a) asymmetric K/V, or (b) a fancier codec
  (iq4_nl; SAW-INT4; sub-4-bit TurboQuant/KVarN if engine-added), to extend usable context/residency.
- **result (robust: 6 reps/arm, isolated process + 25s cooldown, 95% CI; deploy MoE Q4_K_M, ncmoe=8, -fa on, base
  720d7fa40)** — **q4_0/q4_0 = 88.6 [87.7,89.4]** (baseline, lossless per §Q) · **q8_0/q8_0 = 89.8 [86.3,93.3]** (≈,
  lossless, more VRAM) · **q8_0/q4_0 = 38.4 [27.2,49.6] (−57%)** · **iq4_nl = 16.1 [15.5,16.7] (−82%)** · q4_0@32k =
  76.4 (graceful). First pass (3 reps, no cooldown) had same signs; robust pass fixed iq4_nl CV 24%→3.4%.
  abandonCriteria (no config beats symmetric q4_0 on decode/context) MET on every arm.
- **verdict** — (1+2) **Asymmetric K/V and iq4_nl offload the FA op to CPU** — source-verified in `ggml-cuda/fattn.cu`:
  default build compiles only 4 symmetric FA KV combos; K≠V or an unwhitelisted type → `BEST_FATTN_KERNEL_NONE` → CPU
  (CPU KV buffer; #20866 ~156 MiB). Build-flag-gated (`GGML_CUDA_FA_ALL_QUANTS=OFF`); ON puts asymmetric on-GPU but
  still dominated (q4 lossless → more VRAM, zero quality). iq4_nl has no FA kernel on ANY arch. (3) **SAW-INT4 is 4-bit
  not sub-4-bit** (arXiv:2604.19157; H100/Triton/FA3/SGLang) — quality-at-4-bit, null for us. **Sub-4-bit TurboQuant/
  KVarN not upstream** and dominated: frees < 1 ncmoe step @128k, zero context (q4 reaches native 262k); batch-1 KV is
  ~3% of bytes (arXiv:2605.30571) → wall-clock-invisible on Ampere. (4) **Dense-27B** blocked by the full-precision GDN
  recurrent state (Phase A #3). **Deploy: keep symmetric q4_0.** Corroboration: #20866, #7527 (Gäßler: KV quant is a
  memory feature not a speed feature), #22411, #24485(not_planned); KIVI/KVQuant need custom kernels; Marlin (weight
  axis is the lever). Full record `A3_KV_QUANT.md`; gate `ops/kv-quant-bench.sh`; raw `runs/context/a3-kv-quant/`.
  **evidenceGrade 5** (robust CIs on all engine arms + source mechanism + upstream same-GPU corroboration + literature;
  sub-4-bit is not-in-engine + physically dominated, no runnable gap remains). Same pattern as S1/S2/S3/A1.

### §A1 — windowed / adaptive MTP on the GDN path — **ANSWERED 2026-08-04 (double-checked): right paper, unreachable regime — CUT**
- **result** — The MTP decode-t/s edge **grows** with depth all the way to the model's NATIVE 262k ceiling: MoE
  **+75% @8k → +134% @128k → +176% @256k** (256k: no-spec 32.7 → mtp 90.4), dense-27B 8k→48k **+122%→+133%**;
  accept stable (T1 context-indep byte-identical 99.17% across 6 depths 8k→256k → rules out slot-boundary bug
  #23658 on our base; T2 realistic reasoning ~50% accept holds ±2pp with edge +12%→+41% → rules out collapse
  #23322). 256k fits VRAM (ncmoe=8, q4 KV, ub1024). abandonCriteria (edge doesn't shrink) MET at every reachable
  depth.
- **CORRECTED verdict (vs first pass):** NOT "reversed/null", and NOT even "revisit ≥256k". The doc's
  `windowed-MTP` cite [ref-105] is a **real** paper (arXiv:2607.21535 "…at Million-Token Context"); the draft-KV
  tax it removes is +27% on the *draft phase* @261k rising to net-negative near 1M — but the draft phase is ~1 of
  ~41 layers, so at our reachable max it's swamped by term (A) target-forward amortization (Leviathan c-ratio;
  MagicDec 1.02×@4k→2.0×@32k; EAGLE-3.1 flat accept; DeepSeek-V3 MTP 85–90%). **We MEASURED native 256k: edge at
  its MAX (+176%), accept pristine.** The tax only dominates near ~1M, unreachable (native 262k; 1M needs YaRN
  past training where quality is gone) and unusable. → **CUT.** Also corrected: **windowing the draft is LOSSLESS**
  (target verifies every token; top-1 unchanged 86–94%), a cost-saver not an accept-killer — my first-pass "can
  only lower accept" was wrong (irrelevant now since we cut it, but recorded). Verified: nextn = full-attn over KV
  (qwen35moe.cpp); accept metric correct (server-context.cpp, bonus-token −1); model not SWA
  (full_attention_interval=4); base 720d7fa40 Jul-25 postdates the May accept bugs; no upstream windowed-MTP PR
  (clean negative). Gate: `ops/a1_mtp_depth_bench.py`; raw `runs/a1-mtp-depth/{a1_depth,a1b_curve,a1c_256k}.csv`.
  Full detail: STATUS §A1. **evidenceGrade 4** (6-depth curve to native 256k + a 2nd realistic-accept regime +
  both hybrids; the only unreached regime, ~1M, is off-limits by arch and quality, so no gap remains that matters).
- **mechanistic grounding (before any run)** — the deploy MoE (`qwen35moe`, 41 blocks) is a **GDN hybrid**:
  only **10 of the 40 base layers bear a KV cache** (full-attention at blk 3,7,11,…,39 — 1-in-4; the other 30
  are GDN/SSM linear, no KV). Crucially the **`nextn`/MTP head (blk 40) IS a full-attention KV-bearing layer**
  (`blk.40.attn_k` present) → the MTP draft *does* re-attend the full-context KV each speculated token. So the
  A1 premise ("draft pays full-context KV") is **mechanically live** (unlike S1/S2's dead premises) — but
  likely **mild**, because only 25% of layers bear KV and GDN already makes long context cheap (~13 MiB/1k KV,
  3× under naive; CONTEXT_PLAN). **This is what A1-0 measures.**
- **hypothesis** — MTP's decode-t/s edge over no-spec (the +27–54% measured at 8k, §E4) **shrinks
  monotonically with context depth**, because the draft's per-token full-attention cost (nextn KV + the verify
  pass's 10 attention layers) grows with depth while the amortized-forward benefit does not scale with it.
- **metric** — paired `(mtp_tps − nospec_tps)/nospec_tps` AND draft accept rate (`draft_n_accepted/draft_n`
  from server `timings`), at depths **8k vs deep** (MoE 128k; dense-27B ~48k — its long-ctx ceiling per
  CONTEXT_PLAN). Fixed generation task (structured code, high-accept regime) so **only depth varies**.
  Deploy config: ncmoe=8, q4_0 KV @128k / q8_0 @8k, ub2048, `--spec-draft-n-max 4`. Isolated arms + cooldown
  (the GPU-A/B variance rule). Quick scope first (extremes, few reps); full 4-depth sweep only if signal.
- **baseline** — the no-spec floor at each depth (same model/ctx, `--spec-type` dropped).
- **successCriteria (to PROCEED to A1-1)** — the MTP edge shrinks materially with depth (e.g. deep-depth
  delta < ~0.5× the 8k delta, CI-separated) AND the accept-rate drop is the driver → a real opportunity for
  windowed-draft attention (mask the nextn layer's attention to a recent window; correctness-safe by
  construction since the draft is always verified over full KV — windowing can only move accept, never output).
- **abandonCriteria (close A1 NULL)** — the MTP edge holds across depth (deep delta within noise of the 8k
  delta) → GDN hybrid kills the degradation; premise HW/arch-specific-false (S1/S2 pattern). Bank the
  depth-bench as a standing gate; do NOT build the windowed kernel.
- **factors** — control: context depth; spec on/off. held: model, generation task, placement, KV type per
  depth, clock (undervolt-stable). noise: cold prefill (warm-up discard; cache_prompt reuse so only rep-0 pays
  prefill, decode still runs at true depth). blocking: architecture (MoE vs dense-27B — both GDN hybrids).
- **reversalPlan** — A1-0 is measurement-only (nothing to revert). A1-1, if built, is an env-gated draft-only
  attention window, OFF by default, byte-identical to upstream on the default path (the fork's standing rule).
- **gate/tooling** — `ops/a1_mtp_depth_bench.py`. **Phase A1-0 DONE 2026-08-04 → CLOSED NULL (see result above).**

### Companion / lower-tier (from the research, kept for completeness)
- **§B3** prefetch reconciliation via **GPU-idle%** instrumentation — **ANSWERED 2026-08-02.** The
  harness now samples `utilization.gpu` over the serving window (`HostSample.gpu_util_pct` →
  `Watch.gpu_util_mean` → `RunResult.gpu_util_mean`, printed per config in `ab_isolate`). Null-arms
  placement sweep (ncmoe 6/24/40, reproducible to ~1pp): idle climbs **39%→63%** as offload deepens
  6→24 (decode 98→53 t/s) — the +24pp is the PCIe expert-transfer stall §E1 removes. **Even at the
  optimum the 3090 sits ~40% idle** (batch-1 A3B is bandwidth-bound), so prefetch has little
  expert-idle to fill at ncmoe=6 → a tax; a 16 GB card forced to ncmoe≥32 sits in the high-idle regime
  → prefetch wins. Same mechanism, opposite sign. Limit: `utilization.gpu` floors ~37% for ncmoe≥24
  (coarse duty, not SM-occupancy), so it resolves the deploy range but not the heavy tail. STATUS §B3;
  `runs/ab-null-qwen36-35b-b3idle/`.
- **§B4** CUDA-graph × pinning — **ANSWERED 2026-08-02.** CUDA graphs are a **+27% decode lever**
  (ncmoe=6, n=4, Cliff +1.0: graphs-off ~79 → graphs-on ~100 t/s) but llama.cpp has them **ON by
  default**, so every number here already banks it — no untapped SGLang-style win. **Pinning-enables-
  graphs FALSIFIED**: graphs gave +27% on plain MASTER_BIN with no pinning (gated by arch+op, not
  host-memory pinning). STATUS §B4; arm-set `b4graph`.
- **§B5** `--pin-hot-experts` (#25932) — **ANSWERED 2026-08-02: N/A on this box.** The flag is
  unmerged/experimental (#25932 closed, successor #26414 open) and its mechanism is anti-disk-eviction
  for a MoE that EXCEEDS RAM, not VRAM expert-pinning. Precondition measured ABSENT: `probe_b5_spill.sh`
  at ncmoe=40 (all ~18 GB experts on CPU) shows 23 cold fault-ins then **0 major faults** on steady-state
  decodes 2–3 — experts stay resident, never spill to disk (model fits 64 GB with margin). No win, no
  build warranted; revisit only for a model exceeding RAM. STATUS §B5.
- **§B2** pinned KV in RAM — **novel for llama.cpp**, **ANSWERED 2026-08-02: precondition CONFIRMED,
  patch works.** §B2a (`probe_b2_kvram.sh`): `--no-kv-offload` at ncmoe=6 is a large context-scaling
  transfer-bound regime — decode −69.7% (~800 tok) → −77.1% (~8000 tok) vs KV-on-GPU (~95 t/s), nokv
  falling 30→22 t/s as KV grows. §B2b: KV lands in a PAGEABLE `ggml_backend_cpu_buffer_type()`
  (`llama-kv-cache.cpp:212`); an env-gated patch (`GGML_KV_PIN_HOST`, `patches/b2b-kv-host-pin.patch`)
  swaps it for the pinned `CUDA_Host` buffer. Verified engaged (20/80 KV tensors on `CUDA_Host(B2b)`).
  Pin vs pageable recovers **+2.5% → +16.8%** (800→8000 tok), rising with depth — a lower bound (only
  25% of tensors pinned). Held in reserve: KV-on-GPU (~94) still dominates, so this is a **128k
  long-context / VRAM-starved** lever, not for the 8k deploy. STATUS §B2; `runs/b2-kvram/`.
- **§E5** MoE expert cache (`--moe-cache-slots`/`--moe-cache-profile`, in the `stack` build; = Fable's
  post-snapshot `moe-expert-cache` / upstream #20757) — **ANSWERED 2026-08-02: NULL/redundant on this
  model.** Routing skew (from a real `llama-moe-trace`, `analyze_moe_skew.py`) is only mild — top-8
  experts = 28% of decode accesses, top-64 = 79% (Qwen3 load-balancing aux-loss). Measured at ncmoe=40
  vs static `--n-cpu-moe` at matched VRAM (`probe_e5_cache.sh`, `probe_e5b_static.sh`): static equals or
  beats the cache at every budget (~6.3 GB: static 38.9 vs cache 38.5; ~8.8 GB: ~45.4 vs 45.8 tie). The
  cache recovers heavy-offload decode only +23% (to ~46 t/s) and no more VRAM-efficiently than lowering
  ncmoe. Not housed in the fork; revisit only for a concentrated-routing model. STATUS §E5; `runs/e5-moe-cache/`.

---

_Status (2026-08-01): §B1 campaign **CLOSED** — 5 MoE + 3 dense controls done. Result: the
generation benefit scales with **active-expert count**, not transfer bytes (the pre-registered
bytes/token reframe was **falsified**, r=−0.84); granite is a compute-bound Mamba exception, not
a dose miscalibration. See STATUS.md §B1 and `[[pin-dose-active-count]]`. **§E1 DONE — the biggest
decode win in the project:** placement (ncmoe 40→6) took qwen36-35B decode **27.6 → 101.7 t/s
(+268%)**; KV q4_0 null at 8 k. New stock baseline **~102 t/s at ncmoe=6**. **Next: §E2 (ik_llama.cpp
head-to-head)** — build it on the idle GPU, then A/B at matched placement (ncmoe=6). `robust.py`
carries `trend_slope_ci` and `non_inferiority` for the synthesis and quality A/Bs._
