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
(−0.32%, p=1.0); qwen36 (8) **+2.14%** (p=0.039); pending Granite (10), Gemma-4 (8), Ernie (6).
Dense controls (Mistral, Qwen3.6-27B, ThinkingCap: 0 experts) must read **null** — the negative
control that proves the mechanism is expert-streaming, not something generic to offload.
**evidenceGrade** — 3 now; 4 if the trend holds across the 5 arches + the 3 dense nulls.

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

### §E1 — expert placement + KV format (stock llama.cpp, FREE, do first)
- **hypothesis** — decode t/s is maximized by keeping as many expert layers on the GPU as fit
  before VRAM spills (our runs used *max* offload, likely past the optimum), and by a KV format
  that stays on Ampere's flash-attention fast path.
- **metric/baseline** — `gen_tps` vs `ncmoe` (dose) at fixed model/ctx; baseline = our current
  max-offload number per model. **success** — an `ncmoe` below max with higher decode t/s, VRAM
  not spilling. **abandon** — max already optimal.
- **factors** — control: `ncmoe`, and `-ctk/-ctv` {q8_0, q4_0}. nuisance: cold start (warm-up).
- **how** — reuse `ab_isolate`'s dose axis: sweep `ncmoe` from smallest-that-fits up to max, plus
  a KV-format arm. No fork, no build — the cheapest Tier-1 win, and it recalibrates every later A/B.

### §E2 — ik_llama.cpp vs our stock (philosophy (b) vs (a)) — the key head-to-head
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

### §E4 — MTP / speculative decode (amortize the movement)
- **hypothesis** — running the MTP/`nextn` head as a draft (`--spec-type draft-mtp`) amortizes
  weight *and* KV movement → higher decode t/s in the memory-bound regime, scaling with accept rate.
- **metric/baseline** — `gen_tps` AND accept rate, draft-mtp on vs off. **success** — decode t/s
  up beyond the floor at an accept rate near upstream's ~82%; spec-decode is exact by construction,
  so **verify token-identical** output (quality-neutral). **abandon** — low accept / no gain here.
- **factors** — control: spec on/off. held: model, ctx.
- **dependency** — a model whose MTP head llama.cpp actually *uses* (the DavidAU Fable-Fusion has
  MTP; gpt-oss has EAGLE3). Ties directly to `[[agentic-local-model-plan]]`. **Target to beat:
  upstream #25642 — +30% t/s, ~82% accept.** Biggest single-stream lever for the long-context agent.

### Companion / lower-tier (from the research, kept for completeness)
- **§B3** prefetch reconciliation via **GPU-idle%** instrumentation — explains our tax vs the
  3060's win; cheap, and adds the discriminating metric to the harness.
- **§B4** CUDA-graph × pinning — does pinning's real payoff come from *enabling graph capture*
  (SGLang's 8→197 t/s was largely graph)? Test if llama.cpp CUDA-graph + our pinned buffers beats
  pinning alone.
- **§B5** `--pin-hot-experts` (upstream #25932) — selective vs blanket pin on the generation side.
- **§B2** pinned KV in RAM — **novel for llama.cpp** (no prior art); do carefully, no upstream
  baseline to lean on.

---

_Status: §B1 campaign running (5 MoE done/in-flight + 3 dense controls). §B2–§B5 and §E1–§E4
designed and pre-registered here; not started — the campaign runs to completion first, then the
pivot is Tier-1 (§E1 → §E2 → §E3/§E4), ordered by inference-speed payoff. `robust.py` now carries
`trend_slope_ci` (dose-response) and `non_inferiority` for the synthesis and the quality A/Bs._
