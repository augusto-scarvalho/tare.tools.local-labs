# A4 — full sourced research behind the double-check (spec-decode instrumentation)

Captured 2026-08-04 from two parallel research passes, so the detail survives a context
clear. The distilled verdict + corrections live in `A4_INSTRUMENTATION.md`; this file is
the raw sourced reference (exact PR numbers/states, formulas, arXiv IDs, thresholds) that
that record only summarized. Two extraction caveats are flagged at the ends of each part.

---

# PART 1 — llama.cpp speculative-decoding instrumentation (upstream PRs/issues/forks)

Repo `ggml-org/llama.cpp`. `[src]` = confirmed by reading the file on our pinned base or
master; `[API]` = confirmed via GitHub REST (authoritative `merged` field);
`[search]` = seen only in search results, exact wording/state lower-confidence.

## Where spec-decode stats surface (3 surfaces, different coverage)

| Surface | Spec coverage | Mechanism |
|---|---|---|
| llama-server per-request `timings` JSON | `draft_n`, `draft_n_accepted` (only when `draft_n>0`) | PR #12603 (merged 2025-03-28) |
| Server logs (per-request + aggregated) | mean acceptance length, acceptance-rate-per-position, per-impl aggregates | PR #24536 (merged 2026-06-16) |
| `/metrics` Prometheus endpoint | **NONE as of early 2026** | attempts unmerged/open (below) |
| CLI `examples/speculative/speculative.cpp` | `n_draft`, `n_predict`, `n_drafted`, `n_accept`, `accept %` | long-standing |

## The per-request `timings` object — exact fields `[src: server-task.h/.cpp on our base]`
`result_timings::to_json()` emits exactly:
```
cache_n, prompt_n, prompt_ms, prompt_per_token_ms, prompt_per_second,
predicted_n, predicted_ms, predicted_per_token_ms, predicted_per_second
// then, ONLY if draft_n > 0:
draft_n, draft_n_accepted
```
- 9 base fields always present. `cache_n` = prompt tokens reused from cache.
- `draft_n`/`draft_n_accepted` are the ONLY speculative JSON fields, present ONLY when SD
  ran. **No `draft_accept_ratio`, no `draft_n_rejected`** — compute acceptance yourself.
- Included when the request sets `"timings_per_token": true` (or on final/usage chunk).
- The richer stats are LOG-ONLY (`server-context.cpp print_timings`):
  `draft_ratio = n_draft_accepted/n_draft_total` (=α),
  `mean_acc_len = 1 + n_draft_accepted/n_draft_verif_steps` (=τ), and per-position
  acceptance. Internal counters in `common/speculative.cpp`: `n_gen_drafts`, `n_acc_drafts`,
  `n_gen_tokens`, `n_acc_tokens`, `n_acc_tokens_per_pos`, and `t_begin_us/t_draft_us/
  t_accept_us` (the drafter-cost accounting hook — also log-only, not JSON).

## `/metrics` Prometheus — no SD metric `[src: tools/server/README.md]`
Exported (require `--metrics`): `llamacpp:prompt_tokens_total`, `:prompt_seconds_total`,
`:prompt_tokens_seconds`, `:tokens_predicted_total`, `:tokens_predicted_seconds_total`,
`:predicted_tokens_seconds`, `:requests_processing`, `:requests_deferred`,
`:n_tokens_max`, `:n_decode_total`, `:n_busy_slots_per_decode`. No TTFT/TPOT metric name;
derive from `timings`. Attempts to add SD metrics:
- **#25327** "server: expose speculative decoding metrics" — **CLOSED, not merged**
  (2026-07-05, ~1 day after open). Proposed `llamacpp:draft_tokens_total`,
  `:draft_tokens_accepted_total`, `:draft_verify_steps_total`, `:draft_acceptance_rate`,
  `:draft_mean_accept_len`. Clearest rejected data point. `[API]`
- **#24850** "WIP: expose cache, speculative-decode, resource metrics" — OPEN/WIP (2026-06-21). `[API]`
- **#26389** "Adding spec-decode counters to /metrics" — OPEN PR (2026-07-31), matches vLLM
  schema (`:spec_decode_num_draft_tokens`, `:...num_accepted_tokens`, `:...num_drafts`,
  `:...num_accepted_tokens_per_pos{position="N"}`). `[API]`
- **#26516** "Feature Request: expose SD counters in /metrics" — OPEN issue (2026-08-03),
  wants vLLM parity for NVIDIA AIPerf. Its body wrongly claims `draft_accept_ratio` is
  already in per-request timings (it is not). `[API]`

## PRs that ADDED spec reporting
- **#12603** — added `draft_n`/`draft_n_accepted` to timings. Merged 2026-03-28. `[API]`
- **#24536** "spec: add spec metrics mean acceptance length and acceptance rate per
  position" — Merged 2026-06-16. Uses vLLM defs VERBATIM:
  `mean_acceptance_length = 1 + accepted_tokens/draft_verification_steps`;
  `acceptance_rate_per_position[i] = accepted_at_pos_i/draft_verification_steps`. Reports in
  server timing logs AND aggregated per-impl stats. **In our base** (print_timings has it). `[API]`
- **#26320** "server: correct accepted tokens when need draft token replay" — Merged
  2026-07-31 (**AFTER our base ~Jul-25**). Bug: on full-checkpoint-restore replay the
  target's correction token was miscounted as an accepted draft, "inflating draft
  acceptance, per-position acceptance and mean len by 1 for each checkpoint restore."
  → pin-watch; triggers only on long-ctx slot save/restore. `[API]`

## `--spec-type` / MTP `[src: common/arg.cpp, common/speculative.cpp]`
`--spec-type` (comma-separated, exempt from deprecation path); enum
`COMMON_SPECULATIVE_TYPE_DRAFT_MTP / _DRAFT_DFLASH / _DRAFT_EAGLE3 / _DRAFT_DSPARK` + the
draft-model path. Draft-model knobs `--draft-max`, `--draft-min`, `--draft-p-min`.
- **#25980** = "model: add NextN/MTP speculative decoding support for GLM_DSA (GLM-5.2)",
  merged 2026-07-29 — the FIFTH in-tree draft-mtp impl (after qwen35, qwen35moe, step35,
  cohere2moe). A model/draft path, NOT new metrics. (Our `ab_isolate.py` comment had cited
  it as "the --spec-type PR" — wrong; corrected.) `[API]`

## Benchmarking-discipline threads
- **#26100** (OPEN, 2026-07-24) "draft-cache replay path bypasses p_min - inflates
  repeated-prompt benchmarks ~10x, degrades mixed-traffic ~3x". On repeated/identical
  prompts acceptance "climbs toward 1.0" vs "~0.6 for this model pair" varied; "repeated-
  prompt protocol reads 9.5x-16.9x higher tok/s than varied-prompt traffic." Path from
  #12635 (commit 94933c8c2). The §63.4 anti-pattern, upstream-confirmed. **We defend via
  `cache_prompt=False` + unique per-rep prefix.** `[API]`
- **#23869** (MERGED 2026-05-29) "server-bench: add speed-bench for speculative decoding".
  "SD speedups really depend on the data, the serving regime, and the system... everyone
  ends up writing one-off scripts." Wires in NVIDIA's SPEED-Bench dataset. Upstream prior
  art; our probe is the local equivalent. `[API]`
- `[search, unconfirmed numbers]` #26222 (HIP, quotes "draft acceptance = 0.62745 (96/153)"),
  #25908 (draft-simple p_min default, acceptance 0.070 vs 0.898), #26010 ([SYCL], has a
  "Draft Acceptance (MTP)" table), #26551 (OPEN PR "Deterministic Draft Filter",
  `--det-draft-accept-all`).

## Forks
`ikawrakow/ik_llama.cpp`: search for spec-acceptance metrics returned zero — no fork-
specific SD instrumentation beyond upstream CLI printout. No other fork with distinctive
SD metrics found. (Soft negative — search-based.)

**Extraction caveat:** #26516's body inaccurately claims `draft_accept_ratio` is already a
per-request timings field — it is not (confirmed against `to_json()` source).

---

# PART 2 — spec-decode & LLM-serving benchmarking methodology (papers)

## Canonical spec-decode math
**Leviathan, Kalman & Matias — "Fast Inference from Transformers via Speculative Decoding"**,
arXiv:2211.17192 (2022; rev 2023), ICML 2023 (PMLR v202 leviathan23a). Mp=target, Mq=draft,
γ=draft tokens/iteration.
- Acceptance rate: β = P(accept x_t~q | prefix); **α = E(β)** (i.i.d. assumption).
- **Expected tokens/iteration (Eq. 1):** `E(#tokens) = (1 − α^(γ+1)) / (1 − α)`. This is the
  mean accepted length (+1 bonus token). Between 1 and γ+1 tokens/iteration.
- **Cost coefficient c** = time(1 draft run) / time(1 target run).
- **Walltime improvement (Thm 3.8):** `(1 − α^(γ+1)) / [(1 − α)(γc + 1)]`. Needs c<1;
  speedup is super-linear in α.

**Chen et al. — "Accelerating LLM Decoding with Speculative Sampling"**, arXiv:2302.01318
(DeepMind, 2023). Accept x~q(x) w.p. **min(1, p(x)/q(x))** (p=target, q=draft); on reject,
resample from normalized residual **(p−q)₊/Σ(p−q)₊** → provably samples exactly from p
(distribution-preserving / lossless). Empirical acceptance = accepted / (K+1), K=lookahead;
DECREASES as K grows. 2–2.5× on Chinchilla 70B.

**Definitions (consensus):** α = accepted/proposed (per-token accept prob, i.i.d.);
mean accepted length τ = (1−α^(γ+1))/(1−α) = tokens-per-forward-pass; "tokens per forward"
= synonym for τ.

## Why single tokens/s misleads
**Spec-Bench — Xia et al.**, arXiv:2401.07851 (ACL Findings 2024). 6 task domains
(MT-bench multi-turn, WMT14 DE-EN translation, CNN/DM summarization, Natural-Questions QA,
GSM8K math, DPR RAG) precisely because "the speedup of Speculative Decoding methods varies
significantly across different subtasks" and acceleration "primarily hinges on the
acceptance rate of drafted tokens at each step." Reports wall-time speedup ratio + Mean
Accepted Tokens τ. → the formal basis for §63.4's never-one-prompt rule.

**Report instead:** α and τ (hardware-independent), measured c / γc+1 accounting, TTFT +
TPOT/ITL, per-task breakdown, and a TRUE no-spec floor on the identical stack.

## TTFT / TPOT / ITL / goodput
vLLM/SGLang `vllm bench serve` reports percentiles for `ttft, tpot, itl, e2el`:
- **TTFT** = arrival → first output token (queue + prefill).
- **TPOT** = (E2E − TTFT)/(N_out − 1) — mean decode time/token after the first.
- **ITL** = per-gap time between consecutive tokens (report percentiles; TPOT is its mean).
- **E2EL** = submit → complete. **Goodput** = rate of requests meeting ALL SLOs jointly.

**MLPerf Inference (MLCommons):** prompt phase→TTFT, generation phase→TPOT, treated as
latency constraints; score = max sustainable throughput within bounds at p99.
- Llama-2-70B Server (Conversational): TTFT ≤ 2 s, TPOT ≤ 200 ms.
- Llama-2-70B Interactive: TTFT ≤ 450 ms, TPOT ≤ 40 ms (≈25 tok/s/user), p99.
- Llama-3.1-405B: p99 TTFT = 6 s, p99 TPOT = 175 ms.

**"On Evaluating Performance of LLM Inference Serving Systems"** — Agrawal et al.,
arXiv:2507.09019 (2025). Pitfalls: (1) workload dependence — no single number generalizes;
(2) mean masks tail — report p50/p99; (3) warmup — discard initial, measure steady state;
(4) throughput must be paired with latency bounds; (5) specify all workload params.

## DOE for systems benchmarking
- **L18 = L18(2¹×3⁷)**: 18 runs screening one 2-level + up to seven 3-level factors (vs
  2×3⁷=4374 full). Taguchi OAs are fractional-factorial designs concentrating on MAIN
  effects; **two-factor interactions are aliased/confounded into the residual** and the
  design ASSUMES them negligible/known. → a screen identifies candidate factors only and
  MUST be confirmed by a focused follow-up (full-factorial on the shortlist). (NIST
  e-Handbook §5; LibreTexts "DOE via Taguchi Methods".) — exactly what `taguchi_screen.py`
  docstring already states, and our `ab_isolate.py` is the confirmation stage.
- **Jain, *The Art of Computer Systems Performance Analysis*** (Wiley 1991) — 2ᵏ, 2ᵏʳ,
  2^(k−p) fractional factorial; factor-effect isolation; CIs on effects. Canonical systems-DOE cite.
- **Georges, Buytaert & Eeckhout, "Statistically Rigorous Java Performance Evaluation,"**
  OOPSLA 2007 — startup vs steady-state, multiple runs + CIs over single numbers.
- **Bulej et al., "Quantifying Performance Changes with Effect Size Confidence Intervals,"**
  arXiv:2007.10899 — bootstrap effect-size CIs.
- **Hoefler & Belli, "Scientific Benchmarking of Parallel Computing Systems,"** SC'15.
- Robustness practices (all already in our `robust.py`): paired comparisons; interleave to
  cancel drift; non-parametric (sign test / Wilcoxon, bootstrap CI); noise-floor/null (A/A).

## Evolution 2024–2026
- Adaptive draft length / acceptance prediction: **SpecDec++** (arXiv:2405.19715, COLM 2025;
  trained acceptance-prediction head, +7–9% over fixed length), **AdaEDL** (training-free
  entropy lower-bound early stop), **DISCO** (learned threshold), **AdaEAGLE**,
  **BanditSpec** (arXiv:2505.15141, draft-length as multi-armed bandit at inference).
  **SpecForge / SPEED-Bench / MMSpec** (broader suites incl. vision-language).
- **"Speculative Decoding: Performance or Illusion?"** (SpecDecode-Bench) —
  Liu, Yu, Park, Stoica, Cheung, arXiv:2601.11580 (2026), specdecode-bench.github.io.
  Prior evals inflate at **batch size 1** ("unrealistic setting that inflates speedup") on
  "research prototypes, not production-grade systems." **Batch-size collapse:** SD always
  helps but gains shrink with batch — EAGLE Llama-3-70B **1.96×@b1 → 1.21×@b128**; larger
  models degrade more (−4.3% @8B vs −14.0% @70B going 1→32). **Verification dominates
  execution 42–95%** → every rejected token wastes verification compute. Task-dependent
  ("no single method wins everywhere"; n-gram wins code editing, EAGLE-3 best all-rounder;
  oracle ceiling ~4.9× code vs ~2.2× realized). → **our §CC already measured this collapse
  (MTP OFF at N≈4).** Batch-1 α/τ/speedup = upper bound, not serving prediction.

## Harness verification table
| Quantity | Correct definition | Source |
|---|---|---|
| α (acceptance rate) | E[β], per-token accept prob, i.i.d. | Leviathan §3 |
| τ (mean accept length / tokens-per-pass) | (1−α^(γ+1))/(1−α); empirically 1+γα for fixed-γ drafting | Leviathan Eq.1; #24536 |
| c (cost coefficient) | time(1 draft)/time(1 target) | Leviathan §3 |
| Speedup (walltime) | (1−α^(γ+1))/[(1−α)(γc+1)] | Leviathan Thm 3.8 |
| Accept rule (lossless) | accept x~q w.p. min(1,p/q); resample (p−q)₊ | Chen 2023 |
| TPOT | (E2E−TTFT)/(N_out−1) | vLLM/MLPerf |
| Goodput | rate meeting all SLOs jointly | vLLM/Agrawal 2025 |

**Extraction caveats:** (a) the arXiv→markdown pass swapped p/q labels in Chen 2023's accept
rule — canonical is p=target, q=draft, accept w.p. min(1,p/q); (b) both closed-form speedup
formulas assume i.i.d. acceptance + batch-1 — treat as an upper bound (per 2601.11580).

---

## Note on the empirical fit for OUR setup
Our probe measured α=0.69776, γ=4 → τ=1+γα=3.79 (== server log). Note the FIXED-γ empirical
τ=1+γα differs from Leviathan's i.i.d.-and-stop-at-first-reject closed form (1−α^(γ+1))/(1−α):
llama.cpp's MTP drafts all γ regardless then verifies, and `draft_n` counts all γ proposed
per step, so with `n_verif_steps = draft_n/γ` the identity τ=1+γα is exact for our path. The
i.i.d. closed form is the theoretical block-efficiency model; the 1+γα form is what our
counters actually produce. Use 1+γα for our harness; the closed form only to sanity-check α.
