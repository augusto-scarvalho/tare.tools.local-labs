# A4 — Spec-decode & benchmark instrumentation discipline: DONE, mostly already-captured — the standing harness now logs draft acceptance, and the §63.4 checklist is closed with three corrections

**Verdict (2026-08-04, deep-dived before implementing):** A4 is an **instrumentation** item, not a
performance lever, and it was **largely already-captured** (the S2 pattern). The genuine gap was narrow —
draft **acceptance** never reached the *standing* A/B harness — and it is now closed with a small additive
change, source- and empirically-validated. Two doc claims and one metric formula were corrected in the
process. No deploy config changes.

## The question (IDEAS_BACKLOG A4, research doc §63.4 / §40 / §22)

§63.4 lists the **mandatory** spec-decode metrics and the anti-pattern to kill:

> accepted draft tokens **per verification**; drafter time vs. time saved; incremental draft/KV VRAM;
> **TTFT, TPOT, wall-clock**; distribution equivalence/divergence under sampling; code quality + retry rate.
> **Anti-pattern:** report only tokens/s on one long deterministic output — coding agents alternate large
> prompts, tools, short outputs, patches, JSON, and **acceptance can change radically**.

§40 adds a **fractional-factorial** experiment matrix (factors: `model_quant, kv_type, context, speculation,
cache, engine, concurrency, deterministic`; "≥3 reps/cell, promotion cells more"). §41 lists acceptance +
ITL + p50/p95/p99 among promotion metrics.

## What we already had vs. §63.4 (the audit that set the scope)

| §63.4 mandatory metric | Already in the harness | Gap |
|---|---|---|
| accepted draft tokens / verification (τ) | `ops/a1_mtp_depth_bench.py`, `verify_mtp.py` (ad-hoc) | not in standing A/B |
| TTFT | `collectors/request.py` (streaming, first *content* token) | — |
| TPOT / wall-clock | `predicted_ms/predicted_n` (= 1/gen_tps); `total_s` | TPOT unnamed |
| draft/KV incremental VRAM | `min_free_vram_mb` (not draft-attributed) | partial |
| distribution equivalence/divergence | `verify_mtp.py`, `compare_base2.py`, §Q HumanEval+ | — |
| code quality + retry rate | HumanEval+ (§Q), `quality_bench.py`, `agentic_gate.py` | retry not instrumented (Track H) |
| fractional-factorial matrix (§40) | `taguchi_screen.py` (L18 orthogonal + S/N + orthogonality self-check) | factors differ (see below) |

So the **only** machine-readable §63.4 metric missing from the standing harness was **draft acceptance**.
That is now wired in.

## What was implemented (additive, low-risk, unit- + GPU-validated)

1. **`collectors/request.py`** — captures the two speculative fields the server exposes in the per-request
   `timings` JSON (`draft_n`, `draft_n_accepted`), stored **raw** (the file's "never store a quotient" rule),
   and exposes:
   - **`accept_rate`** (α = accepted/drafted) — the intrinsic, hardware-free drafter quality.
   - **`tpot_ms`** (decode ms/token, server-exact, prefill excluded) — the standard interactivity metric,
     named so a reader need not invert a t/s.
2. **`workloads/throughput.py`** — `RunResult` aggregates `accept_rate` + `tpot_ms` across reps (None on
   no-spec arms, by construction).
3. **`ab_isolate.py`** — a new **SPEC-DECODE METRICS** report block prints α, **τ**, TPOT and gen t/s per
   arm, so a spec win is never a bare tokens/s (the §63.4 anti-pattern). The base arm reads `no-spec (no
   draft)` — the asymmetry shows *which* arm drafted.
4. **`a4_spec_metrics_probe.py`** (WSL fork tree) — the standing gate: launches the deploy MoE MTP with
   stderr KEPT, and asserts the harness numbers reproduce the server's own logged values.

## Metric correctness — source- and empirically-verified

Server source (`tools/server/server-context.cpp`, `server-task.cpp`, our pinned base `720d7fa40`):

- The per-request JSON `timings` carries **only** `draft_n` and `draft_n_accepted`, and **only when
  `draft_n > 0`** (`result_timings::to_json()` guards them). No ratio field; a no-spec arm omits the keys
  entirely — so **None, not 0**, is the correct "spec was off" sentinel (a 0 would mean "spec ran, accepted
  nothing"). The harness honours this.
- **α = draft_n_accepted / draft_n** is exactly the server's logged `draft_ratio`. **Validated to 5 decimals**
  by the probe: JSON `187/268 = 0.69776` == the server's logged `draft acceptance = 0.69776`.
- **τ (mean accept length)** = the §63.4 "accepted tokens per verification" = the server's / vLLM's / Leviathan
  Eq. 1 `mean_acc_len = 1 + n_draft_accepted / n_draft_verif_steps`. **`n_draft_verif_steps` is NOT in the JSON**
  (it lives only in the stderr log, upstream #24536), so τ is **not derivable from the JSON alone**.

### The correction the probe caught (before it shipped)

The first implementation derived τ from `predicted_n`, on the reasoning that each verification step emits one
target token plus its accepted drafts, so `n_verif_steps = predicted_n − draft_n_accepted − 1`. The probe
compared it to the server's own logged `mean len` and it was **wrong**: derived **3.75** vs. logged **3.79**
(the boundary constant is run-dependent — here the true step count was 67, not the 68 that `−1` implies).

The **robust exact relation** is instead:

> the drafter proposes γ (= `--spec-draft-n-max`) tokens per step, so `n_verif_steps = draft_n / γ`
> ⇒ **τ = 1 + γ·α**.

Check against the log: `draft_n = 268 = 4 × 67` (γ = 4) and `1 + 4 × 0.69776 = 3.79` — matches the server's
`mean len` to the decimal. τ is therefore computed **where γ is known** (the `ab_isolate` spec block, from the
arm's own flags), never from `predicted_n`, and the gate asserts the identity against the log every run.

## Measured (deploy MoE Qwen3.6-35B-A3B MTP Q4_K_M, ncmoe=8, `-fa on`, base 720d7fa40; probe, single greedy code+prose completion)

| arm | gen t/s | TPOT (ms/tok) | draft_n | accepted | α | τ = 1+γα |
|---|---:|---:|---:|---:|---:|---:|
| nospec | 90.6 | 11.03 | — (absent) | — | — | — |
| mtp (γ=4) | 121.1 | 8.26 | 268 | 187 | **0.698** | **3.79** |

(+33.6% decode; α and τ reproduce the server log exactly. A single deterministic prompt is used **only to
validate the plumbing**; real acceptance is workload-dependent — see the anti-pattern below.)

## Two doc corrections banked

1. **"§40 factors map ~1:1 to our levers" is false.** Our L18 (`taguchi_screen.py`) screens **build knobs**
   (pin, prefetch, ubatch, ncmoe, kv, cache); §40's matrix is **workload-facing** (`speculation, context,
   deterministic, engine, concurrency`). They overlap only on `kv` and `ncmoe`. The honest statement: we have
   *a* fractional-factorial screen with the correct discipline (orthogonality self-checked; two-factor
   interactions confounded into the residual and confirmed by `ab_isolate`, exactly the NIST/LibreTexts
   warning), but it is **not** the §40 matrix and does not include a `speculation` factor.
2. **`ab_isolate.py`'s `e4mtp` comment cites `#25980` as "the `--spec-type` PR".** #25980 is actually
   "NextN/MTP for GLM_DSA (GLM-5.2)" (merged 2026-07-29); the `--spec-type draft-mtp` machinery predates it.
   Corrected inline.

## Why MTP works for us at batch-1, and the ceiling — literature corroboration

- **Leviathan/Chen** (arXiv:2211.17192 / 2302.01318): speculative sampling is **distribution-preserving**
  (accept x∼q w.p. min(1, p/q), resample the residual); speedup `f = (1−α^{γ+1})/((1−α)(γc+1))`. For MTP
  self-draft c is tiny (one extra head, ~1/41 layers), so the amortization dominates — consistent with our
  +33–83% across models (S3/A1).
- **Spec-Bench** (arXiv:2401.07851) reports τ across **6 task domains** precisely because "speedup varies
  significantly across subtasks" and "hinges on the acceptance rate" — the formal basis for §63.4's
  never-one-prompt rule. Our S3 already tests 3 regimes (GEN/EDIT/pure-copy); this instrumentation makes α/τ
  first-class in every future run.
- **"Speculative Decoding: Performance or Illusion?"** (arXiv:2601.11580, 2026): batch-1 numbers **overstate**
  gains — EAGLE Llama-3-70B **1.96×@b1 → 1.21×@b128**; verification dominates 42–95% of execution. **We
  independently measured this collapse** (§CC: MTP flips OFF at N≈4, halves throughput at N=8). So: **batch-1
  α/τ/speedup are an upper bound, not a serving prediction** — recorded on every spec report.

## Upstream corroboration (PRs/issues, verified)

- **#12603** (merged Mar-2025) added `draft_n`/`draft_n_accepted` to the timings JSON — the exact fields we
  read. **#24536** (merged Jun-2026, in our base) added the log-only `mean_acc_len = 1 + accepted/verif_steps`
  + per-position acceptance, using vLLM's definitions verbatim — the authority the probe checks against.
- **`/metrics` (Prometheus) has NO spec-decode metric** as of early 2026: the PR to add one (**#25327**) was
  **closed without merge**; #24850/#26389 open, #26516 requests it. → we read per-request `timings`, never
  `/metrics`, for acceptance. (Correct choice, source-forced.)
- **⚠ #26320** (merged 2026-07-31, **after** our base) fixes a bug where checkpoint-restore *replay*
  miscounts the target's correction token as an accepted draft, **inflating acceptance/mean-len by +1 per
  restore**. Our base likely predates it → acceptance could be marginally inflated **only** on long-context
  slot save/restore; batch-1 decode benchmarks don't trigger it. Pin-watch note added.
- **#26100** (open): the draft-cache replay path bypasses `p_min`, so **repeated/identical prompts inflate
  acceptance toward 1.0 and throughput ~10× (9.5–16.9×)** vs varied traffic — the §63.4 anti-pattern,
  upstream-confirmed. **Our harness already defends** against it: `request.py` sends `cache_prompt=False` and
  a unique per-rep prefix, so we never measure the inflated cache-replay path.
- **#23869** (merged): upstream `server-bench` SPEED-Bench for SD ("everyone ends up writing one-off scripts")
  — prior art; our probe/gate is the local equivalent. `ik_llama.cpp` carries no fork-specific SD metrics.

## Methodology & statistics — validated against the literature

Our A/B discipline (paired, interleaved arm-flip, even-rounds-only, warm-up discarded, `sign_test_p`,
`bootstrap_ci`, `hodges_lehmann`/`mad`/`cliffs_delta`, the **noise-floor null** A/B) is textbook: Jain
*Art of Computer Systems Performance Analysis* (DOE, 2^{k−p} fractional factorial), Georges et al. OOPSLA'07
(startup vs steady-state, CIs over single numbers), Hoefler & Belli SC'15, Agrawal et al. arXiv:2507.09019
(workload dependence, tail over mean, warm-up, latency-bounded throughput). The Taguchi confounding caveat our
`taguchi_screen.py` docstring already states is exactly NIST/LibreTexts. **No stats gap for A4.**

## Deliberately NOT built (with reason)

- **Drafter-time-vs-saved** and **per-position acceptance**: the counters exist (`common/speculative.cpp`
  `t_draft_us`, `n_acc_tokens_per_pos`) but are **stderr-log-only**, not JSON. Low ROI at batch-1 (MTP's c is
  negligible and the +33–83% already proves saved ≫ drafter cost). Parse the log only if a future drafter's
  cost becomes non-trivial.
- **ITL p50/p95/p99**: needs the per-token `timings_per_token` stream; at batch-1 on this box decode is
  near-deterministic (CV ~0.006) so mean TPOT carries the interactivity story. Add if we ever serve concurrent.
- **A §40 `speculation` factor in the L18**: the spec lever is better measured by the dedicated paired
  `e4mtp` A/B (clean floor vs mtp) than folded into a screen; and S3 already answered the drafter-policy
  question. Not worth a redesign of the array.

## Re-open / re-run trigger

Run the gate `a4_spec_metrics_probe.py` after **any pin bump** (the fields or the τ identity could move — e.g.
#26320 changes the counts on restore), or if a served model uses a **variable** draft length (then
`n_verif_steps ≠ draft_n/γ` and τ must come from the log, not the identity). Otherwise A4 is closed: the
standing harness reports α, τ, TPOT and wall-clock on every spec run, and the one-number-t/s anti-pattern is
retired.
