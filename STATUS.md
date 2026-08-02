# STATUS — what this project has settled, and what it has not

Referenced from `collectors/host.py` and implied across the experiment scripts; this is
its actual home. Every number below was produced by `python analyze_ab.py` over the
`runs/ab-*/records.json` directories, paired by `(round, ncmoe)`, using the project's own
robust machinery (exact sign test, seeded percentile bootstrap, Cliff's delta). Medians,
never means — this host produces cold-start outliers and a mean was trusted once already.

Regenerate everything here with:

    python analyze_ab.py            # human report, straight off disk
    python analyze_ab.py --json     # provenance blob
    python -m model_lifecycle.cli report -o reports/EVIDENCE.md   # same numbers, from the Store

The number tables below are **authored prose citing regenerable evidence**. That evidence —
the noise floor and every paired comparison — is regenerated into [`reports/EVIDENCE.md`](reports/EVIDENCE.md)
by `model_lifecycle.reports.status`, reading the A/B records out of the Store (where the
backfill unified them). The disk reader (`analyze_ab.py`) and the Store reader share one
pairing core and produce byte-identical numbers, so this prose can be checked against the
data at any time and can never again quietly drift from it.

---

## The instrument's resolution: the noise floor

From `ab-null-qwen36-35b` — the same binary and the same environment in both arms, so the
true delta is **zero by construction**. This is the single most important number in the
project, because every published delta is read against it.

| quantity | value | reading |
|---|---|---|
| null prefill Δ, median | **+0.29%** | the pair is unbiased |
| null prefill Δ, median \|%\| | **~2.3%** | round-to-round paired scatter |
| null order effect (sign_p) | 1.0000 | interleaving IS balanced |

Two consequences, both load-bearing:

1. **A prefill delta below ~2.3% is not evidence, however tidy its median.** The floor is
   ~2.3%, NOT the sub-1% previously assumed offhand.
2. `sign_p = 1.0` on the null means arm order does not bias the pair. This validates the
   interleaving for *every* paired comparison below — without it, they would all inherit a
   systematic bias invisible to replication.

At n=6 the exact sign test bottoms out at **p = 0.0312** (every round agreeing). That is
the strongest claim a 6-round paired run can make distribution-free, and where you see it
below it means the effect was unanimous across rounds.

---

## SETTLED

### 1. Pinning (`GGML_CUDA_REGISTER_HOST=1`) roughly doubles prefill — in three geometries

`cudaHostRegister` on the model-loader's mmap is the fork's real value. Prefill delta of
`pin - base`, paired:

| model | experts | Δ prefill | sign_p | Cliff's δ |
|---|---|---|---|---|
| qwen36-35b | 256 | **+104.9%** | 0.031 | +1.00 |
| qwen3-30b | 128 | **+123.3%** | 0.031 | +1.00 |
| gpt-oss-20b | 32 | **+114.6%** | 0.031 | +1.00 |

Three independent geometries, unanimous every round, ~50× the noise floor. This is the
most solid result in the project. **Pinning is on by default in `serve.py`.**

### 2. Adding the prefetch to an already-pinned baseline COSTS prefill

The switches are not independent and there is no cheap middle. `pinpf - pin`, paired:

| model | experts | Δ prefill | sign_p |
|---|---|---|---|
| qwen36-35b | 256 | **−22.1%** | 0.031 |
| qwen3-30b | 128 | **−11.3%** | 0.031 |
| gpt-oss-20b | 32 | **−8.6%** | 0.031 |

The tax scales with expert count (256 > 128 > 32), unanimous every round, all above the
floor. **Prefetch is off by default in `serve.py`.**

### 3. The published "+58%" was one number hiding two effects

`pinpf - base` on qwen36-35b = **+60.2%** (sign_p 0.031) — the headline, reproduced. It
decomposes exactly: pinning (+105%) followed by prefetch (−22%). The fork's entire value
is the `cudaHostRegister` call; its headline overlap is a tax, not a feature.

### 4. Generation does not move — IN THE NEAR-RESIDENT REGIME

Every `gen_tps` comparison lands inside the noise floor. `pin - base` generation:
−0.15% / +0.24% / +0.04% across the three models (all sign_p ≥ 0.69); the dedicated
`genpin` 35B run gives −0.18% (sign_p 1.0). Pinning buys prefill and nothing else here.

> **SCOPE — do not overread.** All of this is the *near-resident* regime (35B at
> ncmoe=24 generates at ~43 t/s, most weight near the card). The question pinning was
> *supposed* to answer — does it help generation when generation is **transfer-bound** —
> is NOT settled by this data. See OPEN §B1.

### 5. The fork is worth its mechanism, and the rebase reproduces it exactly (n=18)

| comparison | Δ prefill | sign_p | reading |
|---|---|---|---|
| fork − base | +59.3% | <0.0001 | the fork earns its keep |
| rebased − base | +60.4% | <0.0001 | so does the patch on today's upstream |
| rebased − fork | +0.48% | 0.096 (within floor) | **rebase ≡ fork** |

(Cliff's δ is diluted here because this run spans ncmoe {8,24,40} and pools raw rates
across doses; the *paired* sign test and median% are the trustworthy read.) **Actionable:
the 3-week-stale fork can be dropped and the patch carried on current upstream with no
measurable loss** — 266 commits of upstream work for free.

### 6. `turbo-mma-decode` does nothing measurable — do not carry it

`turbo - base`, both models, both metrics, all within the floor: generation −0.04% (35B)
and +0.15% (30B), sign_p 1.0; prefill likewise. The fork author's fused-GQA decode path
is inert on this hardware. n=6 here could detect roughly a >5% effect; this is <0.4%.

### 7. Question 4 resolved: the L18 / A-B disagreement is NOT the build

The 2×2 (`ab-stack`, BUILD × PREFETCH):

| | prefetch off → on |
|---|---|
| **rebase build** | −22.1% (sign_p 0.031) |
| **stack build** | −23.3% (sign_p 0.031) |

The prefetch is a ~22% tax on *both* builds — essentially identical. Build alone
(prefetch off) is +0.65%, within the floor. So the stack's prefill changes do **not**
interact with the prefetch, and the L18's non-monotonic ordering was confounding (its
`cache` factor was inert by construction), not a build×prefetch effect.

---

## TIER-1 — the speed pivot: §E1 placement is the biggest decode win in the project (2026-08-01)

After §B1 closed, the pivot to engine/placement levers (`LANDSCAPE.md` §5) paid off on the
first experiment. **Every A/B this project ran used *maximum* offload (`ncmoe=40`, all 40 expert
layers on the CPU) — the single worst decode placement** — while **21.2 GB of the 24 GB card sat
idle** (qwen36-35B-Q4 uses only ~2.8 GB at max offload). §E1 sweeps `ncmoe` DOWN, bringing
experts back onto the GPU until the VRAM reserve binds. Stock llama.cpp, no fork, no build:

| ncmoe | decode t/s (stock) | VRAM free | status |
|---:|---:|---:|---|
| **40** | **27.6** | 20.7 GB | the campaign's placement — worst case |
| 28 | 40.8 | 15.1 GB | |
| 16 | 70.4 | 9.5 GB | |
| 8 | 93.1 | 5.8 GB | |
| **6** | **101.7** | 4.9 GB | **← recommended optimum (respects the 4 GB VRAM reserve)** |
| 4 | 113.4 | 3.9 GB | REJECTED — 3.9 GB free < 4 GB reserve |

**Decode 27.6 → 101.7 t/s = +268% (3.7×), free**, purely by expert placement. The 4 GB VRAM
reserve is the binding constraint (ncmoe=4 would give ~113 t/s but breaks it). Decode is
near-deterministic here (CV ~0.006), so n=2 resolves the curve; the reserve boundary is sharp.

**The KV-format factor (q4_0 vs q8_0) is a NULL lever at this context.** At ctx=8192, q4_0
decode matches q8_0 within noise (±2%) and frees only **~46 MB** of VRAM — the KV cache is tiny
relative to weights at 8 k, and Ampere's flash-attention fast path is already active for both.
q4_0's payoff is reserved for **long context** (the 128 k agentic case, `[[agentic-local-model-plan]]`),
where KV VRAM dominates and would otherwise force `ncmoe` back up. Untested here; flagged for §E4.

**What this does to the pinning story.** The `pin` arm ran free alongside (genpin), giving the
within-model dose-response: pin's gen benefit **rises with offload** (ncmoe8 ~0% → ncmoe40 +5%)
and is **null at the optimal placement** (ncmoe≤8, near-resident). So *at the operating point you
would actually use*, the fork's pinning is irrelevant — the win is 100% placement. Pinning earns
its keep only in the **forced-heavy-offload** regime (a model too big to place well, or a smaller
GPU), which is exactly the §B1 finding seen from the speed side. **The stock baseline every later
Tier-1 A/B must beat is now ~102 t/s at ncmoe=6, not the 27 t/s at ncmoe=40 we had been citing.**
Raw records: `runs/ab-genpin-qwen36-35b-e1-place-q8/` and `-e1-place-q4/`; the `--kv` flag and the
placement axis are in `ab_isolate.py`.

## §E2 — ik_llama.cpp vs stock: TIE on decode at the operating point; ik wins only where our envelope won't let it run (2026-08-02)

The Tier-1 head-to-head: philosophy **(a) stream-to-GPU** (our stock llama.cpp) vs **(b)
compute-on-CPU** (`ikawrakow/ik_llama.cpp`, fused-MoE + optional run-time-repack). Same GGUF
(qwen36-35B-Q4), same KV (q8_0), `-fa on` both arms, matched placement. **Swept ncmoe** (per the
sweep-first rule — the two philosophies converge at low offload and diverge as experts leave the
GPU, so a single point would mislead). Envelope-clean via a new `--reset-between` (per-config WSL
reset — ik's larger host footprint otherwise contaminates the next arm's load).

**At the operating point (ncmoe=6, where a well-fitting model actually runs), n=4 clean:**

| metric | stock (a) | ik (b) | Δ |
|---|---:|---:|---|
| decode | ~95.3 t/s | ~95.5 t/s | **+0.29% — within noise** (\|Δ\|<1.4%) |
| prefill | ~568 t/s | ~992 t/s | **+75% — ik faster** |

**The engine swap does NOT speed up decode where you would run.** ik's only decode-relevant win is
prefill (compute-bound → its fast CPU/fused kernels help). Mechanically expected: at ncmoe=6, 34 of
40 expert layers are already on the GPU for BOTH engines, so decode is nearly the same computation.

**Sweeping ncmoe reveals ik's decode advantage GROWS with offload** (r0, single-shot — decode here
is near-deterministic):

| ncmoe | stock decode | ik decode | Δ decode | Δ prefill |
|---:|---:|---:|---:|---:|
| 6 | 93.5 | 95.3 | +2.0% | +64% |
| 16 | 64.6 | 68.0 | **+5.3%** | +116% |
| 28 | 39.7 | 43.5 | **+9.6%** | +122% |
| 40 | 27.7 | *crash* | — | — |

So philosophy (b) degrades **less** than (a) as experts move to the CPU — for a model FORCED into
heavy offload (too big to place well; the 128 GB future), ik would win decode. **But ik cannot run
cleanly there on this box, three ways:**

1. **`-rtr` (ik's fast CPU-GEMM repack) OOMs at heavy offload** — repacking all 40 expert layers
   (~18 GB) blows the WSL RAM cap and kills the load (reproduced twice). Its speed mode needs
   *offline* `_R4` quants, not run-time repack, once offload is heavy.
2. **ik's host-RAM footprint breaches the 16 GB Windows reserve** at moderate offload even without
   `-rtr`: ncmoe16 → 15.3 GB free, ncmoe28 → 9.6 GB free (both REJECTED by the guard). Stream-to-GPU
   keeps experts GPU-side and stays within the reserve.
3. **ncmoe=40 generation crashes ik outright** (loads fine, dies running the workload with all 40
   expert layers computing on CPU).

**Net for our hardware:** the engine swap is a **decode tie** at the placement we use and is
**RAM-unsafe at the offload depths where it would help** — its advantage lives exactly in the
regime our safety envelope forbids on 64 GB. Revisit ik when **128 GB RAM** lands (relieves the
reserve breach) AND a model too big for GPU-placement is in play; then its +75% prefill and
offload-scaling decode edge become reachable. **The `--defer-experts` squeeze FAILED to rescue it**
(all 4 rounds at ncmoe16/28 still REJECTED on RAM — deferring load-time residency does not shrink
the steady-state compute RSS): the RAM wall is hard, not a load-phase artifact. Raw:
`runs/ab-e2ik-qwen36-35b-e2-ik-ncmoe6/` (clean n=4), `runs/ab-e2ikdef-qwen36-35b-e2-ikdef/`,
`runs/ab-e2iknr-qwen36-35b-e2-iknr-sweep/` (the offload curve). Arm-sets `e2ik`/`e2iknr`/`e2ikdef`
and `--reset-between` in `ab_isolate.py`.

---

## OPEN — as prominent as the answers, on purpose

- **§B1 — transfer-bound generation — CLOSED as UNREACHABLE on this hardware (2026-07-31).**
  The one regime that would test whether pinning helps *generation* is a big model whose
  experts stream over PCIe (Nemotron-120B-A12B: 12B active/token, ~0.64 t/s at ncmoe=99).
  It cannot be measured on this box, and this is now a *measured* fact rather than the
  earlier `.wslconfig` guess:
  - Nemotron-Q3 is **61.7 GB** — resident + the 16 GB Windows reserve alone exceeds the
    63.8 GB of physical RAM. The old 12/12 REJECTED run was correct; the 0.64 t/s seen once
    was **disk-thrash** (model > the 44 GB WSL cap, `swap=0`, mmap paging from the GGUF),
    not clean PCIe transfer, and pinning cannot even be applied to a model larger than the cap.
  - The smallest quant either quantiser publishes — bartowski **IQ1_S, 46.4 GB** — was
    downloaded and probed. It loads **only at ncmoe=50**, and there leaves **594 MB VRAM free**
    (reserve 4 GB) *and* **2.0 GB Windows free** (reserve 16 GB): both envelopes blown at once.
    Raising ncmoe relieves VRAM but starves RAM; lowering it relieves RAM but OOMs VRAM. **No
    ncmoe satisfies both** — the 46.4 GB model + 24 GB GPU + 64 GB RAM + a live desktop has no
    room for the reserves.
  - **Nemotron was discarded**: both quant files deleted, removed from the model registries.
    Answering §B1 needs different hardware (128 GB RAM makes Nemotron-Q3 trivial), not a
    tuning change here. The historical evidence is kept in `runs/ab-genpin-nemotron-120b/`
    and `runs/residency_nemotron-120b.json`.

  **The general constraint, found while chasing this (2026-08-01).** Testing pinning needs
  mmap on, because the fork's `register_host()` only fires when mmap is in use. But mmap holds
  the whole GGUF resident in RAM, so the *file* — not the offloaded slice — must fit under the
  reserve: on a clean 44 GB baseline, `file <= ~26 GB` to keep Windows above the 16 GB reserve.
  This was confirmed on **Laguna-S-2.1** (poolside, ~117B MoE, 6-7B active): its Q2_K_XL loads
  but leaves Windows at **5.5 GB available** from a clean baseline — the whole 39.7 GB file sits
  in RAM regardless of ncmoe. Every MoE quant that is *also* high-active (transfer-bound) is
  larger than 26 GB; the ones that fit (Laguna-XS, qwen3-30b) are ~3B active, no more
  transfer-bound than qwen36. **High-active and small-file are mutually exclusive here.**
  Laguna-S was discarded (files deleted); Laguna-XS not worth downloading.

  **The reachable proxy — ANSWERED (2026-08-01): pinning DOES move generation, but only when
  transfer-bound, and only ~2%.** qwen36-35B-Q4 (22 GB, fits pinning+mmap) at maximum offload
  (ncmoe=40), all 40 expert layers streaming over PCIe, genpin `pin - base`, n=12:

  | metric | Δ median | sign_p | Cliff's δ | reading |
  |---|---|---|---|---|
  | gen_tps | **+0.58 t/s (+2.14%)** | **0.039** | +0.72 | CI95 [+0.28, +0.88], excludes zero |
  | prompt_tps | +171.1 (+123.7%) | 0.000 | +1.00 | pinning still ~doubles prefill |

  10 of 12 rounds favour pinning; the 2 that don't are r0/r1 (cold-start, base runs fast while
  the machine warms) — from r2 on it is 10 straight positives. This is the project's **first
  positive generation result**, and it settles the SHAPE of §B1: near-resident (ncmoe=24)
  generation does not move (Δ ~0.2%, noise); transfer-bound (ncmoe=40) it moves +2.1%,
  distribution-free significant. Two honest caveats: (1) the effect is SMALL — pinning's
  headline value is still prefill; generation gets a modest transfer-bound bonus. (2) This is a
  **lower bound**: at 3B active it cannot show the 12B-active severity of Nemotron/Laguna-S,
  where ~4x the per-token transfer could make it larger — but that regime is unreachable here.
  Raw records: `runs/ab-genpin-qwen36-35b-maxoff12/` (n=12) and `-maxoff/` (n=6, corroborating).

  **The cross-architecture dose-response — mechanism confirmed, and a pre-registered reframe
  FALSIFIED (2026-08-01).** To locate the dose variable we ran genpin across **5 MoE
  architectures** (all at maximum offload, n=12 each) plus **3 dense controls** (n=4-12). The
  dense controls are the clean negative: gen Δ ~0, prefill strongly positive — the generation
  benefit is **expert-streaming-specific**, not a generic pin effect. The sharpest isolation is
  same-architecture: **Qwen3.6-27B-dense (Δ +0.74%, CI crosses 0) vs Qwen3.6-35B-MoE (+2.1%)**.

  | dense control | gen Δ | prefill Δ |
  |---|---|---|
  | Mistral-24B (ngl20, n=12) | −0.13% (\|Δ\|<0.04) | +90.1% |
  | Qwen3.6-27B-dense (ngl40, n=4) | +0.74% (CI∋0) | +58.6% |
  | ThinkingCap-27B (ngl40, n=4) | +0.23% (sign p=1.0) | +64.0% |

  The 5 MoE, sorted by active-expert count:

  | model | active | expert size | **bytes/token** | gen Δ | base tps |
  |---|---:|---:|---:|---:|---:|
  | gpt-oss | 4 | 13.2M | 1269M | +0.01% | 23.0 |
  | ernie | 6 | 7.65M | 1239M | +0.84% | 23.7 |
  | gemma | 8 | 3.35M | 803M | +1.99% | 25.2 |
  | qwen36 | 8 | 1.90M | 608M | +1.95% | 27.1 |
  | granite* | 10 | 6.12M | 2448M | −0.19% | 11.2 |

  *granite = Granite-4.0-H, a Mamba-hybrid; base tps ~half the others → **compute-bound, not
  transfer-bound**.

  We had pre-registered a **"corrected dose = per-token expert-transfer bytes"** hypothesis
  (built to explain why granite looked off-curve on the raw active-count axis). Recomputing the
  bytes/token from GGUF geometry and correlating against the measured Δ **falsifies it**:

  - Δ% vs **bytes/token**: Pearson **r = −0.84** — *wrong sign*. gpt-oss moves ~2× qwen's bytes
    and is null; qwen moves the least and gains +2%. More bytes ⇒ **less** benefit. Dead.
  - Δ% vs **active-expert count** (the original axis): clean monotone **4→6→8**, and the
    decisive discriminator — **gemma and qwen share active=8 and give the same Δ (~1.97%)
    despite experts differing 1.76× in size**. Same count → same gain, size-independent.

  **Two conclusions.** (1) The benefit scales with the **number of discrete per-token expert
  H2D transfers**, not the bytes moved — consistent with pinning removing **per-transfer
  overhead/latency** (staging setup, sync), *not* bandwidth-time; at decode the bandwidth term
  does not dominate. The original active-count axis was correct; the bytes "correction" is
  discarded. (2) **Granite was never "off-curve by dose"** — it is off-curve because a
  Mamba-hybrid is compute-bound (tps ~11 vs 23-27) and sits **outside the transfer-bound
  regime** where pinning can act. Architectural exception, not a miscalibrated dose. The
  headline is unchanged and now mechanistically grounded: transfer-bound-only, active-count-
  scaled, ~2% ceiling on this hardware. Analysis: `scratchpad/dose_bytes.py`; raw records:
  `runs/ab-genpin-{gpt-oss,ernie-4.5-21b,gemma-4-moe,granite-4.0-h,qwen36-35b}-*/` and the
  dense controls `runs/ab-genpin-{mistral-24b,qwen36-27b-dense,thinkingcap-27b}-*/`.

- **The −10.4% no-mmap residual.** One of the three historically disputed deltas. Not
  covered by any `ab-*` directory here; **still open**, needs its own clean paired A/B.

- **`stackpf − prefetch` = −1.57%, sign_p 0.031 but below the floor.** Sign-consistent
  across all 6 rounds yet sub-2.3% in magnitude — a candidate real-but-tiny build effect
  when prefetch is on. **Not promotable at n=6**; would need more rounds to separate from
  noise. Recorded, not claimed.

- **The quality axis is dark.** `quality_bench` starved on 33/40 HumanEval+ problems (the
  thinking model spends the 1024-token budget reasoning before emitting code). No pass@1
  is measurable until the budget/thinking issue is fixed.

- **SGLang head-to-head on gpt-oss-20b.** New (2026-07-31). The only model both engines
  can serve on this box; llama.cpp numbers exist (base 189 → pin 406 t/s prefill). Awaits
  the SGLang setup now building in WSL.

---

## Stale markers found while writing this (fix in place)

- `config/environment.yaml`: `transient_handling: UNRESOLVED` predates `guard.py`, which
  now implements sustained-breach over K samples plus a load-phase exemption. The marker
  lags the code.
- `collectors/host.py:45`: "There is no Python on the Windows side of this desktop" — there
  is, 3.12.10. The `.exe` collector suffix is still correct for a different reason (it
  measures the Windows host, and the suffix also works when run natively), but the stated
  premise is false.

_Last regenerated: 2026-07-31 from `analyze_ab.py`._
