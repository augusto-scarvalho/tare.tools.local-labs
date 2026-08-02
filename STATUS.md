# STATUS — what this project has settled, and what it has not

> **New here / post-context-reset? Read [`DEPLOY.md`](DEPLOY.md) first** — the one-page consolidated
> best config, the decode-lever stack (placement +268%, CUDA graphs +27%, MTP +27–83%), the safety
> envelope, the machine baseline, and the remaining optional experiments.

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
| 40 | 31.3† | 36.4† | **+16.5%** | +146%† |

†ncmoe=40 re-measured 2026-08-02 with `swap=16GB` (see failure-mode §3 below); both arms ran in one
session, so the **within-run Δ is clean**, but the absolute t/s are post-XMP (DDR5-5600) and ~13% above
the pre-XMP 6/16/28 rows — expected, since offloaded experts compute on the CPU over RAM bandwidth. The
**trend** (2.0→5.3→9.6→16.5%) is what carries, not the absolute rates across sessions.

So philosophy (b) degrades **less** than (a) as experts move to the CPU — for a model FORCED into
heavy offload (too big to place well; the 128 GB future), ik would win decode. **But ik cannot run
cleanly there on this box, three ways:**

1. **`-rtr` (ik's fast CPU-GEMM repack) OOMs at heavy offload** — repacking all 40 expert layers
   (~18 GB) blows the WSL RAM cap and kills the load (reproduced twice). Its speed mode needs
   *offline* `_R4` quants, not run-time repack, once offload is heavy.
2. **ik's host-RAM footprint breaches the 16 GB Windows reserve** at moderate offload even without
   `-rtr`: ncmoe16 → 15.3 GB free, ncmoe28 → 9.6 GB free (both REJECTED by the guard). Stream-to-GPU
   keeps experts GPU-side and stays within the reserve.
3. **ncmoe=40 is RAM-bound, not a hard crash — re-measured 2026-08-02.** The earlier "crashes
   outright" was `swap=0` (no headroom). With `swap=16GB` added to `.wslconfig` (memory kept at 44 GB,
   backup saved), ik at ncmoe=40 loads in 8 s and **completes generation at 36.4 t/s — +16.5% vs
   stock's 31.3**, filling the offload curve's missing point and continuing its monotone growth. It is
   still **REJECTED by the guard**: Windows-available fell to **11.3 GB, sustained under the 16 GB
   reserve** (`reason: ram 11283MB < 16384MB for 3 samples`). So the RAM wall is *measured*, not
   inferred from a crash — and severe enough that the sustained pressure OOM-killed the orchestrator
   twice mid-run (the iGPU desktop-app move, which now renders from system RAM, compounds it). Net
   unchanged: ik's heavy-offload decode edge is real but unreachable inside the 64 GB safety envelope.

**Net for our hardware:** the engine swap is a **decode tie** at the placement we use and is
**RAM-unsafe at the offload depths where it would help** — its advantage lives exactly in the
regime our safety envelope forbids on 64 GB. Revisit ik when **128 GB RAM** lands (relieves the
reserve breach) AND a model too big for GPU-placement is in play; then its +75% prefill and
offload-scaling decode edge become reachable. **The `--defer-experts` squeeze FAILED to rescue it**
(all 4 rounds at ncmoe16/28 still REJECTED on RAM — deferring load-time residency does not shrink
the steady-state compute RSS): the RAM wall is hard, not a load-phase artifact. Raw:
`runs/ab-e2ik-qwen36-35b-e2-ik-ncmoe6/` (clean n=4), `runs/ab-e2ikdef-qwen36-35b-e2-ikdef/`,
`runs/ab-e2iknr-qwen36-35b-e2-iknr-sweep/` (the offload curve) and
`runs/ab-e2iknr-qwen36-35b-e2-iknr-ncmoe40-swap/` (the ncmoe=40 point, `swap=16GB`). Arm-sets
`e2ik`/`e2iknr`/`e2ikdef` and `--reset-between` in `ab_isolate.py`.

## §E4 — MTP speculative decode: EXACT, ~80% accept, and it decouples decode from placement (2026-08-02)

The Tier-2 lever, and the first that moves decode *without* touching placement. `--spec-type
draft-mtp` self-drafts from the model's own multi-token-prediction head (no external draft model):
the head proposes N tokens, the model verifies them in one forward pass, accepted tokens are free.
**Exact by construction** — verified byte-identical below — so this is pure speed, quality-neutral.

Needed the MTP weights: our on-disk Q4 GGUFs ship no MTP head, so `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`
(22.7 GB, `qwen35moe.nextn_predict_layers=1`, `blk.40.nextn.*`) was downloaded to its own dir.
MASTER_BIN (commit `720d7fa40`, carries the #25980 MTP infra) loads and runs it. Arm-set `e4mtp`:
one model in both arms, differing only by `--spec-type draft-mtp --spec-draft-n-max 4`.

**The clean A/B (ncmoe=8, n=4, both arms inside the envelope):**

| metric | base | mtp | Δ |
|---|---:|---:|---|
| decode | ~92 t/s | ~116.5 t/s | **+26.75%** (Cliff δ +1.00, CI95 [+23.8,+25.3], 4/4) |
| prefill | ~549 t/s | ~523 t/s | −4.5% (within noise; the draft processes the prompt too) |

**Verification (greedy, temp=0, a structured coding prompt) — `verify_mtp.py`:**

| quantity | value |
|---|---|
| token-identical (base vs mtp) | **TRUE** — 945 chars, byte-for-byte. Exact, quality-neutral. |
| accept rate | **194/241 drafted = 80.5%** — matches upstream #25642's ~82% target |
| decode on THIS prompt | base 90.4 → mtp **139.6 t/s = +54%** |

**The speedup is content-dependent** (as spec-decode must be): +27% on the free-form reasoning
benchmark, **+54% on structured code** where the head predicts well. The **accept rate is the
invariant**; the t/s follows how predictable the text is.

**MTP decouples decode from placement.** The `mtp` arm holds ~116 t/s at BOTH ncmoe=6 and ncmoe=8,
while `base` falls 100→92 as experts move to CPU — MTP amortizes the forward pass, so the slower the
pass, the more it saves. Envelope consequence: at the optimal ncmoe=6 mtp hits **116 t/s (+16% over
base's 100)** but leaves only **3316 MB VRAM free < the 4 GB reserve** (the draft context costs
~1.15 GB), so the guard REJECTS it; at ncmoe=8 both fit. **Net deployable: ~116 t/s with MTP inside
the safe envelope (ncmoe=8) vs base's best safe 100 t/s (ncmoe=6) — +16% deployable, +27% at matched
placement, +54% on code, output identical.** The biggest *safe* single-stream decode win since §E1's
placement, and it **stacks on** placement rather than trading against it.

**Beats the target where it counts:** #25642 is +30% t/s / ~82% accept; we match the accept (80.5%)
and beat +30% on structured content, land just under (+27%) on reasoning. **Actionable: turn
draft-mtp ON for the agentic long-context deployment** (tool-call-heavy, structured output is exactly
the high-accept regime), at ncmoe=8 to seat the draft context. Raw:
`runs/ab-e4mtp-qwen36-35b-mtp-e4-mtp-moe-ncmoe8/` (clean n=4) and `-ncmoe6/` (the +16%/rejected optimum).
Arm-set `e4mtp`, models `qwen36-35b-mtp`/`qwen36-27b-mtp` in `models.py`; greedy check in `verify_mtp.py`.

**The 27B "dense" (Gated Delta Net hybrid) — ANSWERED, the biggest MTP uplift in the project.**
Qwen3.6-27B is a Gated Delta Net *hybrid*, not a plain dense transformer; this build DISABLES the fused
delta-net kernel (`resolve_fused_ops: ... not supported, set to disabled`) — **non-fatal**: it loads and
serves at `-ngl 65` (all layers on GPU), the warning only means the fused fast-path is off, so base
decode is a modest 33 t/s. draft-mtp on the same GGUF, n=3 (one round dropped as a *transient* RAM-reserve
breach — `ram 11948MB < 16384MB`: the 17 GB mmap + draft + iGPU RAM sit near the 16 GB reserve):

| metric | base | mtp | Δ |
|---|---:|---:|---|
| decode (benchmark) | ~33.3 t/s | ~49.8 t/s | **+49.4%** (Cliff +1.0, CI95 [+16.40,+16.52], MAD 0.08) |
| decode (greedy code) | 34.4 t/s | **63.0 t/s** | **+83%** |
| accept rate | — | 190/258 = **73.6%** | token-identical ✓ |

**Bigger than the MoE's uplift despite a LOWER accept rate (73.6 vs 80.5%).** Mechanism: a 27B *dense*
forward pass (all weights every token) is far more expensive than the MoE's 3B-active pass, so each
amortized pass saves more absolute time. This is the §E4 thesis at its cleanest — **the MTP payoff
scales with per-token forward-pass cost, not with accept rate.** It straddles the published +73% (under
it on reasoning, over it on code). Both arms fit VRAM (base 7.0 / mtp 4.5 GB free); the binding
constraint here is the **16 GB Windows RAM reserve, not VRAM**. A newer build with the fused Gated Delta
Net kernel would raise base decode and likely shrink the ratio. Raw:
`runs/ab-e4mtp-qwen36-27b-mtp-e4-mtp-dense-ngl65/`.

## §B4 — CUDA-graph capture is a +27% decode lever, and llama.cpp already has it ON (2026-08-02)

SGLang's Paged-Experts got 8→197 t/s "largely from CUDA-graph capture over the streamed decode path"
(LANDSCAPE §4) — because its offload path had graphs OFF (#23664) and re-enabling them was the win.
§B4 asked whether llama.cpp leaves the same lever on the table, and whether *pinning* is what enables
graph capture (the fork's suspected hidden value). Source inspection said graphs are compiled in
(`GGML_CUDA_GRAPHS=ON`), the 3090 (Ampere > Volta) is not arch-blocked, and the compatibility check
keeps them ON for quantized MoE decode at batch=1 — so they should already be active. The A/B (one
binary MASTER_BIN, toggled only by `GGML_CUDA_DISABLE_GRAPHS`, qwen36-35B at ncmoe=6, n=4) measures
exactly what they buy:

| CUDA graphs | decode |
|---|---:|
| OFF (`GGML_CUDA_DISABLE_GRAPHS=1`) | ~79 t/s |
| ON (default) | ~100 t/s |
| **Δ** | **+26.8%** (Cliff +1.0, CI95 [+20.5,+21.8], MAD 0.73) |

**Two settled facts:**
1. **CUDA graphs are a +27% decode lever here** — as big as MTP's benchmark gain, and the same kind of
   win SGLang reported (killing per-kernel launch overhead matters a lot at batch=1 decode).
2. **llama.cpp has them ON by default**, so every decode number in this project — §E1's ~102 t/s, §E4's
   MTP ~116 t/s — ALREADY banks it. The SGLang-style graph win is **not an untapped lever here; it is
   already captured.** The +27% is what you would LOSE by disabling graphs, not a gain waiting to be
   claimed.

**The pinning-enables-graphs hypothesis is FALSIFIED for llama.cpp.** Graphs delivered +27% on plain
MASTER_BIN with **no pinning** (`GGML_CUDA_REGISTER_HOST` unset) — graph capture is gated by GPU arch
and op-compatibility, not host-memory pinning; and pinning is null at ncmoe=6 anyway (§E1), so there
is no pin×graph interaction to chase at the operating point. Prefill unmoved (−1.1%, within noise —
compute-bound, large batch, nothing for graphs to save). This also reframes the engine race: part of
why stock llama.cpp (stream-to-GPU) stays competitive is that **its CUDA graphs are already working at
our placement** — a rival engine must beat a 100 t/s that already banks the graph win. Raw:
`runs/ab-b4graph-qwen36-35b-b4-cudagraph-ncmoe6/`, arm-set `b4graph` in `ab_isolate.py`.

---

## §B3 — GPU-idle instrumentation: the placement penalty is a stall, and it's now a harness metric (2026-08-02)

Every decode number in this project is a black box on WHY a placement costs what it does. §B3 adds the
missing instrument: the harness now samples `utilization.gpu` every second during the serving window and
reports mean **%busy** (idle% is its complement) alongside t/s, per config. Serving-window only — the
accumulator is gated on `mark_healthy()`, so the load transient (card near-idle while 21 GB streams in)
never dilutes the mean. One extra CSV column on the query the guard already runs; code in
`collectors/host.py` (`HostSample.gpu_util_pct`), `control_plane/guard.py` (`Watch.gpu_util_mean`),
`workloads/throughput.py` (`RunResult.gpu_util_mean`). Validated on a null-arms placement sweep
(MASTER, ncmoe {6,24,40}, n=2), `runs/ab-null-qwen36-35b-b3idle/`:

| ncmoe | GPU busy | GPU idle | decode |
|---:|---:|---:|---:|
| 6  | ~60% | ~40% | ~98 t/s |
| 24 | ~37% | ~63% | ~53 t/s |
| 40 | ~37% | ~63% | ~31 t/s |

(base vs the identical `same` arm agree within ~1pp at every dose — the metric is reproducible.)

**Three facts:**
1. **Idle% tracks the placement transition.** Idle climbs 39%→63% as offload deepens 6→24, exactly
   mirroring the 98→53 t/s halving. That **+24pp of idle is the PCIe expert-transfer stall** that §E1's
   placement lever removes by bringing experts back onto the card — the mechanism, now measured, not
   inferred.
2. **Even at the OPTIMUM the 3090 sits ~40% idle.** Batch-1 A3B decode activates only ~3B params/token;
   the card finishes each token's math and waits on memory. So ~40% of decode is intrinsic
   bandwidth-bound idle present even at ncmoe=6 (little of it PCIe — most weights are resident there),
   and the offload penalty stacks PCIe-stall idle ON TOP. This is why a prefetch that fills
   *expert-transfer* idle can only ever recover a slice, and why it is a **tax at ncmoe=6**: at the good
   placement there is barely any expert-transfer idle to fill, only VRAM/KV-bandwidth idle it cannot touch.
3. **The metric floors in deep offload.** ~37% busy at BOTH ncmoe 24 and 40 though decode halves again
   (53→31 t/s). `utilization.gpu` is a coarse "was a kernel resident this window" duty, not SM-occupancy,
   so it resolves the **deploy-relevant** 6↔24 range cleanly but saturates in the heavy-offload tail —
   finer resolution there needs DCGM/Nsight SM-activity counters, not the nvidia-smi field. Recorded as a
   known limit of the instrument, not smoothed over.

**Reconciles §B1's tax vs the 3060's win:** a 16 GB card is forced to ncmoe≥32, deep in the high-idle
regime where the GPU genuinely stalls on expert transfers — prefetch has idle to fill, so it wins. Our
3090 runs at ncmoe=6 with little fillable expert-idle — prefetch is a tax. Same mechanism, opposite sign,
set by where on this curve the card operates.

---

## §B5 — `--pin-hot-experts`: N/A on this box, and the precondition is measured absent (2026-08-02)

`--pin-hot-experts` is a proposed lever that `mlock()`s the hottest MoE experts so the OS page cache
cannot evict mmap'd expert weights **to disk when the model exceeds RAM**. The PR's own summary: *minor
benefit when the model fits in RAM; the value is disk-paging under over-capacity.* Two reasons it is a
non-lever here, one structural and one measured:

1. **Unmerged and experimental.** PR #25932 is **closed** (superseded); its successor **#26414 is open,
   not merged**. No built binary in this project has the flag. Testing it means building an experimental
   branch — justified only if the precondition holds.
2. **The precondition does NOT hold on this box — measured, not assumed.** `probe_b5_spill.sh` launches
   the deploy model at the **heaviest** placement (ncmoe=40, all ~18 GB of experts resident on the CPU,
   VmRSS 21.2 GB) and counts major page faults across back-to-back steady-state decodes:

   | decode | major faults |
   |---:|---:|
   | 1 (cold) | 23 |
   | 2 | **0** |
   | 3 | **0** |

   The 23 first-pass faults are the one-time lazy-mmap fault-in tail (~0 against ~480k expert-accesses
   per decode); decodes 2–3 take **zero** disk-backed faults. Experts fault in once and **stay resident**
   — they never round-trip to disk. qwen36-35B fits in 64 GB with margin, so the eviction the flag
   prevents simply does not occur.

**Verdict:** no win available and no experimental build warranted. Revisit `--pin-hot-experts` (#26414)
only for a MoE that **exceeds this box's RAM** (the too-big-to-place regime — same door as §E2's "revisit
at 128 GB"). This is the §B3/§E2/§E3 thesis again from the RAM side: a box with abundant RAM and a strong
GPU gets nothing from a lever built to survive RAM over-capacity. Probe: `probe_b5_spill.sh`.

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
