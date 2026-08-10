# LAB-SERVE-001c — Topology Reconciliation + Realistic Open-Loop Traffic

Two sequential stages: **(1) reconcile the 001b topology discrepancy** (gate); **(2) minimal
open-loop realistic-traffic campaign** on the MoE deploy candidate. Owner-approved scope: minimal,
**hard ≤4h**; realism = capped outputs with a per-request cap distribution. Deploy profile NOT modified.
Historical 001/001b raw evidence untouched (corrections are supersession artifacts).

---

## STAGE 1 — TOPOLOGY_RECONCILIATION = **RESOLVED**

**Discrepancy:** 001b prose said `n_ctx=8192` shared / slot0 8192; committed `blocks.json` recorded
`slot0_n_ctx=2048, default_gen_n_ctx=2048, n_ctx=null` under argv `--ctx-size 8192 --parallel 4`.

**Root cause (pinned fork `068764d92`, cited from source, not prose):**
- `tools/server/server.cpp:146-151` sets `n_parallel=4` **and** `kv_unified=true` **only** inside
  `if (params.n_parallel < 0)` — the *auto* case. 001b passed `--parallel 4` **explicitly**, so this
  branch never ran and `kv_unified` kept its default **false** (`src/llama-context.cpp:3498`).
- `src/llama-context.cpp:286-300`: with `kv_unified=false`, `n_ctx_seq = n_ctx / n_seq_max = 8192/4
  = 2048` (static KV partition). With `kv_unified=true` it would be `n_ctx_seq = n_ctx = 8192`.
- The `n_ctx:null` in blocks.json was a capture defect: the probe read `props["n_ctx"]`, but `/props`
  has **no** top-level `n_ctx` (only `default_generation_settings.n_ctx`). The effective 2048 *was*
  captured in `default_gen_n_ctx`/`slot0_n_ctx`.

**Empirical confirmation** (re-ran the pinned server twice; raw in `topology/`):
| config | startup log | /props dg.n_ctx | /slots per-slot |
|---|---|---|---|
| `--parallel 4` (001b) | `n_slots=4, n_ctx_slot=2048, kv_unified='false'` | 2048 | [2048×4] |
| auto (no `--parallel`) | `n_slots=4, n_ctx_slot=8192, kv_unified='true'` | 8192 | [8192×4] |
The auto case reproduces the earlier "8192" prose; the explicit case reproduces blocks.json exactly.

**Reconciled values (001b config):** configuredGlobalContext=8192 · serverSlotCount=4 ·
effectiveSlotContext=**2048** · sharedKvCapacity = *not shared* (4 static cells of 2048) ·
maxContextForOneActiveSlot=2048 · maxContextForFourActiveSlots=2048/slot.

**Impact on 001b — NONE.** The 001b workload was 1024+128 = **1152 tok/request**, inside the 2048
per-slot envelope (896 margin). Static partition ⇒ each slot kept its full 2048 even at N=4 ⇒ no
truncation. Both arms shared the identical topology ⇒ paired deltas unbiased. **001b metrics stand;
the correction is purely descriptive** (4×2048 partitioned, not 8192 shared). Supersession:
`runs/serving/LAB-SERVE-001b/TOPOLOGY_CLARIFICATION.md`; machine-readable `topology/topology_interpretation.json`.

---

## STAGE 2 — Open-loop realistic traffic (MoE deploy candidate)

### Design (pre-registered BEFORE results — `../PRE_REGISTRATION.md`)
- **Server (both arms):** `-fa on --ctx-size 73728 --parallel 4 --jinja --n-cpu-moe 8 -ctk q8_0 -ctv q8_0`
  → 4 slots × **18432 tok/slot** (verified in-run; VRAM ~20.4 GB). Arms differ ONLY by
  `--spec-type draft-mtp --spec-draft-n-max 4` (ON). This is a NEW, explicitly-chosen envelope sized
  for the coding class — the 001b 2048/slot would NOT fit it (Stage 1 consequence).
- **Open-loop:** NO `--max-concurrency` cap (the 4 slots queue the excess); load set by Poisson
  `--request-rate`; `--seed` fixes arrivals **and** prompt shuffle → **identical arrival schedule per
  arm** within a rep (§14). Natural-EOS path on (`--disable-ignore-eos`), but caps bind (below).
- **Workload (pinned, `workload/workload_001c.jsonl`, sha in manifest):** 120 items, 2 disjoint-band
  classes — INTERACTIVE (prompt 532–4091 tok, caps {128,256,512,1024}), CODING (prompt 8234–9201 tok,
  caps {512,1024,1536}). Class recoverable per request from `input_len` band.
- **Realism caveat (owner-approved):** the deploy MoE is a **reasoning model**; its `<think>` output
  exceeds the packet's nominal caps, so **outputs pin to their cap** — this is *cap-bounded*, not
  natural-EOS. The cap *distribution* supplies the §20 length strata by design. Documented deviation from §9.
- **Load points** from a saturated capacity probe (C≈0.085 req/s): LOW 0.030 / NEAR 0.072 / OVERLOAD
  0.110 req/s. **2 paired blocks × 2 arms × 3 points = 12 server-starts.** n=2 ⇒ direction+magnitude,
  **no p<0.05** (minimal scope). Hard 4h guard + per-cell timeout.

### Execution — all clean
12/12 cells succeeded (rc 0), **0 request errors**, token accounting **1.000** every cell, elapsed
**48.5 min** (`stopped_on_budget=false`). Every cell sampled the same 24 prompts (seeded) ⇒ identical
work (out-len mean 650.7 everywhere); only arrival rate varies ⇒ a clean controlled comparison.
OPEN_LOOP_INSTRUMENT: **QUALIFIED**.

### Load curve (pooled 2 reps = 24 req per point×arm)
| point | arm | offered | completed | TTFT med | TTFT p95 | E2E med | E2E p95 | TPOT med | J/out-tok* |
|---|---|---|---|---|---|---|---|---|---|
| LOW  | off | 0.030 | 0.031 | 2.21 s | 8.6 s | 11.66 s | 20.95 s | 10.55 ms | 5.07 |
| LOW  | on  | 0.030 | 0.031 | 2.24 s | 8.8 s | 9.84 s | 18.56 s | 9.27 ms | 4.83 |
| NEAR | off | 0.072 | 0.069 | 2.03 s | 10.2 s | 16.04 s | 40.84 s | 15.78 ms | 3.72 |
| NEAR | on  | 0.072 | 0.070 | 2.22 s | 10.5 s | 12.71 s | 32.23 s | 15.28 ms | 3.57 |
| OVER | off | 0.110 | **0.090** | 2.09 s | 14.1 s | 20.79 s | 42.46 s | 23.56 ms | 3.32 |
| OVER | on  | 0.110 | **0.091** | 2.24 s | 15.7 s | 19.19 s | 38.64 s | 18.06 ms | 3.21 |
*Energy OBSERVED_ONLY (window includes warmup).

### Queueing onset — QUEUEING_ONSET: **ESTIMATED**, OVERLOAD_BEHAVIOR: **CHARACTERIZED**
- `completed` tracks `offered` at LOW (0.031/0.030) and NEAR (0.069/0.072), but at OVERLOAD **completed
  0.090 << offered 0.110** → the system cannot keep up. **Sustainable capacity ≈ 0.09 req/s; onset
  between NEAR (0.072) and OVERLOAD (0.110).**
- E2E median rises monotonically with load (off 11.7→16.0→20.8 s); E2E p95 jumps LOW→NEAR (21→41 s)
  then plateaus at the saturation ceiling. TTFT **p95** grows with load (off 8.6→10.2→14.1 s) while
  TTFT **median** stays ~2 s — so queue delay lives in the tail. Per §15 we do **not** relabel TTFT
  median as queue delay; the TTFT-p95 inflation over the LOW baseline is the (labeled, derived) queue signal.
- §18: at OVERLOAD, `completed<offered` + TTFT p95 (14 s) ≫ prefill (~2 s) ⇒ requests waited for a free
  slot ⇒ outstanding demand exceeded the 4 slots. (Inferred from these signals; instantaneous slot
  occupancy was not directly instrumented — stated as inference, not measurement.)

### MTP effect (paired ON−OFF, 2 reps, all direction-agree unless noted) — MTP is the practical winner
| metric | LOW Δ | NEAR Δ | OVER Δ |
|---|---|---|---|
| **E2E median** | −1.48 s | −3.14 s | −2.61 s |
| **E2E p95**    | −2.44 s | −4.98 s | −1.89 s |
| **TPOT median**| −1.74 ms | −1.07 ms | **−4.66 ms** |
| TTFT median    | +24 ms | +135 ms | +164 ms |
| TTFT p95       | +105 ms (mixed) | −13 ms (mixed) | **+2.09 s** |
| completed rate | ~0 | ~0 | ~0 |

- **MTP improves E2E median AND p95 at every load point** (2/2 direction) — the user-visible completion
  latency benefit dominates. LOW_LOAD_MTP: **BETTER**; NEAR_CAPACITY_MTP: **BETTER**.
- **MTP improves TPOT at every point**, including −4.66 ms at OVERLOAD. Cost: a small TTFT-median
  penalty (+24…+164 ms, draft setup) and a TTFT-p95 penalty **only** at OVERLOAD (+2.1 s).
- MTP does **not** change sustainable capacity (~0.09 req/s both arms).

### The 001b MoE N=4 TPOT paradox under realistic arrivals — MOE_TPOT_REAL_WORLD_IMPACT: **MASKED**
001b (closed-loop, forced 128-tok, N=4) found MTP **worse** on TPOT (+8.5 ms). Under 001c open-loop
Poisson traffic, TPOT was **better** with MTP at every load point (incl. −4.66 ms at OVERLOAD), and E2E
strictly favored MTP. So the 001b N=4 TPOT penalty did **not** translate into worse user-visible latency
— it is **masked** under realistic arrivals. **Hypothesis (NOT proven):** sustained exactly-4-way decode
(where draft-verify most competes with expert routing) is rare under Poisson arrivals compared with a
saturated closed-loop N=4; isolating that mechanism needs a controlled concurrency sweep (future packet).
We do **not** claim expert routing as the proven cause (consistent with the 001b caveat).

### Fairness / per-class (§17)
| point | arm | INT TTFT | INT E2E | COD TTFT | COD E2E |
|---|---|---|---|---|---|
| LOW  | off | 2.19 s | 9.09 s | 8.43 s | 16.86 s |
| OVER | off | 2.00 s | 15.31 s | 8.27 s | 27.85 s |
| OVER | on  | 2.18 s | 11.93 s | 9.02 s | 26.65 s |
Interactive TTFT stays ~2 s and coding TTFT ~8.4 s **even at OVERLOAD** — neither class is starved by
the other; both classes' E2E degrade proportionally with load (no catastrophic head-of-line). MTP helps
both classes' E2E. **Caveat:** only ~3 coding req/cell (6/point) — class-level numbers are thin.

### Output-length stratification (§20) — PARTIAL
Because caps bind and the seeded sample is identical across cells, every cell carries the same length
mix (bins present: 128/256/512/1024/1536) — so a *within-run* MTP-vs-length interaction cannot be
cleanly isolated here (same mix everywhere). The length *distribution* is captured; a dedicated
length-swept run is the clean way to test whether MTP's benefit scales with output length. Recorded as
a limitation, not a result.

### Energy — ENERGY: **OBSERVED_ONLY**
MTP lower J/out-token at every load point (LOW 4.83 vs 5.07, NEAR 3.57 vs 3.72, OVER 3.21 vs 3.32,
~3–5%). Window includes warmup ⇒ not qualified; deferred to LAB-ENERGY.

---

## Status outcomes (§23)
- TOPOLOGY_RECONCILIATION: **RESOLVED**
- OPEN_LOOP_INSTRUMENT: **QUALIFIED**
- LOW_LOAD_MTP: **BETTER** · NEAR_CAPACITY_MTP: **BETTER**
- OVERLOAD_BEHAVIOR: **CHARACTERIZED** (capacity ~0.09 req/s; completed<offered; tail inflation)
- MOE_TPOT_REAL_WORLD_IMPACT: **MASKED** (open-loop TPOT favored MTP; 001b N=4 penalty did not appear)
- QUEUEING_ONSET: **ESTIMATED** (~0.09 req/s; onset between 0.072 and 0.110)
- ENERGY: **OBSERVED_ONLY**

## Limitations
n=2 reps (direction+magnitude, no p-values). Outputs cap-bounded (reasoning model), not natural-EOS —
so this characterizes *capped* open-loop serving. Coding-class n thin (~6/point). Single deploy MoE;
no dense reference this packet. Length×MTP interaction not isolable within this run. Capacity is low and
real (heavy reasoning outputs + 8k coding prefill on the CPU-offloaded MoE). Deploy profile unchanged.

## Reproduce
```
# Stage 1
bash (topology capture, see topology/*.log)                # dual-config /props+/slots
# Stage 2
/home/augus/sglang-venv/bin/python ops/lab_serve_workload_gen.py --tokenizer /home/augus/models/fp16/base \
    --outdir runs/serving/LAB-SERVE-001c/workload
bash ops/lab_serve_openloop_replicate.sh                    # 12 server-starts, ≤4h guard
PYTHONPATH=src python ops/lab_serve_openloop_analyze.py runs/serving/LAB-SERVE-001c/campaign/raw
```

## One recommended next packet
**LAB-SERVE-001d — controlled concurrency isolation of the MoE draft-verify vs expert-routing TPOT
interaction:** closed-loop concurrency sweep N∈{1,2,4,6,8} at fixed output lengths, MTP on/off, to test
the 001b-vs-001c TPOT-sign hypothesis directly (is the N=4 penalty specific to sustained 4-way decode?),
plus a length-swept cell to settle §20. Keep it closed-loop (isolation), leave open-loop realism to 001c.
