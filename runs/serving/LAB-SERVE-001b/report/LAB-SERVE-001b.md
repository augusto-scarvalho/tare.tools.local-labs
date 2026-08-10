# LAB-SERVE-001b — Variance, Server Topology, MTP Semantics & MoE Transfer

Calibration/replication packet. Supersedes one interpretation of LAB-SERVE-001 (see
`../../LAB-SERVE-001/SUPERSEDED_INTERPRETATION.md`); 001's raw evidence stays immutable.
Statistical unit = the independent **server-level block** (one server start). 5 paired blocks per
cell; MTP arm order alternated per block (schedule in `*/raw/schedule.json`). Forced output length,
exact-token calibration workload. Metrics: median/p95 + block-level Hodges–Lehmann paired delta with
seeded bootstrap CI (`analysis/robust.py`). n=5 → the sign-test floor is p=0.0625 (2·0.5⁵); we
emphasize **CI-excludes-0 + 5/5 direction agreement**, not p<0.05 (§10–11).

## Server topology — QUALIFIED (§2–3)
`/props`+`/slots` on the pinned lifecycle fork (`068764d92`): **total_slots=4, kv_unified=true,
n_ctx=8192 shared**, `-fa on`. The fork defaults `n_parallel` auto→4 (`server.cpp:146-149`), so
LAB-SERVE-001 (no `--parallel`) already had 4 slots — it was NOT 1-slot queuing. Distinct concepts,
recorded per block: `clientOutstandingConcurrency` (1 or 4) · `serverSlotCount`=4 ·
`activeDecodeConcurrency`=min(N,4). Both 001b arms pin `--parallel 4`; at N=1 one slot is active, at
N=4 all four (no queuing — N≤slots). Exact argv per block in `*/raw/blocks.json`; the on/off argv
differ only by `--spec-type draft-mtp --spec-draft-n-max 4` (mechanically confirmed).

## Token accounting (§8)
Forced length + exact tokenizer (`fp16/base` for both — the MoE's 35B GGUF has 248320 tokens vs
248044, but the extra ~276 are unused specials: the **MoE canary retokenized EXACTLY**, so it is
accounting-qualified). Dense: ratio 1.000 all cells, **bit-exact in 11/20** (retokenization boundary
differs by ≤~0.1% in 9 cells — reported honestly, not silently accepted at the generic ±10%). MoE:
**bit-exact in 20/20**.

## MTP acceptance — observable (§12)
Via the **native `/completion` timings** (`draft_n`, `draft_n_accepted`) — NOT via the OpenAI path
bench uses, and NOT via bench's sglang-specific `accept_length`. Median accept rate: **dense 42.5%,
MoE 51.2%** (deterministic probe, temp 0). Non-invasive; no new instrumentation built.

## Dense-27B (fable-tc-l1.0) — paired block deltas (on−off), 5 blocks
| Metric | N=1 on / off / Δ | N=4 on / off / Δ | CI excl 0 |
|---|---|---|---|
| output throughput (tok/s) | 45.8 / 34.6 / **+10.7** | 85.8 / 66.9 / **+19.1** | both yes |
| TPOT median (ms) | 13.8 / 21.6 / **−7.8** | 35.1 / 39.7 / **−5.8** | both yes |
| E2E median (ms) | 2822 / 3747 / **−900** | 5510 / 7617 / **−2154** | both yes |
| TTFT median (ms) | 1107 / 992 / +108 | 1191 / 2618 / **−1404** | both yes |
| J / output token | 9.70 / 12.65 / **−2.96** | 4.50 / 6.26 / **−1.81** | both yes |

**Dense: MTP wins throughput, TPOT, E2E at BOTH N.** The LAB-SERVE-001 "TPOT crossover at N=4"
**does NOT replicate** — it was 1-rep noise. TTFT is the only mixed axis (MTP +108 ms at N=1 = draft
setup; −1404 ms at N=4).

## MoE-35B (qwen36-35b-a3b-mtp, ncmoe=8, q8 KV) — paired block deltas (on−off), 5 blocks
| Metric | N=1 on / off / Δ | N=4 on / off / Δ | CI excl 0 |
|---|---|---|---|
| output throughput (tok/s) | 66.9 / 58.5 / **+8.3** | 97.2 / 90.3 / **+6.8** | both yes |
| TPOT median (ms) | 7.84 / 10.07 / **−2.21** | 31.5 / 23.2 / **+8.5** | both yes |
| E2E median (ms) | 1948 / 2276 / **−305** | 4804 / 5539 / **−729** | both yes |
| TTFT median (ms) | 1012 / 1004 / +29 | 775 / 2589 / **−1817** | N=1 **no** / N=4 yes |
| J / output token | 4.07 / 4.42 / **−0.27** | 2.48 / 2.73 / **−0.24** | both yes |

**MoE: MTP wins throughput & E2E at both N, but LOSES TPOT at N=4 (+8.5 ms, CI [7.56, 9.54]).**
This is the §9 non-contradiction: at MoE N=4 MTP **improves total completion time (E2E −729 ms) and
TTFT (−1817 ms) while worsening steady-state per-token latency (TPOT +8.5 ms)** — draft-verify
compute competes with the MoE's expert routing under 4-way batching.

## Architecture comparison (§16)
| Question | Answer |
|---|---|
| Same sign — throughput? | **Yes** (MTP wins both architectures, both N) |
| Same sign — E2E median? | **Yes** (MTP wins both, both N) |
| Same sign — TPOT? | **No** — dense: MTP better at N=4 (−5.8); MoE: MTP worse at N=4 (+8.5) |
| Same crossover region? | dense: **no** TPOT crossover in N≤4; MoE: TPOT crossover **between N=1 and N=4** |
| Different resource cost? | MoE higher absolute throughput; MTP energy gain larger on dense (−23/−29% J/tok) than MoE (−6/−9%) |

**The TPOT crossover is architecture-dependent** — absent on the dense, present on the MoE. E2E and
throughput effects DO transfer (MTP wins on both). Not asserted as causal beyond what the paired
design supports.

## Status outcomes (§19)
- SERVER_TOPOLOGY: **QUALIFIED** (4 slots, kv_unified, 8192 shared)
- DENSE_VARIANCE: **ESTABLISHED** (5 blocks, tight CIs, MAD small)
- DENSE_MTP_TPOT_CROSSOVER: **NOT_SUPPORTED** (MTP wins TPOT at N=1 and N=4)
- DENSE_MTP_E2E_EFFECT: **SUPPORTED** (MTP wins E2E median+p95 at both N)
- MOE_TRANSFER: **DIFFERENT** (E2E/throughput transfer; TPOT does not — MoE shows the crossover)
- ENERGY_SIGNAL: **OBSERVED_ONLY** (MTP lower J/token on both; integration window includes warmup → not qualified)

## Caveats
n=5 blocks (sign-test floor p=0.0625; CI + direction used instead). Only N=1,4 (no queuing regime;
N=8 remains 001 1-rep descriptive). Forced length (calibration, not realistic EOS). Dense and MoE
share the `fp16/base` tokenizer (MoE qualified via exact canary). Energy window includes warmup.

## Reproduce
```
MSYS_NO_PATHCONV=1 python ops/lab_serve_replicate.py --serve-target <fable-tc-l1.0-q4|qwen36-35b-mtp-q4> \
  --model <m> --tokenizer /home/augus/models/fp16/base --base-extra "<topology+moe flags>" \
  --outdir runs/serving/LAB-SERVE-001b/<dense|moe>/raw --reps 5
PYTHONPATH=src python ops/lab_serve_analyze.py runs/serving/LAB-SERVE-001b/<dense|moe>/raw <label>
```
