# Wave B — LAB-SERVE-001b Handoff (2026-08-10)

Calibration/replication packet: server-topology resolution, dense-27B variance, MTP semantics, MoE
transfer. Read-only except committed code/evidence. Mobile-delivered: this file + the report +
combined matrix CSV/JSON + an evidence ZIP are attached to the chat response.

## 1. Git state (WA-CLOSE-001 terminology)
- Branch **master**. **Starting HEAD `2c81cb7`** → **Ending HEAD `a522279`**.
- `trackedTreeClean=true` · `stagedTreeClean=true` · `untrackedArtifactsPresent=true` (`.harness/`
  handoffs) · overall = clean tracked tree + intentional untracked handoffs. **Not pushed.**
- Session commits:
```
a522279 bench(serving): LAB-SERVE-001b-B MoE transfer + report + 001 supersession note
73cbd12 bench(serving): LAB-SERVE-001b-A dense-27B variance (5 paired blocks, N=1/4)
49e3aeb feat(serving): LAB-SERVE-001b paired-block replication + block-level analysis
```

## 2. Server topology — QUALIFIED (§2–3)
Source: `llama.cpp-master/tools/server/server.cpp:146-149` → `n_parallel` auto→4, `kv_unified=true`.
Empirical `/props`: `total_slots=4, n_ctx=8192`; `/slots`: 4 slots, slot0 n_ctx 8192. Exact server
argv (captured per block in `*/raw/blocks.json`), on vs off differ ONLY by the MTP flags:
```
llama-server -m <gguf> --host 0.0.0.0 --port 8080 -fa on --ctx-size 8192 --jinja \
  <-ngl 99 (dense) | --n-cpu-moe 8 --cache-type-k q8_0 --cache-type-v q8_0 (moe)> --parallel 4 \
  [--spec-type draft-mtp --spec-draft-n-max 4]      # <- the ONLY on/off difference
```
Recorded distinct: `clientOutstandingConcurrency` (1 or 4) · `serverSlotCount`=4 ·
`activeDecodeConcurrency`=min(N,4). N≤4 ⇒ no queuing. **This re-frames LAB-SERVE-001: it was 4-slot
batching, not 1-slot queuing.**

## 3. Statistical unit & analysis (§10–11)
Unit = independent **server-level block** (one server start); 5 blocks/cell; MTP arm order alternated
per block (`schedule.json`). Per (N, metric): pair on/off BY REP → 5 paired deltas → Hodges–Lehmann
delta + seeded percentile bootstrap CI + sign test (`analysis/robust.py`). n=5 ⇒ sign floor p=0.0625;
we report **CI-excludes-0 + 5/5 direction**, not p<0.05. p99 kept but descriptive only.

## 4. Metric-by-metric verdict (paired Δ = on−off; CI excludes 0 unless noted)
**Dense-27B (fable-tc-l1.0):**
| N | thr Δ | TPOT Δ ms | E2E Δ ms | TTFT Δ ms | J/tok Δ |
|---|---|---|---|---|---|
| 1 | +10.7 | −7.8 | −900 | +108 | −2.96 |
| 4 | +19.1 | **−5.8** | −2154 | −1404 | −1.81 |

**MoE-35B (qwen36-35b-mtp, ncmoe=8, q8 KV):**
| N | thr Δ | TPOT Δ ms | E2E Δ ms | TTFT Δ ms | J/tok Δ |
|---|---|---|---|---|---|
| 1 | +8.3 | −2.21 | −305 | +29 (CI incl 0) | −0.27 |
| 4 | +6.8 | **+8.5** | −729 | −1817 | −0.24 |

## 5. Superseded interpretation from LAB-SERVE-001 (§1)
`runs/serving/LAB-SERVE-001/SUPERSEDED_INTERPRETATION.md` (001 raw evidence untouched): "MTP loses on
latency at N≥4" was too broad. Use **TPOT crossover**. Under 001b the dense **TPOT crossover does NOT
replicate** (was 1-rep noise); MTP wins dense TPOT at both N. The MoE **does** show it (TPOT +8.5 ms
at N=4). E2E median favored MTP in 001 at all N — retained.

## 6. Token accounting (§8)
Forced length + exact tokenizer (`fp16/base`). MoE 35B GGUF has 248320 tokens vs 248044, but the
extra ~276 are unused specials → **MoE canary retokenized EXACTLY (1024/1024)** ⇒ accounting-
qualified. Dense: ratio 1.000 all cells, **bit-exact 11/20** (≤0.1% retokenization boundary drift in
9 cells — reported, not silently accepted at ±10%). MoE: **bit-exact 20/20**.

## 7. MTP acceptance observability (§12)
Observable via the native `/completion` timings `draft_n`/`draft_n_accepted` (NOT the OpenAI path,
NOT bench's sglang-specific `accept_length`). Median accept: dense 42.5%, MoE 51.2% (temp-0 probe).
Non-invasive; no new instrumentation built.

## 8. Energy signal (§13)
`energy_J ≈ power_mean_w × wall_s` (sampler window ≈ bench window, but includes warmup). MTP lower
J/output-token on both architectures (dense −23/−29%, MoE −6/−9%), CI excludes 0.
**ENERGY_SIGNAL: OBSERVED_ONLY** — integration window not exactly the scored interval (§13 gate).

## 9. Status outcomes (§19)
- SERVER_TOPOLOGY: **QUALIFIED** · DENSE_VARIANCE: **ESTABLISHED**
- DENSE_MTP_TPOT_CROSSOVER: **NOT_SUPPORTED** · DENSE_MTP_E2E_EFFECT: **SUPPORTED**
- MOE_TRANSFER: **DIFFERENT** (E2E/throughput transfer; TPOT does not — MoE shows the crossover)
- ENERGY_SIGNAL: **OBSERVED_ONLY**

## 10. Failures / negative evidence (§ honesty)
- **Two dense runs invalidated** by launch-time plumbing bugs, both fixed, both documented:
  (a) Windows `str(Path)` gave a backslash `--outdir` that broke inside WSL bash; (b) Git Bash
  MSYS path-conversion rewrote `--tokenizer /home/augus/...` → `C:/Program Files/Git/home/...`, and
  the space broke argparse. Fix: `outdir.as_posix()` + launch with `MSYS_NO_PATHCONV=1`. The
  measurement code (`lab_serve_bench.py`) worked standalone throughout — proven by isolated repro.
  Invalid data was discarded and re-run (raw `*.debug.txt` kept as diagnostic evidence).
- My naïve `setsid nohup` detach died earlier; used `lmctl serve` (correct holder detach).

## 11. Commands & material outputs
```
# topology
curl /props -> total_slots 4, n_ctx 8192 ; curl /slots -> 4 slots
# acceptance
curl /completion (n_predict 128) -> timings.draft_n 48, draft_n_accepted 15..  (42.5% / 51.2%)
# replication (per arch)
MSYS_NO_PATHCONV=1 python ops/lab_serve_replicate.py --serve-target <t> --tokenizer /home/augus/models/fp16/base \
   --base-extra "<flags> --parallel 4" --outdir runs/serving/LAB-SERVE-001b/<dense|moe>/raw --reps 5
PYTHONPATH=src python ops/lab_serve_analyze.py runs/serving/LAB-SERVE-001b/<dense|moe>/raw <label>
# MoE token gate canary -> completed 8/8, 1024/1024 EXACT (QUALIFIED)
```

## 12. Source excerpts consulted
- `server.cpp:146-149` — `if (params.n_parallel < 0) { ... params.n_parallel = 4; kv_unified=true; }`
- `sglang/benchmark/serving.py:520` — "Reasoning models stream thoughts via reasoning_content; count
  them like content" (why thinking-model token accounting is sound).
- `ops/lab_serve_analyze.py::(paired loop)` — `deltas=[on[r]-off[r] for r in reps]; bootstrap_ci(...,
  statistic=hodges_lehmann); sign_test_p(deltas)` (block-level pairing).

## 13. Raw evidence paths
`runs/serving/LAB-SERVE-001b/{dense,moe}/raw/*` (per-block blocks.json, per-cell normalized.json +
jsonl + stdout + argv + debug.txt), `.../{dense,moe}/normalized/{matrix,paired_effects,summary}.*`,
`.../normalized/combined_paired_effects.{csv,json}`, `report/LAB-SERVE-001b.md`,
`runs/serving/LAB-SERVE-001/SUPERSEDED_INTERPRETATION.md`. Committed diffs: `git show 49e3aeb 73cbd12 a522279`.

## 14. Rollback / reproduction
Rollback whole packet: `git reset --hard 2c81cb7` (nothing pushed). Reproduce: §11 commands (GPU;
serve → replicate → analyze; ~20 min dense, ~30 min MoE). Analyzer is pure (no GPU) over blocks.json.

## 15. Exactly one recommended next packet
**LAB-SERVE-001c — Realistic Traffic (open-loop):** normal EOS, finite Poisson arrival rates below /
near / above sustainable capacity, mixed short-interactive + coding-like prompt lengths; measure
queue delay, tail latency, timeout, fairness — and whether the **MoE N=4 TPOT crossover** matters
under realistic arrivals or is masked by E2E/TTFT gains. Keep N=8 (queuing) as an explicit cell.
Do NOT touch the deploy profile; this remains characterization.
