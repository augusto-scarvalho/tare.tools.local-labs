# HANDOFF — Serving Characterization Campaign CLOSURE (2026-08-10)

Durable closure of the Local AI Lab serving-characterization campaign:
**Wave A → LAB-SERVE-QA-001 → LAB-SERVE-001 → LAB-SERVE-001b → LAB-SERVE-001c.**
Housekeeping/closure packet only — no GPU runs. Deploy profile unchanged throughout. Not pushed.

---

## 1. Git identities
- Branch **master**.
- **Starting HEAD `a522279`** (end of LAB-SERVE-001b) → **Final HEAD `4f4e459`**.
- Closure commit: **`d0e55c1fc5ee1e4134cb5f10996e73ec972aae9b`**
  (`bench(serving): close LAB-SERVE-001c open-loop characterization`).
- Housekeeping commit (after closure): **`4f4e459098b4454ed10c36df841a607357e50d78`**
  (`bench(serving): preserve LAB-SERVE-001b combined evidence`) — commits the two canonical derived 001b
  artifacts `runs/serving/LAB-SERVE-001b/normalized/combined_paired_effects.{csv,json}` that were
  reported/attached but had remained untracked; validated by replay against the committed dense/moe
  `paired_effects.json` (52/52 records EXACT match, 0 CSV mismatches). No raw results touched.
- Post-commit: `trackedTreeClean=true` · `stagedTreeClean=true` · **not pushed**.
- `untrackedArtifactsPresent=true`, intentionally: **only** `.harness/` (all handoffs, incl. this file —
  the campaign has kept handoffs untracked throughout). The previously-untracked
  `runs/serving/LAB-SERVE-001b/normalized/combined_paired_effects.{csv,json}` were committed in the
  housekeeping commit `4f4e459` (see above), so the working tree is otherwise clean.

### Relevant commits (oldest→newest)
```
e0155fd feat(bench-qa): single-source harness glue (LAB-QA-001)
1fa3261 test(bench-qa): LAB-QA-001 harness self-test (16 cases, no GPU)
068dc59 feat(bench-qa): capture run identity (LAB-QA-002)
ae5a5d1 feat(promotion): lexicographic promotion semantics (LAB-QA-003)
a47a9c2 docs: Backlog V2 reconciliation + Wave A/P0 status
35a02ef feat(bench-qa): Wave-A closure (evalplus sentinel + identity guard + provenance)
894b341 feat(serving): LAB-SERVE-001 thin adapter + normalizer (vllm-chat)     # LAB-SERVE-QA-001 qualified here
2c81cb7 bench(serving): LAB-SERVE-001 pilot evidence + report
49e3aeb feat(serving): LAB-SERVE-001b paired-block replication + block-level analysis
73cbd12 bench(serving): LAB-SERVE-001b-A dense-27B variance (5 paired blocks, N=1/4)
a522279 bench(serving): LAB-SERVE-001b-B MoE transfer + report + 001 supersession
d0e55c1 bench(serving): close LAB-SERVE-001c open-loop characterization
4f4e459 bench(serving): preserve LAB-SERVE-001b combined evidence                 # FINAL HEAD
```

## 2. Final status table
```
Wave A (LAB-QA-001/002/003 + closure)   DONE
LAB-SERVE-QA-001  serving-harness qual   QUALIFIED
LAB-SERVE-001     saturated pilot        PILOT_COMPLETE
LAB-SERVE-001b    paired replication     COMPLETE  (5 server-blocks, dense+MoE)
LAB-SERVE-001c    open-loop realistic    COMPLETE  (LOW/NEAR/OVERLOAD, 2 blocks, MoE)
LAB-SERVE-001d    concurrency×MTP TPOT   PARKED / OPTIONAL RESEARCH
LAB-SERVE-002     workload profiles      NOT PROMOTED / future
LAB-REL-001       24h soak               DEFERRED
LAB-ENERGY        energy instrumentation DEFERRED
```

## 3. Topology correction (auditable; historical raw evidence untouched)
001c established that the **001b topology *description* was wrong while the 001b *benchmark* stayed valid.**
```
001b configured:  --ctx-size 8192  --parallel 4
effective:        4 static slots × 2048 tokens,  kv_unified = false
```
The prior prose "8192 per slot" is **superseded**. Root cause (pinned fork `068764d92`):
`server.cpp:146-151` sets `kv_unified=true` only in the auto branch (`n_parallel<0`); passing
`--parallel 4` explicitly skipped it, so `kv_unified` kept its default `false`
(`llama-context.cpp:3498`), and `llama-context.cpp:286-300` computed `n_ctx_seq = n_ctx/n_seq_max =
8192/4 = 2048`. Confirmed empirically by re-running both configs (explicit→2048/false, auto→8192/true).
**001b remains valid:** its 1024+128 = 1152-tok workload fit the 2048 envelope (896 margin), static
partition ⇒ full 2048/slot even at N=4 ⇒ no truncation; both arms shared the topology ⇒ deltas unbiased.
Clarification artifact: `runs/serving/LAB-SERVE-001b/TOPOLOGY_CLARIFICATION.md`; machine-readable
`runs/serving/LAB-SERVE-001c/topology/topology_interpretation.json` (+ raw `/props`,`/slots`, startup logs).

## 4. Final serving interpretation (no overclaiming)
For the tested MoE endpoint and **capped open-loop** workload:
- **MTP improved E2E median AND p95 at LOW, NEAR and OVERLOAD** (paired, 2/2 direction; Δ −1.5…−3.1 s
  median, −1.9…−5.0 s p95).
- **MTP also improved TPOT in the open-loop experiment** at every load point (−4.66 ms at OVERLOAD).
- **The closed-loop N=4 TPOT penalty from 001b (+8.5 ms) did NOT translate into worse open-loop
  user-visible completion latency** — masked under realistic Poisson arrivals.
- **Sustainable capacity ≈ 0.09 req/s** for this workload; **queueing onset between 0.072 and 0.110 req/s**
  (at OVERLOAD completed 0.090 < offered 0.110).
- Small cost: TTFT-median penalty (+24…+164 ms) and a TTFT-p95 penalty only at OVERLOAD (+2.1 s).

**Caveats (preserved):** 001c n=2 paired blocks; cap-bounded outputs (reasoning model), not strict
natural-EOS; coding-class sample thin (~6/point); single MoE config; energy OBSERVED_ONLY; queue
occupancy inferred (completed<offered + TTFT-p95≫prefill), not directly instrumented.
**NOT claimed:** expert routing proven as cause · universal MTP superiority · production SLO
qualification · production reliability.

## 5. Superseded interpretations (immutable raw evidence; corrections via artifacts only)
- LAB-SERVE-001 "MTP loses on latency at N≥4" → too broad → **TPOT crossover**
  (`runs/serving/LAB-SERVE-001/SUPERSEDED_INTERPRETATION.md`).
- LAB-SERVE-001 dense TPOT crossover → **did not replicate** in 001b (1-rep noise).
- LAB-SERVE-001b "n_ctx 8192/slot shared" → **corrected to 4×2048 partitioned, kv_unified=false** (§3 here).

## 6. Failures / negative evidence (honesty)
- 001c first workload (16k coding, single 2048 cap) → capacity ~0.06 req/s and every output pinned to
  cap (the deploy MoE is a reasoning model). Trimmed to 8k coding + a per-request cap distribution per
  owner decision; both capacity canaries retained as evidence (`runs/serving/LAB-SERVE-001c/capacity/`).
- Two throwaway sanity-waiter scripts hit nested-quote bugs; they never touched the campaign.
- 001b: two dense runs invalidated by Windows/WSL argv-mangling (fixed: `outdir.as_posix()` +
  `MSYS_NO_PATHCONV=1`); 001c orchestrator is all-WSL to avoid this entirely.
- 001c campaign: 0 request errors, 0 timeouts, token accounting 1.000 every cell; nothing dropped.

## 7. Methods (001c)
Server (both arms): `-fa on --ctx-size 73728 --parallel 4 --jinja --n-cpu-moe 8 -ctk q8_0 -ctv q8_0`
→ 4 slots × 18432 tok/slot (verified in-run). ON adds `--spec-type draft-mtp --spec-draft-n-max 4`.
Open-loop: no `--max-concurrency` (slots queue excess); Poisson `--request-rate`; `--seed` fixes
arrivals+shuffle identically per arm (common schedule). Pinned workload with per-request output-cap
distribution; class recoverable from disjoint input-length bands. Load points pre-registered from a
saturated capacity probe (C≈0.085): LOW 0.030 / NEAR 0.072 / OVERLOAD 0.110. 2 paired blocks × 2 arms ×
3 points; per-rep seeds 101/102; arm order alternated. n=2 ⇒ direction+magnitude, no p<0.05. Hard 4h
guard (used 48.5 min). Pairing/analysis reuse `src/model_lifecycle/analysis/robust.py`.

## 8. Raw evidence locations
- `runs/serving/LAB-SERVE-001/**`, `runs/serving/LAB-SERVE-001b/**`, `runs/serving/LAB-SERVE-001c/**`
  (topology, workload, capacity, campaign/{raw,normalized}, report, PRE_REGISTRATION.md).
- Supersession/clarification: `LAB-SERVE-001/SUPERSEDED_INTERPRETATION.md`,
  `LAB-SERVE-001b/TOPOLOGY_CLARIFICATION.md`.
- Tools: `ops/lab_serve_bench.py` (001/001b), `ops/lab_serve_replicate.py`, `ops/lab_serve_analyze.py`,
  `ops/lab_serve_workload_gen.py`, `ops/lab_serve_bench_openloop.py`,
  `ops/lab_serve_openloop_replicate.sh`, `ops/lab_serve_openloop_analyze.py`.

## 9. Source excerpts (pinned fork 068764d92)
- `tools/server/server.cpp:146-151` — `if (params.n_parallel < 0) { ...=4; kv_unified=true; }`
- `src/llama-context.cpp:286-300` — `kv_unified ? n_ctx_seq=n_ctx : n_ctx_seq=n_ctx/n_seq_max`
- `src/llama-context.cpp:3498` — default `kv_unified = false`

## 10. Exact commands (reproduce 001c; no GPU for analysis)
```
/home/augus/sglang-venv/bin/python ops/lab_serve_workload_gen.py \
    --tokenizer /home/augus/models/fp16/base --outdir runs/serving/LAB-SERVE-001c/workload
bash ops/lab_serve_openloop_replicate.sh                                  # 12 server-starts, ≤4h guard
PYTHONPATH=src python ops/lab_serve_openloop_analyze.py runs/serving/LAB-SERVE-001c/campaign/raw
```

## 11. Verification outputs (this closure, no GPU)
- Serving analyzer pure replay: `cells=12 ok=12 failures=[] timeouts=[]` (reproduces committed numbers).
- Benchmark harness self-test (LAB-QA-001): **16/16 PASS — ALL GREEN**.
- Promotion self-check (LAB-QA-003): **promotion self-check OK**.
- EvalPlus sentinel: skipped (quality-gate tool, unrelated to serving; report already present; not cheap).

## 12. Deferred / parked
- **LAB-SERVE-001d — PARKED/OPTIONAL:** closed-loop concurrency sweep N∈{1,2,4,6,8} at fixed output
  lengths, MTP on/off, to isolate the draft-verify × expert-routing TPOT interaction (test the
  001b-vs-001c TPOT-sign hypothesis) + a length-swept cell for §20. Not required for endpoint
  qualification now; preserved in the backlog.
- DEFERRED: LAB-REL-001 (soak), LAB-ENERGY (qualified J/token), LAB-SERVE-002 (profile promotion).

## 13. Rollback / reproduction
- Rollback this closure commit: `git reset --hard a522279` (nothing pushed). The 001c working files then
  return to untracked; `git clean -nd` to preview any removal.
- Reproduce: §10 (workload gen → orchestrator → analyzer). Analysis is pure/CPU over committed raw.

## 14. Single recommendation
**Close/compact the serving session. Choose the next independent research campaign separately.**
Do NOT auto-execute LAB-SERVE-001d.
