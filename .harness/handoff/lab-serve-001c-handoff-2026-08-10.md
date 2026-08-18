# LAB-SERVE-001c Handoff — Topology Reconciliation + Open-Loop Realistic Traffic (2026-08-10)

Two stages: (1) resolve the 001b topology discrepancy (gate); (2) minimal open-loop campaign on the MoE
deploy candidate. Owner scope: minimal, **hard ≤4h**; capped-output realism. Deploy profile untouched.
Mobile-delivered: this file + report + normalized CSV + summary JSON + evidence ZIP attached to the chat.

## 1. Git state
- Branch **master**. **Starting HEAD `a522279` → Ending HEAD `a522279`** (nothing committed; all 001c work
  is untracked pending your go-ahead — you have historically kept commits local/manual).
- Untracked additions: `ops/lab_serve_workload_gen.py`, `ops/lab_serve_bench_openloop.py`,
  `ops/lab_serve_openloop_replicate.sh`, `ops/lab_serve_openloop_analyze.py`,
  `runs/serving/LAB-SERVE-001c/**`, `runs/serving/LAB-SERVE-001b/TOPOLOGY_CLARIFICATION.md`,
  `.harness/handoff/lab-serve-001c-handoff-2026-08-10.md`.
- Nothing pushed. No existing files edited except analyzer print fix.

## 2. STAGE 1 — TOPOLOGY_RECONCILIATION = RESOLVED
- The 001b "8192 shared" prose was wrong; the recorded 2048/slot was right. Cause: `--parallel 4`
  passed **explicitly** skips the fork's auto-branch (`server.cpp:146-151`), so `kv_unified` stayed
  default-`false` (`llama-context.cpp:3498`) → `n_ctx_seq = 8192/4 = 2048` (`llama-context.cpp:286-300`).
  `n_ctx:null` was a capture bug (`/props` has no top-level `n_ctx`; use `default_generation_settings.n_ctx`).
- Verified by re-running both configs: explicit→`n_ctx_slot=2048,kv_unified='false'`; auto→`8192,'true'`.
  Raw: `runs/serving/LAB-SERVE-001c/topology/{topology_props_raw.*,topology_slots_raw.*,topology_server_start.*.log,topology_interpretation.json}`.
- **001b remains VALID**: its 1152-tok workload fit inside 2048 (896 margin); static partition ⇒ full
  2048/slot even at N=4 ⇒ no truncation; both arms identical topology ⇒ deltas unbiased. Correction is
  descriptive only. Supersession note: `runs/serving/LAB-SERVE-001b/TOPOLOGY_CLARIFICATION.md`.
- Reconciled values (001b config): configuredGlobalContext=8192 · serverSlotCount=4 ·
  effectiveSlotContext=2048 · sharedKvCapacity=not-shared(4×2048 static) · max-1-slot=2048 · max-4-slot=2048/slot.

## 3. STAGE 2 design (pre-registered before results — `runs/serving/LAB-SERVE-001c/PRE_REGISTRATION.md`)
- Server both arms: `-fa on --ctx-size 73728 --parallel 4 --jinja --n-cpu-moe 8 -ctk q8_0 -ctv q8_0`
  → 4 slots × 18432 tok/slot (in-run verified; ~20.4 GB). ON adds `--spec-type draft-mtp --spec-draft-n-max 4`.
  This envelope was chosen because the 001b 2048/slot cannot fit the coding class (Stage 1 consequence);
  feasibility canary confirmed all candidate ctx sizes fit 24 GB.
- Open-loop: NO `--max-concurrency` (4 slots queue excess); Poisson `--request-rate`; `--seed` fixes
  arrivals+shuffle identically per arm (§14 common schedule). `--disable-ignore-eos` (caps bind, below).
- Workload pinned: `workload/workload_001c.jsonl` (sha in `workload_manifest.json`), 120 items, 2
  disjoint-band classes; per-request output-cap distribution (interactive {128,256,512,1024}, coding
  {512,1024,1536}). **Reasoning model ⇒ outputs pin to caps (cap-bounded, not natural-EOS)** — owner-
  approved documented deviation from §9; the cap distribution supplies §20 length strata.
- Capacity probe C≈0.085 req/s → load points LOW 0.030 / NEAR 0.072 / OVERLOAD 0.110.
- 2 paired blocks × 2 arms × 3 points = 12 server-starts; per-rep seeds 101/102 (arms share within rep,
  differ across reps); arm order alternated per rep. n=2 ⇒ direction+magnitude, no p<0.05. 4h guard.

## 4. Execution
All 12 cells rc 0; **0 request errors**; token ratio **1.000** every cell; elapsed **48.5 min**;
`stopped_on_budget=false`. Same 24 prompts sampled everywhere (out-len mean 650.7) ⇒ identical work,
only rate varies. OPEN_LOOP_INSTRUMENT=QUALIFIED. Manifest `campaign/raw/manifest.jsonl`; per-cell
`campaign/raw/*_rep*_*.normalized.json` (+ `.jsonl`, `.stdout.txt`, `.argv.txt`, `.props.json`,
`.slots.json`, `server_*.log`); analysis in `campaign/normalized/`.

## 5. Headline results (details + tables in the report)
- **Queueing onset:** completed tracks offered at LOW/NEAR; at OVERLOAD completed **0.090 < offered
  0.110** → sustainable capacity **~0.09 req/s**, onset between 0.072 and 0.110. E2E median rises
  monotonically; TTFT **p95** (not median) inflates with load = tail queue delay (TTFT median NOT
  relabeled as queue delay, §15). Demand>4 inferred at OVERLOAD from completed<offered + TTFT-p95≫prefill.
- **MTP practical winner:** E2E median AND p95 better at every load point (Δ −1.5…−3.1 s median,
  −1.9…−5.0 s p95, 2/2 direction); TPOT better at every point (−4.66 ms at OVERLOAD). Cost: small
  TTFT-median penalty (+24…+164 ms) and TTFT-p95 penalty only at OVERLOAD (+2.1 s). Capacity unchanged.
- **MoE TPOT paradox MASKED:** the 001b closed-loop N=4 MTP TPOT penalty (+8.5 ms) did NOT appear under
  open-loop — TPOT favored MTP. HYPOTHESIS (not proven): sustained exactly-4-way decode is rare under
  Poisson vs saturated closed-loop N=4. Expert routing NOT claimed as proven cause.
- **Fairness:** interactive TTFT ~2 s, coding TTFT ~8.4 s even at OVERLOAD — no starvation; both classes
  degrade proportionally; MTP helps both. (Thin: ~6 coding req/point.)
- **Energy OBSERVED_ONLY:** MTP ~3–5% lower J/out-token at every point (window includes warmup).

## 6. Status outcomes (§23)
TOPOLOGY_RECONCILIATION=RESOLVED · OPEN_LOOP_INSTRUMENT=QUALIFIED · LOW_LOAD_MTP=BETTER ·
NEAR_CAPACITY_MTP=BETTER · OVERLOAD_BEHAVIOR=CHARACTERIZED · MOE_TPOT_REAL_WORLD_IMPACT=MASKED ·
QUEUEING_ONSET=ESTIMATED (~0.09 req/s) · ENERGY=OBSERVED_ONLY.

## 7. Causal claims vs hypotheses
- CLAIMED (paired, direction-consistent): MTP improves E2E median+p95 and TPOT across LOW/NEAR/OVERLOAD
  for this workload; capacity ~0.09 req/s; onset between 0.072 and 0.110.
- HYPOTHESIS ONLY: why the 001b N=4 TPOT penalty vanishes open-loop (concurrency-distribution argument);
  "draft-verify competes with expert routing" remains an unproven mechanism for future isolation.

## 8. Failures / honesty
- Two throwaway helper scripts hit nested-quote bugs (my sanity waiters) — did not touch the campaign.
- Initial 16k-coding workload made capacity ~0.06 req/s and every output pinned to cap (reasoning model);
  trimmed to 8k coding + cap distribution per owner decision. Both capacity canaries retained under
  `campaign`/`capacity` dirs as evidence.
- No requests dropped from denominators (0 errors, 0 timeouts this run).

## 9. Reproduce / rollback
- Reproduce: §Reproduce in the report (workload gen → `ops/lab_serve_openloop_replicate.sh` → analyzer).
- Rollback: everything is untracked; `git clean -nd` to preview, or delete `runs/serving/LAB-SERVE-001c/`,
  the four new `ops/*` tools, and `runs/serving/LAB-SERVE-001b/TOPOLOGY_CLARIFICATION.md`.

## 10. One recommended next packet
**LAB-SERVE-001d** — closed-loop concurrency isolation (N∈{1,2,4,6,8}, fixed output lengths, MTP on/off)
to test the 001b-vs-001c TPOT-sign hypothesis directly, + a length-swept cell to settle §20. Keep it
closed-loop (isolation); open-loop realism stays with 001c. Do NOT modify the deploy profile.
