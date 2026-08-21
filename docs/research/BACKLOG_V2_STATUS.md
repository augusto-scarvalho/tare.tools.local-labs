# Backlog V2 — reconciliation against the current repo (2026-08-10)

Marks every V2 item ALREADY DONE / PARTIAL / MISSING / BLOCKED / OBSOLETE / PARKED against the
actual source, and states the concrete next step. **Wave A / P0 was implemented this session**
(see the `feat(bench-qa)` / `feat(promotion)` commits). Everything from Wave 1 on needs GPU or
long runs and is recorded as planned backlog, NOT executed (Backlog V2 §26.7).

Legend: ✅ done · ◑ partial · ○ missing · ⛔ blocked (on a prerequisite) · ⏸ parked · ⚑ this session.

---

## Wave 0 — qualify the lab (P0) — DONE THIS SESSION
| Item | State | Evidence / plan |
|---|---|---|
| **LAB-QA-001** Benchmark Harness Qualification | ✅ ⚑ | `tests/benchmark_harness/` — 16/16 green, no GPU. Real glue extracted to `benchmark_harness_qa.py` (imported by `a2_concision_bench.py`/`score_subset.py`). Both incident regressions (missing-prompt, stale-cache) have explicit tests. Rule enforced: no new benchmark becomes a gate without a self-test here. |
| **LAB-QA-002** Benchmark/Dataset Identity | ✅ ⚑ | `benchmark_harness_qa.run_identity()` + `{stem}__identity.json` sidecar per run; anchor `runs/quality-market/DATASET_IDENTITY.json` (humaneval-plus/gsm8k dataset hashes + harness commit). Full `model_sha256` deferred to LAB-PROV-001 (weights in the WSL VHDX). |
| **LAB-QA-003** Promotion semantics | ✅ ⚑ | `src/model_lifecycle/analysis/promotion.py` — lexicographic eligibility→correctness→quality→performance, reusing `analysis/gates.py` (which already did eligibility). PROMOTE/REJECT/HOLD, not a weighted score. Self-check covers crash/non-termination/low-quality → REJECT; clean+fast → PROMOTE. |

## Wave 1 — real endpoint (P0) — SERVING CHARACTERIZATION CAMPAIGN CLOSED (2026-08-10)
Serving campaign ran and is closed as of LAB-SERVE-001c. Final status table:

| Item | State | Evidence |
|---|---|---|
| **LAB-SERVE-001** Realistic serving benchmark | ✅ **PILOT_COMPLETE** | Bounded saturated pilot (dense-27B, MTP on/off, N∈{1,2,4,8}). Thin adapter `ops/lab_serve_bench.py` over `sglang.benchmark.serving` (backend `vllm-chat`). Evidence `runs/serving/LAB-SERVE-001/`; one interpretation superseded (see 001b). |
| **LAB-SERVE-001b** Variance/topology/MTP/MoE-transfer | ✅ **COMPLETE** | Paired calibration/replication, 5 server-level blocks, dense + MoE. `runs/serving/LAB-SERVE-001b/`. Server topology CLARIFIED by 001c (see below). |
| **LAB-SERVE-001c** Realistic open-loop characterization | ✅ **COMPLETE** | Minimal open-loop (LOW/NEAR/OVERLOAD, 2 paired blocks, MoE primary). `runs/serving/LAB-SERVE-001c/`. Finding: MTP improved E2E med+p95 and TPOT at all load points; sustainable capacity ~0.09 req/s; onset 0.072–0.110 req/s. Caveats in the report. |
| **LAB-SERVE-001d** Closed-loop concurrency × MTP TPOT isolation | ⏸ **PARKED / OPTIONAL** | 001c answered the practical serving question sufficiently for the lab's current purpose. 001d would isolate the mechanistic closed-loop concurrency×MTP-TPOT interaction, but is NOT required for endpoint qualification now. Preserved for future investigation. |
| **LAB-SERVE-002** Workload-specific profiles | ○ **NOT PROMOTED / future** | `serve_profiles.py` has `ServeSpec`+`SERVE_PROFILES` but not workload-typed. 001c produced *candidate* interactive/throughput characterizations but explicitly did NOT promote a profile (deploy defaults unchanged). Promotion belongs to a later explicit packet after serving + reliability evidence exist. |

## Wave 2 — reliability/soak (P0/P1) — CANCELLED
| LAB-REL-001 24h soak | ◼ **CANCELLED_BY_USER 2026-08-21** | Stopped by request after 369/369 successful partial operations, with zero operation or health failures. Receipts remain in `runs/reliability/LAB-REL-001-24h-2026-08-21/`. The incomplete run is preserved but has no pass/fail classification. |
| LAB-REL-002 48/72h soak | ○ ⛔ | only after 24h is clean. |

## Wave 3 — cache correctness (P0/P1) — PARTIAL
| LAB-CACHE-001 | ◑ ⚑ | Live Qwen3.8 slice passes 4/4: divergent-suffix reuse, partial removal, cancel-then-reuse, and 24,552-token long-context reuse; every cached completion was byte-identical to cold, matched a known-answer oracle, and had `cache_n>0`. Explicit slot file save/restore is implemented but blocked because `llm-inference.service` has `Restart=always` and stopping the unit requires unavailable sudo authentication. Speculative/MTP rollback remains untested on this non-speculative server. |

## Wave 4 — agentic capability (P0) — PARTIAL
| LAB-AGENT-001 Agentic suite V2 | ◑ ⚑ | BFCL-inspired local functional suite now passes 8/8 on Qwen3.8 IQ4_XS: selection, nested args, abstention, parallel, sequential, multi-turn, error recovery, and irreversible-no-blind-retry. Raw OpenAI responses retained. This closes the standard functional slice; stress/scale and perturbation robustness remain. |
| LAB-AGENT-002 FC robustness | ○ ⛔ | rephrase/reorder/rename/irrelevant-tool perturbations; blocked on AGENT-001. |

## Wave 5 — coding quality (P0) — TIER-1 MBPP+ COMPLETE
| LAB-CODE-001 Second coding axis | ◑ ⚑ | Official EvalPlus MBPP+ v0.2.0, full 378: base 326/378 = 86.24% (Wilson 82.40–89.35), Plus 284/378 = 75.13% (70.54–79.22); 378/378 fenced/answered. One task (`Mbpp/260`) truncated at both 768 and isolated 2048, confirming non-termination rather than a short budget. Harness QA is 23/23; dataset release hash and full artifact identity retained. BigCodeBench and higher tiers remain. |

## Wave 6 — long-context quality (P1) — LOCAL 32K SLICE PARTIAL
| LAB-CTX-001 Effective context curve | ◑ ⚑ | RULER-inspired (not RULER-comparable) paired live-model matrix at 8k/16k/28k: retrieval, multikey and multihop 18/18; aggregation paired matrix 5/6. Expanded aggregation n=10 per length: 10/10, 9/10, 10/10, exposing one reproducible but non-monotonic positional sensitivity rather than global collapse. Exact template/tokenizer calibration and raw timings retained. Official NVIDIA RULER, repo-context, and 64k/128k require another launch profile. |

## Wave 7 — energy/thermal (P1) — INSTRUMENT QUALIFIED
| LAB-ENERGY-001 Energy instrumentation | ✅ ⚑ | Streaming first-token/final-event boundaries + trapezoidal `nvidia-smi power.draw`, three alternating-order reps at ~2.7k/~13.2k prompt tokens. Median gross prefill 0.206/0.262 J per prompt token; decode 8.80/9.52 J/token at 42.1/39.5 t/s; peak 385.3 W/72 C. First biased-boundary attempt is preserved and superseded by interpolation self-test + rerun. |
| LAB-ENERGY-002 Power-limit curve | ○ | Sweep 100/90/80/70% (keep the undervolt as an explicit condition) → Pareto, not max throughput. GPU. |

## Wave 8 — serve×lab mode (P1) — MISSING
| LAB-OPS-001 Explicit operating modes | ○ | Simple SERVE/LAB state lock in `lmctl` (no scheduler). Cheap + CPU-only, but P1 → deferred to keep this session P0-only. |
| LAB-OPS-002 Interference matrix | ○ | Endpoint + controlled CPU/RAM/disk/GPU contenders → degradation matrix. GPU. |

## Wave 9 — provenance (P1) — PARTIAL
| LAB-PROV-001 Artifact identity | ◑ ⚑ | `run_identity` now carries on-demand streaming SHA-256, bytes, source repo/revision, quantizer, imatrix and fail-closed `VERIFIED_SOURCE/COMMUNITY_REQUANT/UNKNOWN`; QA covers hashing and invalid classes. Current IQ4_XS is pinned by full SHA-256 and classified COMMUNITY_REQUANT. Fleet-wide source revisions remain incomplete/UNKNOWN, so the wave stays partial. |
| LAB-PROV-002 Requant parity probe | ○ | quality/tool/long-ctx/termination smoke vs an official artifact for community requants. GPU. |

## Wave 10/11/12/13 — expansion (P2) — MISSING/BLOCKED
| LAB-ENGINE-001 llama.cpp×SGLang | ○ | Define comparability FIRST (same snapshot/quant class/workload/sampling); find regimes, not a universal winner. GPU. |
| LAB-ENGINE-002 vLLM | ⏸ | only if a truly comparable 3090-compatible config exists. |
| LAB-JUDGE-001 Human calibration | ○ | `a2_gate3_judge.py` panel exists; add 50–100 blind human-preference pairs → agreement/bias vs judges. |
| LAB-VLM-001 Visual coding suite | ◑ | VLM M0 + `vlm_*.py` exist; expand to stack-trace/UI-bug/screenshot-diff cases. GPU. |
| LAB-OPT-001 Multi-objective auto-tune | ○ ⛔ | Optuna/ASHA — blocked until QA + serve + resource foundations are trustworthy. |

## Close-outs
| LAB-CLOSE-001 no-mmap residual | ○ | Paired mmap ON/OFF at fixed placement + cooldown + guarded clocks → real/confounded/noise. GPU, cheap. |
| LAB-CLOSE-002 fable-fusion non-termination | ○ | Sweep max_tokens/EOS/stop/sampling; measure termination_rate (the `flag_truncated` primitive from LAB-QA-001 is the measurement). If still non-terminating → DISQUALIFIED for agentic role. GPU. |

## PARKED (§20) — aligned, do not open without a trigger
custom CUDA kernels w/o measured bottleneck · sub-4-bit KV · learned MoE placement (balanced) ·
EAGLE/DSpark for this fleet · distributed/disaggregated/multi-GPU · MCP wrapper for lmctl ·
tare.tools integration · full agent product · image generation · full RL/FT of 35B · sophisticated
scheduler · k8s/cluster.

## Not obsolete, superseded
Old backlog **B4** (hybrid cache probes) → promoted to **LAB-CACHE-001**. Old **B3** (2nd bench axis)
→ **LAB-CODE-001**. No V2 item is OBSOLETE.

---

## Recommended next executable step
The serving-characterization campaign (LAB-SERVE-001 → 001b → 001c) is **CLOSED**. Recommendation:
**close/compact the serving session and choose the next independent research campaign separately.** Do
NOT auto-execute LAB-SERVE-001d (parked), LAB-REL-001, or any longer soak. Candidate next campaigns
(pick deliberately, not by default): LAB-ENERGY-001 (qualified J/token), LAB-CODE-001 (2nd coding axis),
or LAB-SERVE-001d (mechanistic TPOT isolation) if a later question makes it useful.
