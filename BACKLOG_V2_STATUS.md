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

## Wave 1 — real endpoint (P0) — MISSING (needs GPU + serving)
| Item | State | Plan |
|---|---|---|
| **LAB-SERVE-001** Realistic serving benchmark | ○ | **Adapt** SGLang `python -m sglang.bench_serving` against our `llama-server` OpenAI endpoint (it already does Poisson arrivals, request-rate, max-concurrency, TTFT/ITL/TPOT/p99, JSONL). Prefer a thin wrapper over reimplementation. Workloads A–E, N∈{1,2,4,8}. Re-measure the MTP-flips-at-N≈4 hypothesis empirically (don't hardcode). Add its harness-QA cases per LAB-QA-001 first. |
| **LAB-SERVE-002** Workload-specific profiles | ◑ ⛔ | `serve_profiles.py` has `ServeSpec` + `SERVE_PROFILES` but not workload-typed. Add `interactive`/`throughput`/`long-context` profiles **once SERVE-001 data exists**. Extend `serve_profiles.py`/`lmctl`, don't add a new manager. |

## Wave 2 — reliability/soak (P0/P1) — MISSING (long runs)
| LAB-REL-001 24h soak | ○ | Mixed workload + telemetry (VRAM/RAM/power/temp/latency/errors over time); first run = baseline, then set envelope. Reuse `collectors/host.py` + `collectors/desktop_load.py`. |
| LAB-REL-002 48/72h soak | ○ ⛔ | only after 24h is clean. |

## Wave 3 — cache correctness (P0/P1) — MISSING (old B4 promoted)
| LAB-CACHE-001 | ○ | Correctness (not speed) tasks with known outputs for prompt-cache reuse / slot save+restore / context shift / partial removal / speculative+MTP rollback / cancel-then-reuse / long-ctx reuse. "cached result must match cold result". Add QA cases first. Needs serving (GPU). |

## Wave 4 — agentic capability (P0) — PARTIAL
| LAB-AGENT-001 Agentic suite V2 | ◑ | `agentic_gate.py` + `agent_bench.py` exist as a smoke. Expand with BFCL-inspired cases (selection/args/abstention/parallel/sequential/multi-turn/error-recovery/irreversible-no-blind-retry) + rich metrics + smoke/standard/stress subsets. Case definitions are CPU-authorable; running needs serving. |
| LAB-AGENT-002 FC robustness | ○ ⛔ | rephrase/reorder/rename/irrelevant-tool perturbations; blocked on AGENT-001. |

## Wave 5 — coding quality (P0) — MISSING
| LAB-CODE-001 Second coding axis | ○ | Ladder: keep HumanEval+ (smoke); add MBPP+/BigCodeBench (Tier1) → SWE-Explore (Tier2) → SWE-bench Verified 10–20 tasks (Tier3, Docker for eval isolation only) → FeatureBench/SWE-EVO (Tier4). **Each new benchmark gets a LAB-QA-001 self-test before it gates anything.** Record model+harness config explicitly. GPU + Docker-for-eval. |

## Wave 6 — long-context quality (P1) — MISSING
| LAB-CTX-001 Effective context curve | ○ | Integrate NVIDIA RULER (retrieval/multi-hop/aggregation/multi-key) at 8k/16k/32k/64k/128k, via Lighteval (WSL) if it removes plumbing. Plus a small repo-context task. Output a quality×TTFT×decode×VRAM Pareto curve, not "128k runs". GPU. |

## Wave 7 — energy/thermal (P1) — MISSING (telemetry base exists)
| LAB-ENERGY-001 Energy instrumentation | ○ | Reuse TokenPowerBench concepts; phase-aligned (prefill/decode) J/token via nvidia-smi telemetry. `collectors/host.py` already samples GPU — extend to power. |
| LAB-ENERGY-002 Power-limit curve | ○ | Sweep 100/90/80/70% (keep the undervolt as an explicit condition) → Pareto, not max throughput. GPU. |

## Wave 8 — serve×lab mode (P1) — MISSING
| LAB-OPS-001 Explicit operating modes | ○ | Simple SERVE/LAB state lock in `lmctl` (no scheduler). Cheap + CPU-only, but P1 → deferred to keep this session P0-only. |
| LAB-OPS-002 Interference matrix | ○ | Endpoint + controlled CPU/RAM/disk/GPU contenders → degradation matrix. GPU. |

## Wave 9 — provenance (P1) — PARTIAL
| LAB-PROV-001 Artifact identity | ◑ | `models.py` has path/quant/geometry + dated discard-comments; `run_identity` stubs `model_sha256=null`. Add provenance fields (source repo/rev, quantizer, imatrix?, GGUF hash on-demand, VERIFIED_SOURCE/COMMUNITY_REQUANT/UNKNOWN). Hashing is IO-heavy (no GPU) — on demand. |
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
When a GPU session is available: **LAB-SERVE-001** (adapt SGLang `bench_serving` behind a thin
wrapper) — it is the P0 gateway to Waves 1–2 and gives the real latency/throughput/concurrency
curves the rest of the endpoint work depends on. Its harness-QA cases go into
`tests/benchmark_harness/` first, per the standing rule.
