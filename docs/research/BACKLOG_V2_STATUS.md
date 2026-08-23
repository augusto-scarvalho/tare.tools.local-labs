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
| **LAB-QA-001** Benchmark Harness Qualification | ✅ ⚑ | `tests/benchmark_harness/` — 23/23 green, no GPU. Real glue extracted to `benchmark_harness_qa.py` (imported by `a2_concision_bench.py`/`score_subset.py`). Incident regressions and newer strict-scoring/provenance cases have explicit tests. Rule enforced: no new benchmark becomes a gate without a self-test here. |
| **LAB-QA-002** Benchmark/Dataset Identity | ✅ ⚑ | `benchmark_harness_qa.run_identity()` + `{stem}__identity.json` sidecar per run; anchor `runs/quality-market/DATASET_IDENTITY.json` (humaneval-plus/gsm8k dataset hashes + harness commit). Full `model_sha256` deferred to LAB-PROV-001 (weights in the WSL VHDX). |
| **LAB-QA-003** Promotion semantics | ✅ ⚑ | `src/model_lifecycle/analysis/promotion.py` — lexicographic eligibility→correctness→quality→performance, reusing `analysis/gates.py` (which already did eligibility). PROMOTE/REJECT/HOLD, not a weighted score. Self-check covers crash/non-termination/low-quality → REJECT; clean+fast → PROMOTE. |

## Priority insertion — open-weight breadth for the RTX 3090 (P0) — EXECUTED 2026-08-21

| Item | State | Evidence / plan |
|---|---|---|
| **LAB-MUSE-000 Runtime/artifact admission** | ✅ | Official artifacts matched expected bytes and were SHA-bound; isolated upstream `b10573` loaded Muse, parsed reasoning/content and tools, and the baseline service was restored exactly. |
| **LAB-MUSE-001 3090 residency + DFlash** | ✅ **COMPACT CLOSED** | Text, vision, and DFlash separately preserved the 4 GB VRAM reserve; the combined stack left only 2,475 MiB. DFlash reached +51.4% at `n=8` but changed greedy output, so `DRAFT_REJECTED`. |
| **LAB-MUSE-002 Role quality** | ✅ **EARLY STOP / HOLD** | Agent 7/8 versus Qwen3.8 8/8; historical GSM failures 4/5; `Mbpp/260` never reached a final answer at 2k or 4k. No role promotion; wider gates not opened. |
| **LAB-MUSE-003 Context/reasoning** | ✅ **COMPACT CLOSED** | Strict 16/16 across 8k/32k/64k/120k with one replicate per cell; all reasoning strengths correct on the control, but even `low` was verbose. Persistent state remains unqualified. |
| **LAB-MUSE-004 Multimodal safety** | ✅ **VQA SPECIALIST / SAFETY PASS BOUNDED / HOLD** | Explicitly reopened descriptive expansion: Muse scored 107/150 (71.33%, zero unparsed) on the exact local MMStar panel, ahead of all retained comparators, and passed 5/5 visual-injection/tool/session-isolation cases. Earlier agent/code/cache/DFlash/full-stack failures keep the overall model on HOLD. See `runs/requalification/MUSE-GLIMMER-FULL-2026-08-22/RESULT.md`. |

Authoritative preregistration: `docs/research/MUSE_GLIMMER_3090_EXPERIMENT_PACKET_2026-08-21.md`. Its compact
decision is `HOLD`. Evidence: `runs/requalification/MUSE-GLIMMER-2026-08-21/RESULT.md`.

## Priority continuation — Qwen3.8 external artifact breadth — EXECUTED 2026-08-21/22

| Item | State | Evidence / plan |
|---|---|---|
| **LAB-QWEN38-UNSLOTH-REV Current IQ4_XS** | ✅ **REJECT SUPERSESSION** | Revision-pinned current artifact saved 9.25% bytes, passed agent 8/8, cache 4/4, GSM 94/100, and replicated context 36/36, but MBPP+ regressed from historical 326/284 to 323/280, missing both frozen non-inferiority margins by one task. |
| **LAB-QWEN38-UNSLOTH-REV Current Q2_K_XL** | ✅ **COMPACT REJECT** | Saved 7.94% bytes and 832 MiB residency, but added an 8k aggregation miss and truncated `Mbpp/260` at 2,048 where the historical Q2 terminated correctly. |
| **LAB-COLD-FUSION-001 Base role** | ✅ **COMPACT REJECT** | Exact `NEO-MTP-IQ4_XS` fit with 5,197 MiB free and passed agent/cache, but context was 9/12, `Mbpp/260` failed Base/Plus, and GSM failure replay was 1/5. Broad gates stopped. |
| **LAB-COLD-FUSION-002 Embedded MTP** | ✅ **MTP REJECTED / DESCRIPTIVE** | Explicitly reopened without waiving the base-role failure. Nine counterbalanced cells ran successfully: n2/n3 accelerated longer code/prose 44–72% but slowed the tiny answer 15–29%, changed code/prose bytes versus off, and every prose arm hit the 384-token cap. See `runs/requalification/COLD-FUSION-MTP-2026-08-22/RESULT.md`. |

Evidence: `runs/requalification/QWEN38-UNSLOTH-REVISION-2026-08-21/RESULT.md` and
`runs/requalification/COLD-FUSION-2026-08-22/RESULT.md`. The historical deployment remains unchanged.

## User-unparked continuation — image, harness, and A2 methods (2026-08-22/23)

| Item | State | Evidence / plan |
|---|---|---|
| **LAB-IMG-001 Qwen-Image** | ✅ **FIT/MECHANISM PASS · QUALITY HOLD** | Exact official revision, NF4 transformer/text encoder with BF16 compute and CPU offload. The 768px/30-step panel fit with about 13.1 GiB peak inference use and byte-identical replay; frozen semantic gate 10/13. See `runs/image/LAB-IMG-001-QWEN-IMAGE-2026-08-22/RESULT.md`. |
| **LAB-IMG-002 SDXL baseline** | ✅ **FIT/MECHANISM PASS · QUALITY REJECT** | Exact official FP16 baseline on the matched 768px/30-step panel. About 9.5x faster than Qwen-Image but only 3/13 semantic clauses. See `runs/image/LAB-IMG-002-SDXL-2026-08-22/RESULT.md`. |
| **LAB-HARNESS-001** | ✅ **BOUNDED PASS** | Digest-bound TaskContract delta, structural RepositoryEvidencePack, and baseline non-weakening gate. Required-file recall 5/5 and model-native context reduction 83.85%. See `runs/agent-product/LAB-HARNESS-001-2026-08-22/RESULT.md`. |
| **LAB-HARNESS-002/003** | ✅ **BOUNDED PASS** | Independent Gemma test writer killed 6/7 mutations; deterministic anti-slop gate plus independent critic classified 8/8 frozen patches with zero unsafe accepts. See `runs/agent-product/LAB-HARNESS-002-INDEPENDENT-TESTS-2026-08-22/RESULT.md` and `runs/agent-product/LAB-HARNESS-003-CRITIC-2026-08-22/RESULT.md`. |
| **A2 Stage-2 E1/E2** | ✅ **G0 KILL / NO WEIGHT EDIT** | Corrected non-thinking answer-channel extraction and complete layers 8–51 sweep produced 0/44 eligible directions. All induction deltas were zero and best harmless KL was 0.566 vs the <0.1 gate. Downstream ablation/merge arms remain dependency-blocked. See `runs/a2/stage2-2026-08-22/RESULT.md`. |
| **B5 task-oriented Q3/mixed quant** | ✅ **SUPERSEDED BY QUANT FRONTIER** | Seven-quant Qwen3.8 ladder already gated code, hard math and long context through Q2/IQ2. Q2_K_XL is the measured floor; IQ2_M exposes the long-context cliff. No unresolved layer-map decision remains. See `ops/qwen38-bringup/QUANT_FRONTIER_CAMPAIGN.md`. |

## Wave 1 — real endpoint (P0) — SERVING CHARACTERIZATION CAMPAIGN CLOSED (2026-08-10)
Serving campaign ran and is closed as of LAB-SERVE-001c. Final status table:

| Item | State | Evidence |
|---|---|---|
| **LAB-SERVE-001** Realistic serving benchmark | ✅ **PILOT_COMPLETE** | Bounded saturated pilot (dense-27B, MTP on/off, N∈{1,2,4,8}). Thin adapter `ops/lab_serve_bench.py` over `sglang.benchmark.serving` (backend `vllm-chat`). Evidence `runs/serving/LAB-SERVE-001/`; one interpretation superseded (see 001b). |
| **LAB-SERVE-001b** Variance/topology/MTP/MoE-transfer | ✅ **COMPLETE** | Paired calibration/replication, 5 server-level blocks, dense + MoE. `runs/serving/LAB-SERVE-001b/`. Server topology CLARIFIED by 001c (see below). |
| **LAB-SERVE-001c** Realistic open-loop characterization | ✅ **COMPLETE** | Minimal open-loop (LOW/NEAR/OVERLOAD, 2 paired blocks, MoE primary). `runs/serving/LAB-SERVE-001c/`. Finding: MTP improved E2E med+p95 and TPOT at all load points; sustainable capacity ~0.09 req/s; onset 0.072–0.110 req/s. Caveats in the report. |
| **LAB-SERVE-001d** Closed-loop concurrency × MTP TPOT isolation | ⛔ **BLOCKED_RUNTIME_CRASH** | Reopened by explicit backlog authorization. The first MTP-off N=4 cell crashed with CUDA illegal memory access after completing the workload; a symmetric no-CUDA-graphs recovery reproduced the crash. No TPOT conclusion; prior 001b/001c evidence remains authoritative. See `runs/serving/LAB-SERVE-001d-2026-08-22/RESULT.md`. |
| **LAB-SERVE-002** Workload-specific profiles | ✅ **HOLD_MODEL_DRIFT / PACKET FROZEN** | The promotion audit preserves Qwen3.6-MoE LOW/NEAR/OVERLOAD guidance as descriptive only. The live service is Qwen3.8 with a different topology and memory envelope, and the 24 h run was cancelled, so no profile/default was promoted. Frozen same-artifact gates are in `runs/serving/LAB-SERVE-002-2026-08-22/DECISION_PACKET.md`; decision in `RESULT.md`. |

## Wave 2 — reliability/soak (P0/P1) — EXCLUDED / CANCELLED
| LAB-REL-001 24h soak | ◼ **CANCELLED_BY_USER 2026-08-23** | A mistaken fresh launch was stopped immediately after the user clarified that soaks were excluded. It ended at 8/8 operations and zero health failures, remains incomplete, and is not a PASS. Receipts: `runs/reliability/LAB-REL-001-24h-2026-08-23/`. |
| LAB-REL-002 48/72h soak | ◼ **CANCELLED / NOT RUN** | The queued sequencer was stopped before either successor began. Do not execute soak experiments under the current direction. |

## Wave 3 — cache correctness (P0/P1) — NO-SPEC CLOSED / MTP BLOCKED
| LAB-CACHE-001 | ◑ **NO-SPEC PASS / MTP BLOCKED** | Explicit no-spec slot save/erase/restore passed with exact oracle-correct cached continuation. MTP n3 cache/cancel/reuse passed 4/5 full runs (19/20 cases) and slot persistence passed 1/2, but the original long-context and restored-state completions returned only `!` with zero accepted drafts. Four fresh cache reruns and one slot rerun passed; chat localization was 16/16 per arm and byte-identical through 32k, while raw-depth localization was 10/10 per arm. The intermittent correctness failure is preserved and blocks MTP cache/checkpoint promotion. See `runs/cache/LAB-CACHE-001-MTP-2026-08-22/RESULT.md`. |

## Wave 4 — agentic capability (P0) — PARTIAL
| LAB-AGENT-001 Agentic suite V2 | ◑ ⚑ | BFCL-inspired local functional suite now passes 8/8 on Qwen3.8 IQ4_XS: selection, nested args, abstention, parallel, sequential, multi-turn, error recovery, and irreversible-no-blind-retry. Raw OpenAI responses retained. This closes the standard functional slice; stress/scale and perturbation robustness remain. |
| LAB-AGENT-002 FC robustness | ✅ **FAIL / POSITIONAL ORDER** | Primary matrix 39/40: control, rephrase, rename, and irrelevant-tool arms passed 8/8; reorder failed irreversible recovery. Seed-fixed localization reproduced canonical order 5/5 versus reordered 0/5; reversing tool list alone failed 0/3 while schema-order-only passed 3/3. No blind transfer retry occurred, but the model asked permission instead of dispatching status inspection. See `runs/agent/LAB-AGENT-002-2026-08-22/RESULT.md`. |
| LAB-AGENT-003 Stress/scale | ✅ **PASS / BOUNDED** | Corrected fixed-seed matrix passed 16/16: 32 distractor tools, exact parallel fan-out 12, completed sequential depth 8, and 16 irrelevant history turns. The initial 15/16 receipt is retained but marked INVALID/SUPERSEDED because depth-0 required a `root` token absent from its prompt; full corrected rerun passed. See `runs/agent/LAB-AGENT-003-2026-08-22/RESULT.md`. |
| LAB-AGENT-004 Irreversible recovery policy | ✅ **PROMOTE POLICY / BOUNDED** | A single explicit policy passed the reversed-order target 5/5 and both full canonical/reversed matrices 16/16, with zero blind retries. See `runs/agent/LAB-AGENT-004-IRREVERSIBLE-POLICY-2026-08-22/RESULT.md`. |

## Wave 5 — coding quality (P0) — TIER-1 MBPP+ COMPLETE
| LAB-CODE-001 Second coding axis | ◑ ⚑ | Official EvalPlus MBPP+ v0.2.0, full 378: base 326/378 = 86.24% (Wilson 82.40–89.35), Plus 284/378 = 75.13% (70.54–79.22); 378/378 fenced/answered. One task (`Mbpp/260`) truncated at both 768 and isolated 2048, confirming non-termination rather than a short budget. Harness QA is 23/23; dataset release hash and full artifact identity retained. Higher tiers remain. |
| LAB-CODE-002 BigCodeBench-Hard Tier-1 | ✅ **COMPLETE / BASELINE** | Official Instruct Hard v0.1.4, greedy n=1: 48/148 pass@1 = 32.43% (Wilson 25.42–40.34); adjusted 48/147 = 32.65% after the sole ground-truth failure `/590` reproduced HTTP 403. Generation was 148/148 nonempty, syntax-valid, compilable, and `finish=stop`. `/1042` OOM reproduced in isolation and remains a model fail. See `runs/code/LAB-CODE-002-BCB-HARD-2026-08-22/RESULT.md`. |
| LAB-CODE-003 SWE-bench Verified Tier-2 | ✅ **COMPLETE / 5 OF 10 RESOLVED** | The official gold gate passed, then a revision-pinned mini-SWE-agent 2.4.6 pilot ran the frozen 10-instance spread on the incumbent Qwen3.8. Five patches were submitted and all five resolved; five cases exhausted the frozen 40-call budget with empty patches. Bounded score 5/10 = 50%, with zero infrastructure/evaluator errors and no leaderboard submission. See `runs/code/LAB-CODE-003-SWEBENCH-VERIFIED-2026-08-22/RESULT.md`. |
| LAB-CODE-004/005/005B Loop-efficiency remediation | ✅ **CLOSED NEGATIVE** | Prompt-only did not stop the deterministic loop. The corrected middleware blocked 29 duplicate executions in the gate failure, but the model still exhausted 40 calls with no patch. The prior 5/10 score remains unchanged; no selective rescore. |

## Wave 6 — long-context quality (P1) — COMPLETE / BOUNDED
| LAB-CTX-001 Effective context curve | ◑ ⚑ | RULER-inspired (not RULER-comparable) paired live-model matrix at 8k/16k/28k: retrieval, multikey and multihop 18/18; aggregation paired matrix 5/6. Expanded aggregation n=10 per length: 10/10, 9/10, 10/10, exposing one reproducible but non-monotonic positional sensitivity rather than global collapse. Exact template/tokenizer calibration and raw timings retained. Superseded as the sole broad-context evidence by LAB-CTX-002, but preserved as the denser local positional slice. |
| LAB-CTX-002 Official RULERv1 64k/128k | ✅ **64K PILOT FAIL / 128K BOUNDED PASS** | Official 13-task pilot scored 82.82% at 64k versus 100% at 128k. Gate-triggered n=3 replication of VT/CWE/FWE yielded 66.7/40.0/88.9% at 64k and 100/100/100% at 128k; the bounded mixed-n panel is 91.97%/100%. Four 64k outputs truncated versus 0 at 128k. Length-conditioned instances prevent a causal “128k is better” claim. See `runs/context/LAB-CTX-002-RULER-V1-2026-08-22/RESULT.md`. Repo-context remains next. |
| LAB-CTX-003 LongBench RepoBench-P | ✅ **COMPLETE / QUALITY FAIL** | Full 500 official-data/model-native-template run scored 39.56 versus the 55.0 gate; exact first line 109/500. Similarity fell 45.21 (<4k) → 36.78 (4–8k) → 14.10 (8k+), despite no input truncation. Python 49.83 versus Java 30.38. The stock raw-prompt pilot scored 1.90 and was stopped for instruct prompt-mode mismatch. See `runs/context/LAB-CTX-003-REPOBENCH-P-2026-08-22/RESULT.md`. |

## Wave 7 — energy/thermal (P1) — COMPLETE / QUALIFIED
| LAB-ENERGY-001 Energy instrumentation | ✅ ⚑ | Streaming first-token/final-event boundaries + trapezoidal `nvidia-smi power.draw`, three alternating-order reps at ~2.7k/~13.2k prompt tokens. Median gross prefill 0.206/0.262 J per prompt token; decode 8.80/9.52 J/token at 42.1/39.5 t/s; peak 385.3 W/72 C. First biased-boundary attempt is preserved and superseded by interpolation self-test + rerun. |
| LAB-ENERGY-002 Power-limit curve | ✅ **COMPLETE / RETAIN 420 W** | Qualified 24-cell counterbalanced sweep at 420/378/336/294 W, stock voltage curve with no undervolt. At long context, 378 W retained 99.31% prefill and 95.66% decode throughput but used 1.69%/0.86% more gross energy per token; 336/294 W saved 3–7% energy but lost 7–18% throughput. No reduced limit met the frozen 95% rule, so defaults remain 420 W. See `runs/energy/LAB-ENERGY-002-POWER-CURVE-2026-08-22/RESULT.md`. |

## Wave 8 — serve×lab mode (P1) — COMPLETE / QUALIFIED
| LAB-OPS-001 Explicit operating modes | ✅ **COMPLETE / QUALIFIED** | `lmctl mode show/check/set` now enforces a persistent fail-closed SERVE/LAB state with audit reason, CAS, atomic write and exclusive writer lock. Live tests refused judge launch in SERVE, LAB transition over active 8080, and 8080 launch in LAB; 8081 is an explicit auxiliary. Unit suite 10/10 and live SERVE→LAB→SERVE passed. See `runs/ops/LAB-OPS-001-MODE-LOCK-2026-08-22/RESULT.md`. |
| LAB-OPS-002 Interference matrix | ✅ **COMPLETE / GPU MATERIAL** | Qualified 15-cell baseline/CPU/RAM/disk/GPU matrix. CPU/RAM/disk short-workload prefill degradation was 4.3–5.1%, below the 10% threshold. GPU matmul reduced decode 7.14% but, critically, raised gross prefill energy/token 54.0% and decode energy 8.30%, making same-GPU colocation material. See `runs/ops/LAB-OPS-002-INTERFERENCE-2026-08-22/RESULT.md`. |
| LAB-OPS-003 Canonical context/VRAM envelope | ✅ **COMPLETE / 81,920 MAX PASSING LADDER POINT** | Live-equivalent startup curve with the embedding resident left 4,599/4,151/3,927/3,703/2,807 MiB free at 65,536/81,920/90,112/98,304/131,072 context. The frozen 4 GiB reserve therefore passes through 81,920 and fails from 90,112 upward. Allocation evidence only; live 131k profile was restored unchanged. See `runs/ops/LAB-OPS-003-CONTEXT-VRAM-2026-08-22/RESULT.md`. |
| Canonical context policy | ✅ **RETAIN 131,072 / 81,920 RESERVE PROFILE** | RULER's bounded 128k evidence and the SERVE/LAB no-colocation boundary outweigh the extra idle reserve for the exclusive canonical endpoint. Keep 131,072 as default; use 81,920 only when a task explicitly requires the historical 4 GiB reserve. No service mutation was needed. See `runs/ops/CANONICAL-CONTEXT-POLICY-2026-08-22/DECISION.md`. |

## Wave 9 — provenance (P1) — AUTHORIAL CLOSED / THIRD-PARTY PARTIAL
| LAB-PROV-001 Artifact identity | ◑ **AUTHORIAL LINEAGE CLOSED / ONE MISMATCH** | All 33 resident GGUFs are inventoried: 31 fully pinned, 1 content-pinned local derivation, and 1 local ThinkingCap MTP digest mismatch against its claimed revision. The authorial merge has exact receipts for all 31 parent shards plus quantizer evidence. Third-party builds remain undisclosed. See `runs/provenance/LAB-PROV-001-FLEET-2026-08-22/RESULT.md`. |
| LAB-PROV-002 Requant parity probe | ✅ **PROVENANCE CLOSED / PARITY REJECTED** | Official Qwen3.8-27B revision `1d4bf0f2...` was verified 18/18, converted to BF16, and requantized with pinned llama.cpp `87a416bd...` plus the exact Unsloth imatrix. The authorial IQ4_XS passed agent 8/8, cache 4/4 and context 12/12, but was 5.82% larger, used 680 MiB more VRAM, truncated `Mbpp/260`, and changed deterministic outputs. Retain Unsloth UD-IQ4_XS. See `runs/provenance/LAB-PROV-002-REQUANT-2026-08-22/RESULT.md`. |

## Wave 10/11/12/13 — expansion (P2) — MISSING/BLOCKED
| RNN-10 official recurrent checkpoint | COMPLETE — RWKV7 MECHANISM QUALIFIED / DEPLOY BLOCKED | RetNet official-checkpoint stage is upstream-blocked. Official RWKV7 1.5B passed fit, constant-state, cached-continuation and isolation gates on the 3090; first-use runtime is immature and the publisher does not assert a weight license. See `runs/requalification/RWKV7-1.5B-20260805-2026-08-22/RESULT.md`. |
| RNN-11 Falcon-H1R-7B hybrid | COMPLETE — **HOLD ROLE** | Official Q8 fit with 14,275 MiB free and passed smoke 4/4, tools 8/8 and GSM replay 4/5. `Mbpp/260` returned empty content at both 2,048 and diagnostic 4,096 tokens, so context expansion was not opened. See `runs/requalification/FALCON-H1R-7B-2026-08-22/RESULT.md`. |
| Resident worker breadth | COMPLETE — **5 OF 5 HOLD** | Mistral Small 24B Heretic failed the 4 GiB reserve at its 32k cache-compatible profile; Gemma 4 26B Heretic failed cache 1/4; GPT-OSS 20B remained 6/8 after targeted policy repair; official Gemma 4 26B failed cache 0/4; newly admitted Ornith 1.5 passed fit and agents 8/8 but cache 3/4. All five artifacts were full-hashed and later gates stopped. |
| LAB-ENGINE-001 llama.cpp×SGLang | ✅ **COMPARABLE COMPLETE / REGIME-SPECIFIC** | Same official Qwen3-4B revision in BF16, nonce-controlled fresh prefill and identical client: SGLang led prefill 17.67%, decode was unresolved, while llama.cpp used roughly half the peak VRAM and started 5.3x faster. See `runs/engines/LAB-ENGINE-001-002-2026-08-22/RESULT.md`. |
| LAB-ENGINE-002 vLLM | ✅ **COMPLETE / REGIME-SPECIFIC** | Isolated vLLM 0.27.1 arm on the same BF16 snapshot passed after the documented WSL UVA switch. Prefill tied SGLang and led llama.cpp 18.11%; decode remained unresolved; startup was 101 s and peak VRAM 17,692 MiB. |
| LAB-JUDGE-001 Human calibration | ○ | `a2_gate3_judge.py` panel exists; add 50–100 blind human-preference pairs → agreement/bias vs judges. |
| LAB-VLM-001 Visual coding suite | ✅ **COMPLETE / 4 OF 4** | Added deterministic stack-trace, UI-overflow, screenshot-diff and terminal-failure fixtures. Resident Gemma-4-12B Vision passed all 20 frozen clauses in 4/4 non-empty completions. Synthetic accept boundary only. See `runs/vlm/LAB-VLM-001-2026-08-22/RESULT.md`. |
| LAB-OPT-001 Multi-objective auto-tune | ✅ **QUALIFIED SCREEN / NO DEPLOY DECISION** | Optuna 4.9.0 bounded six-cell `MTP depth × ubatch` screen passed all hard gates and selected `n4/ub1024` versus explicit `n3/ub2048` (+7.08% decode, +1.20% long prefill). Post-run reconciliation found the live ubatch default is 512, invalidating that arm as the deploy control. LAB-OPT-001b then aborted correctly because the exact 131k control left 2,782 MiB free, below the 4 GiB gate. No default changed. See `runs/optimization/LAB-OPT-001-2026-08-22/RESULT.md` and `runs/optimization/LAB-OPT-001b-2026-08-22/RESULT.md`. |

## Close-outs
| LAB-CLOSE-001 no-mmap residual | ✅ **COMPLETE / RESIDUAL CONFOUNDED** | Qualified 6-pair alternating Qwen3.6 MoE/ncmoe=6 replication. Warm decode delta was +0.18% for no-mmap with CI including zero; the historical −10.4% penalty did not reproduce. Fresh-process elapsed time favored no-mmap by median 10.87% (bootstrap 95% CI 3.98–28.29) and avoided cold mmap page-in, so recommend no-mmap for that exact MoE profile only. Current dense Qwen3.8 defaults remain unchanged. See `runs/close-outs/LAB-CLOSE-001-MMAP-2026-08-22/RESULT.md`. |
| LAB-CLOSE-002 fable-fusion non-termination | ✅ **COMPLETE / THINKING AGENTIC DISQUALIFIED** | Qualified 32-cell budget/EOS/stop/sampling matrix. Instruct naturally terminated 8/8; greedy+sampled thinking only 6/16 (37.5%), with two prompts hitting both 512 and 2,048. Explicit `</think>` stop produced 0/4 final answers despite two `stop` reasons; ignore-EOS instruct hit 512 in 4/4. Safe only as bounded instruct on this panel and still not quality-promoted. See `runs/close-outs/LAB-CLOSE-002-FABLE-TERMINATION-2026-08-22/RESULT.md`. |

## PARKED (§20) — aligned, do not open without a trigger
custom CUDA kernels w/o measured bottleneck · sub-4-bit KV · learned MoE placement (balanced) ·
EAGLE/DSpark for this fleet · distributed/disaggregated/multi-GPU · MCP wrapper for lmctl ·
tare.tools integration · full RL/FT of 35B · sophisticated
scheduler · k8s/cluster.

## Not obsolete, superseded
Old backlog **B4** (hybrid cache probes) → promoted to **LAB-CACHE-001**. Old **B3** (2nd bench axis)
→ **LAB-CODE-001**. No V2 item is OBSOLETE.

---

## Recommended next executable step
The image, harness-product primitives and optional A2 Stage-2 methods leg are now measured. A2 stopped
at its dependency gate and B5 is superseded by the existing seven-quant frontier. Soak experiments are
explicitly excluded. Human calibration and upstream/license/auth/hardware-dependent items remain blocked;
product/cloud/cluster builds are not local experiments. Reconciled queue:
`docs/research/REMAINING_EXPERIMENTS_2026-08-22.md`.
