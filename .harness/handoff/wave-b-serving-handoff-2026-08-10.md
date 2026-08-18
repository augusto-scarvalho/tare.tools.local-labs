# Wave B — Wave-A closure + LAB-SERVE-001 serving pilot — Handoff (2026-08-10)

Packet-specific audit record. Read-only w.r.t. the lab except the committed code/evidence below.
Mobile-delivered: this file + the report + matrix CSV/JSON + an evidence ZIP are attached to the chat.

## 1. Git state (precise, WA-CLOSE-001 terminology)
- Branch **master**. **Starting HEAD `a47a9c2`** → **Ending HEAD `2c81cb7`**.
- `trackedTreeClean = true` · `stagedTreeClean = true` · `untrackedArtifactsPresent = true`
  (`.harness/handoff/*.md` — intentional handoffs) · `overallWorktreeStatus = clean tracked tree
  with intentional untracked handoff artifacts` (NOT "clean").
- Session commits (all LOCAL, not pushed):
```
2c81cb7 bench(serving): LAB-SERVE-001 pilot evidence + report (dense-27B, MTP on/off × N)
894b341 feat(serving): LAB-SERVE-001 thin adapter + normalizer (vllm-chat)
35a02ef feat(bench-qa): Wave-A closure — evalplus sentinel + identity guard + 15% provenance
```

## 2. CURRENT reconstruction (measured this session)
- Engine: lifecycle fork `llama.cpp-master` @ **`068764d92`** (server fingerprint `b10159-068764d92`).
- SGLang **0.5.16**; the load-gen module is **`sglang.benchmark.serving`** (2703 lines) — and
  `sglang.bench_serving` is a **22-line DEPRECATED shim** re-exporting it. **Discrepancy vs the
  Wave-B packet**, which named `python -m sglang.bench_serving`. Recorded, path corrected.
- EvalPlus **0.3.1** (in `/home/augus/evalplus-venv`). transformers 5.12.1 (sglang venv).
- Model artifact benchmarked: `fable-tc-l1.0-q4` (`/home/augus/models/merges/fable-tc-l1.0-Q4_K_M.gguf`),
  tokenizer `/home/augus/models/fp16/base` (EXACT Qwen3.6-27B base). Deploy profile `deploy-moe`
  (35B MoE) unchanged and NOT touched (this packet only characterizes).

## 3. Wave-A four-gap disposition
| Gap | Disposition | Evidence |
|---|---|---|
| WA-CLOSE-001 git wording | CLOSED (terminology; no code conflates git states — grep of reporting code empty) | §1 above uses the four-state vocabulary |
| WA-CLOSE-002 real EvalPlus sentinel | CLOSED — `tests/benchmark_harness/evalplus_sentinel.py` drives evalplus 0.3.1 on HumanEval/0 (canonical→PASS, wrong→FAIL, real stale boundary). **QUALIFIED** (all 4 checks). Separate from & not replacing the 16 unit cases. | `evalplus_sentinel_report.json` |
| WA-CLOSE-003 identity constrains comparison | CLOSED — `check_comparable()` fails closed on benchmark/dataset/scorer mismatch (commit/timestamp advisory); `promotion.decide` returns **INCOMPARABLE** instead of ranking. Regression tests added. | `benchmark_harness_qa.py`, `promotion.py` |
| WA-CLOSE-004 15% threshold | CLOSED — classified **OPERATOR_POLICY** (from IDEAS_BACKLOG.md §B3 line 358: "±1pp quality, ≥15% wall-clock, ≥20% reasoning-tokens, 0.5pp crash ceiling"), NOT empirically derived. Documented as configurable operator policy; test proves eligibility/correctness/quality can never be compensated by performance even at ∞ speed. | §B3, `promotion.py` |

## 4. Provenance of the 15% threshold
Traced to `IDEAS_BACKLOG.md §B3` (twice: line 358 and line 212's A2 acceptance rule "≥15% wall-clock
reduction … quality within ±1pp (ROPE)"). It is an **operator-proposed promotion-gate margin**, not a
measured noise-floor result. Classification: **OPERATOR_POLICY**. Represented as a configurable
`PromotionMargins.perf_win_pct` with a docstring stating it is not a scientific invariant and is to be
revisited once LAB-SERVE-001 establishes the serving noise floor.

## 5. Upstream sources actually inspected (file:line)
- `sglang/bench_serving.py:1-22` — deprecation shim (`from sglang.benchmark.serving import *`).
- `sglang/benchmark/serving.py:932-958` — `ASYNC_REQUEST_FUNCS` / endpoint map: `sglang-oai-chat`
  and `vllm-chat` BOTH → `async_request_openai_chat_completions` + `/v1/chat/completions`.
- `serving.py:966-1015` — `BenchmarkMetrics` (has `total_output` vs `total_output_retokenized`, and
  p95/p99 for ttft/tpot/itl/e2e — the exact fields normalized).
- `serving.py:374-545` — `async_request_openai_chat_completions`: payload uses
  `max_completion_tokens` + `ignore_eos = not args.disable_ignore_eos`; **streaming parse counts
  `reasoning_content` toward output** ("Reasoning models stream thoughts via reasoning_content; count
  them like content"); `output_len` taken from `usage.completion_tokens`. This is why a thinking model
  is handled correctly.
- llama.cpp fork `/v1/chat/completions`: live smoke — non-stream returns `usage.completion_tokens`;
  stream emits `delta.reasoning_content` then `delta.content` (SSE); fingerprint `b10159-068764d92`.
- **Backend choice: `vllm-chat`** (generic OpenAI-chat; avoids `sglang-oai-chat`'s spec-decode
  assumptions at `serving.py:1085` and the `assert backend=="sglang-oai-chat"` at :1928 that a
  llama-server won't satisfy). Proven from source + smoke.

## 6. Serving adapter design (`ops/lab_serve_bench.py`, `ops/lab_serve_normalize.py`)
Thin wrapper: shells `python -m sglang.benchmark.serving --backend vllm-chat --base-url … --model …
--tokenizer /home/augus/models/fp16/base --dataset-name random --random-input-len/-output-len
--random-range-ratio 1.0 --num-prompts … --max-concurrency … --request-rate inf --apply-chat-template
--warmup-requests 2 --seed 42 --output-file <jsonl> --output-details`. It owns: identity, a GPU
telemetry sampler thread (peak VRAM / mean util,power,temp), raw JSONL+stdout+argv preservation,
normalization, and validity checks (reported-vs-retokenized token ratio, success, TTFT>0). It does
NOT reimplement arrivals/concurrency/streaming/percentiles.

## 7. LAB-SERVE-QA-001 (instrument canary): QUALIFIED
N=1, 8 prompts, input 256/output 128 forced. Result: completed 8/8, **token ratio 1.000**
(1024/1024), TTFT median 456 ms, TPOT median 12.4 ms, throughput 62.5 tok/s, VRAM peak 20.8 GB.
`transportCompatible=streamingObserved=tokenAccountingSane=latencyMetricsPlausible=instrumentQualifiedForPilot=true`.
Raw: `runs/serving/LAB-SERVE-001/raw/canary.*`.

## 8. LAB-SERVE-001 matrix ACTUALLY executed
concurrency {1,2,4,8} × MTP {on,off}, model fable-tc-l1.0 (dense 27B), input 1024 / output 128
**forced (ignore_eos)**, num_prompts 16/16/32/64, warmup 2, 1 rep, N-order shuffled (4,1,8,2) per
server config. MTP off = same flags minus `--spec-type draft-mtp --spec-draft-n-max 4`.
Exact commands in the report §Reproduce. Raw paths: `runs/serving/LAB-SERVE-001/raw/mtp_{on,off}_n{1,2,4,8}.*`.

## 9. Normalized results (`normalized/matrix.csv`, `mtp_advantage.json`)
| N | thr on | thr off | MTP thr gain | TPOT on | TPOT off | TPOT Δ | TTFT p95 on/off |
|---|---|---|---|---|---|---|---|
| 1 | 46.8 | 34.1 | +37.2% | 13.6 | 22.1 | −8.5 | 1169/1038 |
| 2 | 56.9 | 47.5 | +19.8% | 23.2 | 30.5 | −7.3 | 2761/1749 |
| 4 | 61.5 | 54.3 | +13.3% | 47.6 | 44.8 | +2.8 | 4214/4212 |
| 8 | 59.2 | 49.6 | +19.4% | 52.4 | 45.3 | +7.1 | 12741/17333 |
ITL median: on ≈ 0.01 ms vs off 22–40 ms (speculative signature). VRAM peak: on ≈21.0 / off ≈18.3 GB.

## 10. Statistical interpretation (shape, not significance)
1 rep/cell → variance unknown; medians + p95/p99 preserved. **MTP wins output throughput in every
measured regime (N=1..8)** — no throughput flip. **Per-token latency (TPOT) FLIPS sign near N=4**
(MTP better N≤2, worse N≥4). MTP verdict: *wins in the measured low-concurrency regime; loses on
latency in the measured high-concurrency regime; no detectable throughput loss to N=8; insufficient
evidence for a universal threshold — do not encode N=4.*

## 11. Failures / negative evidence
- My naïve `setsid nohup` server detach DIED on bash-lc exit; used `lmctl serve` (correct holder
  detach) instead — recorded as a real operational failure, not hidden.
- `accept_length` reads **0** for all cells: that meta_info field is sglang-specific and **llama-server
  does not expose draft acceptance through the path bench reads** — an instrument limitation (MTP
  effect is still visible via throughput/TPOT/ITL, not via bench's accept_length).
- No stop condition (§14) triggered: 100% success, token-sane every cell, VRAM ≤ 21.2 GB < 24.

## 12. Source excerpts (this session's code)
`benchmark_harness_qa.py::check_comparable` (WA-CLOSE-003):
```python
COMPARISON_INVALIDATING = ("benchmark_name","benchmark_version","dataset_hash","scorer_version")
def check_comparable(id_a, id_b) -> dict:   # fail closed on invalidating; commit/timestamp advisory
    invalidating = [...]; return {"comparable": not invalidating, "invalidating":..., "advisory":...}
```
`analysis/promotion.py::decide` (Stage 0 INCOMPARABLE + lexicographic):
```python
if candidate_identity and baseline_identity and not check_comparable(...)["comparable"]:
    return PromotionDecision("INCOMPARABLE","identity",...)   # never rank apples vs oranges
```
`ops/lab_serve_bench.py::run_cell` validity gate:
```python
tok_ratio = total_out_re/total_out; validity["token_accounting_sane"] = 0.9<=tok_ratio<=1.1
```

## 13. Committed diffs / staged vs unstaged
All work is committed (tracked+staged trees clean). Diffstat `a47a9c2..2c81cb7`: 48 files,
+37,597 (mostly raw per-request JSONL summaries). Inspect: `git show 35a02ef 894b341 2c81cb7`.
No uncommitted or staged changes remain.

## 14. Rollback / reproduction
- Reproduce QA: `PYTHONPATH=src python -m model_lifecycle.analysis.promotion` (self-check incl.
  WA-CLOSE tests); `python tests/benchmark_harness/benchmark_harness_selftest.py` (16/16);
  `/home/augus/evalplus-venv/bin/python tests/benchmark_harness/evalplus_sentinel.py` (integration).
- Reproduce serving: report §Reproduce (serve → 8 cells → normalize → stop).
- Rollback whole packet: `git reset --hard a47a9c2` (nothing pushed). Single item: revert its commit.

## 15. Decision states (§17)
- **Wave A: CLOSED** (four gaps closed, all tests green).
- **LAB-SERVE-QA-001: QUALIFIED.**
- **LAB-SERVE-001: PILOT_COMPLETE** (bounded, 1 rep/cell, dense-27B, forced length).
- **MTP:** wins in measured low-concurrency regime; loses on latency in measured high-concurrency
  regime; no detectable throughput loss to N=8; insufficient evidence for a universal threshold.

## 16. Exactly one recommended next packet
**LAB-SERVE-001b — variance + realism + MoE:** (a) repeat N∈{1,4} cells ×5 to get per-cell variance
and apply the existing bootstrap/median/noise-floor discipline before any promotion use; (b) add one
realistic-EOS workload (preserve normal termination) alongside the forced-length calibration; (c)
re-run the MTP on/off × N matrix on the **35B MoE deploy** (extract its tokenizer for exact
accounting) to test whether the TPOT flip point differs from the dense-27B pilot. Do NOT expand to
128k or LAB-SERVE-002 profile promotion until variance is known.
