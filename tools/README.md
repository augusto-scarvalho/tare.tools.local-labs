# tare.tools.local-labs — Master Tooling & Script Index 🛠️🔬

> **Index Overview**: All research tools, latency probes, benchmark runners, evaluation gates, and maintenance scripts in `tools/` are categorized below by operational utility.

---

## 🧭 Taxonomy & Directory Map

```text
tools/
├── analysis/       # Statistical analyzers, A/B diff calculators, GGUF inspectors, matrix scorers
├── benchmarks/     # Automated evaluation workloads (HumanEval+, GSM8K, MATH-500, Concision, DSpark)
├── gates/          # Promotion gates, MTP token-exactness verifiers, blind judge quorums
├── probes/         # Telemetry probes (TTFT, KV memory, NIAH context, VLM refusal, Prefill)
├── scripts_sh/     # Shell scripts for WSL/Linux execution (profilers, builds, judge servers)
└── scripts_ps1/    # Windows / PowerShell management scripts (environment, GPU recovery)
```

---

## 🎯 1. Benchmarks & Evaluation Workloads (`tools/benchmarks/`)

| Script / Tool | Category | Target / Model | Purpose & Output |
|---|---|---|---|
| [`a2_concision_bench.py`](benchmarks/a2_concision_bench.py) | Concision | Qwen 3.6 / Fable / Merges | Evaluates token-budget efficiency and verbosity reduction on GSM8K paired sets. |
| [`quality_bench.py`](benchmarks/quality_bench.py) | Quality | All Models | General quality benchmark across reasoning, logic, and standard evaluation sets. |
| [`agent_bench.py`](benchmarks/agent_bench.py) | Agentic | Agent models | Evaluates multi-turn tool-calling and nested function schema parsing. |
| [`a1_mtp_depth_bench.py`](benchmarks/a1_mtp_depth_bench.py) | Speculation | Qwen MTP | Measures parallel token acceptance rates across draft depths $n \in [1, 5]$. |
| [`a5_dspark/`](benchmarks/a5_dspark/) | Speculation | DSpark / Dense | A/B comparative suite for DSpark speculative decoding vs native MTP heads. |
| [`gdn_conc_bench.py`](benchmarks/gdn_conc_bench.py) | Recurrence | GDN / Hybrids | Multi-head concurrency benchmark for Gated Delta Net recurrence kernels. |
| [`vlm_bench.py`](benchmarks/vlm_bench.py) | Multimodal | Gemma-4 / VLMs | Automated vision benchmark measuring OCR and visual reasoning accuracy. |
| [`vlm_vqa_bench.py`](benchmarks/vlm_vqa_bench.py) | Multimodal | VLMs | Visual Question Answering evaluation on desktop screenshots. |
| [`lmctl.py`](benchmarks/lmctl.py) | Control | Local Models | Unified CLI harness for triggering local model evaluation runs. |
| [`run_one.py`](benchmarks/run_one.py) | Runner | Single Prompt | Isolated single-shot runner with token-timing and latency instrumentation. |

---

## 🔍 2. Probes & Hardware Telemetry (`tools/probes/`)

| Probe Script | Target Lever | Purpose & Measured Metric |
|---|---|---|
| [`context_probe.py`](probes/context_probe.py) | Long Context | Single-needle and multi-needle Needle-in-a-Haystack (NIAH) up to 262k context. |
| [`prefill_probe.py`](probes/prefill_probe.py) | Prefill / TTFT | Measures prompt ingestion throughput (tokens/sec) and Time-to-First-Token. |
| [`quant_probe.py`](probes/quant_probe.py) | Quantization | Fast scalar quality and perplexity probe across GGUF quantization types. |
| [`cache_probe.py`](probes/cache_probe.py) | Prefix Reuse | Verifies LCP prefix cache hits and context checkpoint restoration. |
| [`concurrency_probe.py`](probes/concurrency_probe.py) | Parallel Slots | Multi-client concurrency probe measuring inter-token latency under slot contention. |
| [`multihop_probe.py`](probes/multihop_probe.py) | Multi-fact | Multi-hop reasoning and associative retrieval depth probe. |
| [`probe_timings.py`](probes/probe_timings.py) | Latency | Microsecond-resolution breakdown of prefill, sample, and decode phases. |
| [`a4_spec_metrics_probe.py`](probes/a4_spec_metrics_probe.py) | MTP | Speculative decoding acceptance rate and speculative tail-latency telemetry. |
| [`auto_run_tc_mtp.py`](probes/auto_run_tc_mtp.py) | MTP Head | Automated verification of block-64 `nextn` draft head tensors in GGUF models. |
| [`gguf_nextn_probe.py`](probes/gguf_nextn_probe.py) | MTP Tensor | Direct inspection of GGUF metadata for MTP draft head architecture. |
| [`dl_tc_mtp.py`](probes/dl_tc_mtp.py) | MTP Graft | Helper tool to download and graft missing MTP draft tensors. |
| [`a2_refusal_probe.py`](probes/a2_refusal_probe.py) | Alignment | Multi-tier prompt refusal probe measuring compliance on uncensored merges. |
| [`a2_refusal_one.py`](probes/a2_refusal_one.py) | Alignment | Single-prompt refusal and compliance diagnostic. |
| [`heretic_run.py`](probes/heretic_run.py) | Alignment | Evaluation runner for Heretic de-alignment and refusal ablation. |
| [`vlm_probe.py`](probes/vlm_probe.py) | Vision | Latency and bounding box coordinate probe for vision-language models. |
| [`vlm_refusal_probe.py`](probes/vlm_refusal_probe.py) | Vision | Tests refusal boundaries on safety-adjacent visual images and telemetry. |
| [`vlm_mature_probe.py`](probes/vlm_mature_probe.py) | Vision | Extended multi-turn conversational probe for VLM agents. |
| [`vlm_test.py`](probes/vlm_test.py) | GUI Vision | Desktop UI fixture test runner using `workloads/vlm_fixtures/`. |

---

## ⚖️ 3. Promotion & Qualification Gates (`tools/gates/`)

| Gate Tool | Gate Level | Criteria & Qualification Standard |
|---|---|---|
| [`agentic_gate.py`](gates/agentic_gate.py) | Production | Comprehensive multi-step agentic promotion qualification suite. |
| [`taguchi_screen.py`](gates/taguchi_screen.py) | Screening | Taguchi Orthogonal Array screener for hyperparameter sensitivity analysis. |
| [`verify_mtp.py`](gates/verify_mtp.py) | Speculation | Deterministic token-identity gate verifying MTP draft output matches baseline. |
| [`verify_mtp_long.py`](gates/verify_mtp_long.py) | Speculation | MTP token-exactness verifier under deep context ($\ge 65k$). |
| [`verify_prefetch.py`](gates/verify_prefetch.py) | Offloading | Verifies CUDA stream overlap and absence of memory staging corruption. |
| [`a2_reconstruct_gate.py`](gates/a2_reconstruct_gate.py) | Merging | Low-rank SVD reconstruction gate (fidelity $\ge 0.65$, length ratio $\le 1.10$). |
| [`gate3/build_gate3_artifact.py`](gates/gate3/build_gate3_artifact.py) | Quality Quorum | Compiles multi-judge blind pairwise outputs into review artifacts. |
| [`gate3/claude_verdicts_raw.py`](gates/gate3/claude_verdicts_raw.py) | Judge Quorum | Raw parsing adapter for Claude Opus / Sonnet judge verdicts. |
| [`gate3/opus_verdicts_raw.py`](gates/gate3/opus_verdicts_raw.py) | Judge Quorum | Parsed score accumulator for Gate 3 judge quorums. |

---

## 📊 4. Statistical Analysis & Scorer Tools (`tools/analysis/`)

| Analysis Tool | Focus | Methodology & Capabilities |
|---|---|---|
| [`analyze_ab.py`](analysis/analyze_ab.py) | A/B Testing | Core distribution-free A/B analyzer: exact sign tests, bootstrap CIs, Cliff's $\delta$. |
| [`changelog_guard.py`](analysis/changelog_guard.py) | Repository Policy | Enforces meaningful `Unreleased` entries and append-only changelog history in local pre-push and CI; the hook also prints a non-blocking publication/CI reminder. |
| [`launch_watched_experiment.py`](analysis/launch_watched_experiment.py) | Experiment Lifecycle | Starts one experiment and its persistent watcher, records launch provenance, and keeps the controller bound to compact completion delivery by default. |
| [`watch_experiment_processes.py`](analysis/watch_experiment_processes.py) | Experiment Lifecycle | Watches Windows/WSL process state, worker exits, physical progress, GPU and service health; fail-closed advances valid packets only to `EXECUTED` and refreshes the backlog queue. |
| [`smoke_experiment_mode.py`](analysis/smoke_experiment_mode.py) | Experiment Lifecycle | Runs a temporary, non-mutating live canary through foreground completion delivery and backlog refresh. |
| [`mutation_test_experiment_harness.py`](analysis/mutation_test_experiment_harness.py) | Experiment Lifecycle | Runs the frozen seeded mutation gate in isolated copies and persists an independently reproducible JSON report. |
| [`ab_compare.py`](analysis/ab_compare.py) | Comparative | Direct paired round comparator across memory, latency, and throughput. |
| [`ab_isolate.py`](analysis/ab_isolate.py) | Isolation | Environment isolation harness ensuring identical GPU thermals between arms. |
| [`a2_stats.py`](analysis/a2_stats.py) | Concision | Statistical Wilcoxon signed-rank and non-inferiority test calculator. |
| [`a2_merge_raw.py`](analysis/a2_merge_raw.py) | Merging | Full-rank task arithmetic weight merge engine on safetensors: $W = W_1 + \lambda(W_2 - W_0)$. |
| [`a2_stage2_extract.py`](analysis/a2_stage2_extract.py) | Ablation | Directional refusal vector extraction and Arditi layer-selection harness. |
| [`a2_score_humaneval.py`](analysis/a2_score_humaneval.py) | Scorer | HumanEval+ and GSM8K offline scorer and pass@1 calculator. |
| [`summarize_market_bench.py`](analysis/summarize_market_bench.py) | Market Bench | Generates consolidated market baseline summaries (`market-r0`). |
| [`gguf_meta.py`](analysis/gguf_meta.py) | GGUF | Extracts tensor geometry, quantization metadata, and architecture hyper-params. |
| [`gguf_kv.py`](analysis/gguf_kv.py) | GGUF | Direct reader and modifier for GGUF key-value metadata headers. |
| [`read_geometry.py`](analysis/read_geometry.py) | Architecture | Analyzes layer types, head dimensions, and expert layout from GGUF tensors. |
| [`analyze_moe_skew.py`](analysis/analyze_moe_skew.py) | MoE Routing | Computes routing entropy, expert skewness, and load balance coefficients. |
| [`score_matrix.py`](analysis/score_matrix.py) | Scoring | Multi-metric matrix scorer aggregating quality, length, and timing. |
| [`score_subset.py`](analysis/score_subset.py) | Scoring | Scores stratified subsets of evaluation benchmarks. |
| [`export_gsm8k.py`](analysis/export_gsm8k.py) | Exporter | Converts raw GSM8K test outputs into structured comparison tables. |
| [`json2tsv.py`](analysis/json2tsv.py) | Converter | Formats JSON run logs into clean TSV tables for graphing. |
| [`judge_keys.py`](analysis/judge_keys.py) | Judge Quorum | Key extractor and validator for LLM-as-a-judge blind evaluation sets. |
| [`prefill_sweep.py`](analysis/prefill_sweep.py) | Tuning | Analyzes prefill latency curves across batch sizes and context lengths. |
| [`residency_sweep.py`](analysis/residency_sweep.py) | VRAM | Sweeps expert offload allocation to find maximum GPU residency points. |
| [`optimize_config.py`](analysis/optimize_config.py) | Config | Automated solver for optimal `--n-cpu-moe` and `--ubatch-size` parameters. |
| [`quick_archaeology.py`](analysis/quick_archaeology.py) | History | Quick historical search tool across raw run ledgers. |
| [`search_archaeology.py`](analysis/search_archaeology.py) | History | Full semantic and keyword search across historical experiment data. |

---

## 🐚 5. Shell & Environment Scripts (`tools/scripts_sh/` & `tools/scripts_ps1/`)

### Linux / WSL Execution Scripts (`tools/scripts_sh/`)
- [`bless_fork.sh`](scripts_sh/bless_fork.sh): 3-tier qualification gate (G1: B2b Pinning, G2: MTP Identity, G3: `-nkvo` Coherence).
- [`finalize_fork.sh`](scripts_sh/finalize_fork.sh) & [`fork_setup.sh`](scripts_sh/fork_setup.sh): Build and setup scripts for `llama.cpp-master @ lifecycle`.
- [`phase_a_ctx.sh`](scripts_sh/phase_a_ctx.sh) & [`phase_a_batch.sh`](scripts_sh/phase_a_batch.sh): Automated context and batch sweep runners.
- [`probe_b2b_pin.sh`](scripts_sh/probe_b2b_pin.sh), [`probe_b2_kvram.sh`](scripts_sh/probe_b2_kvram.sh), [`probe_b5_spill.sh`](scripts_sh/probe_b5_spill.sh): Direct memory and kernel offloading probes.
- [`probe_e5_cache.sh`](scripts_sh/probe_e5_cache.sh) & [`gen_moe_trace.sh`](scripts_sh/gen_moe_trace.sh): MoE expert trace generators for hot-expert caching.
- [`spec-drafter-bench.sh`](scripts_sh/spec-drafter-bench.sh): Speculative drafter throughput and acceptance benchmark.
- [`kv-quant-bench.sh`](scripts_sh/kv-quant-bench.sh): KV-cache quantization benchmark runner.
- [`serve_gemma_judge.sh`](scripts_sh/serve_gemma_judge.sh) & [`serve_mistral_judge.sh`](scripts_sh/serve_mistral_judge.sh): Background servers for automated judge quorums.
- [`merge_stage1.sh`](scripts_sh/merge_stage1.sh), [`dl_fp16.sh`](scripts_sh/dl_fp16.sh), [`install_mergekit.sh`](scripts_sh/install_mergekit.sh): Mergekit installation and task-arithmetic runners.

### Windows / PowerShell Scripts (`tools/scripts_ps1/`)
- [`gpu_recover.ps1`](scripts_ps1/gpu_recover.ps1): Recovers NVIDIA GPU driver state and resets display adapters after memory faults.
- [`run_market_bench.ps1`](scripts_ps1/run_market_bench.ps1): Automated market baseline benchmark runner on Windows/WSL.
- [`stage1_full.ps1`](scripts_ps1/stage1_full.ps1) & [`stage1_eval.ps1`](scripts_ps1/stage1_eval.ps1): Stage 1 merge evaluation runners.
- [`refusal_rerun.ps1`](scripts_ps1/refusal_rerun.ps1): Automated refusal evaluation re-run script.
- [`gen_test_images.ps1`](scripts_ps1/gen_test_images.ps1): Generates synthetic UI test images for VLM probes.
- [`_apply-wslconfig.ps1`](scripts_ps1/_apply-wslconfig.ps1), [`_probe-python.ps1`](scripts_ps1/_probe-python.ps1), [`_verify-python.ps1`](scripts_ps1/_verify-python.ps1): WSL and Python environment configuration utilities.
