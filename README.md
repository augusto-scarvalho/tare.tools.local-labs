# tare.tools.local-labs 🔬🧪

**Local AI Lab & Inference Test Bench for the `tare.tools` Ecosystem**

`tare.tools.local-labs` is the dedicated empirical research, lifecycle benchmarking, and local inference optimization test bench for high-performance open-source models (Qwen, Llama, Gemma, Mamba/RNN, Mistral). It hosts automated harnesses for long-context evaluation, speculative decoding (MTP), aggressive quantization frontier sweeps, latency/throughput profiling, and multi-stage lexicographic promotion gating.

📖 **Canonical Research Catalog**: For the complete curated index of all studies, hypotheses, empirical results, and negative boundaries, see [`docs/RESEARCH_CATALOG.md`](docs/RESEARCH_CATALOG.md).
📋 **Living Handoff & Research Backlog**: For active operational state, completed milestones (Realizado), ongoing research tracks (Em Andamento), and runbooks, see [`docs/HANDOFF.md`](docs/HANDOFF.md).

---

## 📁 Repository Architecture

```text
tare.tools.local-labs/
├── src/model_lifecycle/        # Core Python package & lifecycle analysis engine
│   ├── analysis/               # Multi-stage gates, statistics, robust metrics, QA
│   ├── collectors/             # Host, telemetry, and streaming response collectors
│   ├── control_plane/          # Guard rails, run planners, recovery policies
│   ├── reports/                # A/B diff analyzers, status generators
│   ├── servers/                # Llama.cpp & SGLang runner adapters
│   ├── storage/                # Run ledger & database storage
│   └── workloads/              # Workload generators & throughput runners
│
├── ops/                        # Operational campaign playbooks & runners
│   ├── qwen38-bringup/         # Qwen3.8 bringup, quant frontier, budget curves
│   ├── fork-consolidation/     # Llama.cpp build trees preservation & audit
│   └── wsl/                    # Reusable low-friction WSL runner (`wslx.sh`)
│
├── tools/                      # Modular research tools & probe suites
│   ├── probes/                 # Prefill, context, KV, VLM, refusal probes
│   ├── benchmarks/             # Concision, HumanEval+, GSM8K, MATH-500 benches
│   ├── gates/                  # MTP verification, gate judges, Taguchi screen
│   ├── analysis/               # A/B comparisons, GGUF inspectors, matrix scorers
│   ├── scripts_sh/             # Campaign automation shell scripts
│   └── scripts_ps1/            # Windows / WSL environment configuration scripts
│
├── docs/                       # Curated research documentation & campaign ledgers
│   ├── HANDOFF.md              # Living master handoff (Realizado / Em Andamento / Encerrado)
│   ├── RESEARCH_CATALOG.md     # Master scientific catalog & findings matrix
│   ├── campaigns/              # Focused empirical campaign directories:
│   │   ├── a1-mtp/             # Multi-token prediction (MTP) spec-decode
│   │   ├── a2-ablation-merging/# Layer ablation, merging, and refusal gates
│   │   ├── a3-kv-quant/        # KV-cache quantization limits (Q4, Q8, Asymmetric)
│   │   ├── a4-instrumentation/ # Telemetry & latency instrumentation
│   │   ├── gdn-kernel/         # Gated Delta Net & custom kernel optimization
│   │   ├── rnn-mamba/          # State model caching & Mamba-2 integration
│   │   ├── vlm/                # Vision-Language Model latency & refusal probes
│   │   └── serving/            # Parallel slot capacity, scheduling & serving
│   ├── architecture/           # System design & relay protocol specifications
│   └── research/               # STATUS.md, EXPERIMENTS.md, LANDSCAPE.md, MECHANISMS.md
│
├── patches/                    # Custom engine kernel patches (e.g. B2b KV host-pin)
├── runs/                       # Durable empirical run logs (JSON/JSONL evidence)
├── tests/                      # Deterministic harness verification suites
│   └── benchmark_harness/      # LAB-QA-001 / LAB-QA-002 harness qualification
└── workloads/                  # Benchmark problem sets (GSM8K, HumanEval+)
```

---

## 🎯 Key Research Highlights & Findings

1. **Quantization Frontier (Qwen 3.8 / 27B)**:
   - **`Q2_K_XL` (9.9GB) is the optimal Pareto point**: Frees ~7GB VRAM vs `Q4_K_XL` (16.7GB) with **zero measurable loss** on coding (HumanEval+ `0.896`), competition math (MATH-500 L5 `90%`), and deep long-context retrieval (100% up to 65k+).
   - **`IQ2_M` Long-Context Cliff**: Extreme quantization (`9.6GB`) holds on short benchmarks but systematically drops deep needle retrieval at $\ge 32k$.

2. **Reasoning vs. Direct Instruction**:
   - For HumanEval-class coding and bounded mathematical tasks, pure `instruct` mode consistently outperforms thinking mode (95.0% vs 86.7% at budget 8192) without token bloat or truncation artifacts.

3. **MTP Speculative Decoding**:
   - Speculative decoding with multi-token prediction heads yields a **~2.1x throughput speedup** on code generation workloads with `n-max 3` draft depth.

4. **RNN & State Caching**:
   - Deterministic in-process state reload and memory caching verified for linear-time architectures (Mamba-2 / GDN) with bit-exact reproducibility (40/40).

---

## 🚀 Running Verification & Benchmarks

### Harness Self-Test (Qualification Gate)
Run the deterministic self-test suite (exercises sample validation, dataset hashing, and cache bust mechanics in < 2 seconds without GPU dependencies):
```bash
python tests/benchmark_harness/benchmark_harness_selftest.py
```

### Context & Retrieval Probes
Run parameterized needle-in-a-haystack sweeps across context lengths:
```bash
bash tools/scripts_sh/phase_a_ctx.sh
```

### Lexicographic Promotion Gates
Evaluate model candidates using strict non-inferiority margins:
```python
from model_lifecycle.analysis.promotion import evaluate_promotion, PromotionMargins
# Strict evaluation: eligibility -> correctness -> quality -> performance
```

---

## 🛡️ Governance & Integrity
- All raw experiment runs produce structured JSON/JSONL artifacts committed under `runs/`.
- Binary SQLite databases and temporary logs are strictly ephemeral derivatives excluded by `.gitignore`.
- Deterministic dataset hashes guarantee immutable cross-run auditability.
