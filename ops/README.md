# tare.tools.local-labs — Operational Campaigns & Playbooks 🚀🎯

> **Overview**: `ops/` contains active and completed operational campaign playbooks, bring-up harnesses, detached execution scripts, and environment stability configurations.

---

## 📁 Operational Subdirectories

```text
ops/
├── qwen38-bringup/         # Active Qwen 3.8-27B bringup, quant frontier & budget curves
├── rnn-campaign/           # Recurrent state models (Mamba-2, GDN, TPTT, NoLiMa) runners
├── serving-campaign/       # High-concurrency serving benchmarks & slot topologies
├── qualified-model-fleet/  # Embedding reindex server launcher
├── windows-host/           # Host status snapshot read by the node's fleet view
└── wsl/                    # Windows -> WSL script runner (`wslx.sh`)
```

Machine-specific playbooks (GPU driver and registry settings, build-tree archaeology, one-off close-out
probes, service swaps on the maintainer's GPU node) are kept private and are not in this repository.

---

## 🎯 1. Active Bring-Up: `ops/qwen38-bringup/`

The primary active deployment playbook for Qwen 3.8-27B on consumer 24GB GPUs:

- [`README.md`](qwen38-bringup/README.md): Master bringup report, phase results, frozen serve config, and shootout matrix.
- [`serve.sh`](qwen38-bringup/serve.sh): Official production launcher for `llama-server` (UD-Q2_K_XL @ 65k, draft-mtp-n3, q4_0 KV).
- [`code_eval.sh`](qwen38-bringup/code_eval.sh): Automated HumanEval+ evaluation harness via HTTP completions.
- [`ctx_curve.sh`](qwen38-bringup/ctx_curve.sh): Deep single-needle NIAH long-context probe (8k to 131k context).
- [`mtp_throughput.sh`](qwen38-bringup/mtp_throughput.sh): Speculative decoding throughput bench (GEN vs EDIT modes).
- [`kv_recall_sweep.sh`](qwen38-bringup/kv_recall_sweep.sh): KV-cache quantization retrieval verification (f16 vs q8 vs q4).
- [`ab60_vs_frota.sh`](qwen38-bringup/ab60_vs_frota.sh): Exact 60-problem HumanEval+ shootout vs ThinkingCap and Fable-TC.
- [`quant-frontier.html`](qwen38-bringup/quant-frontier.html): Interactive visual analytics dashboard for the Pareto frontier.
- [`QUANT_FRONTIER_CAMPAIGN.md`](qwen38-bringup/QUANT_FRONTIER_CAMPAIGN.md): Formal research ledger for the 7-quant ladder.
- [`BUDGET_CURVE_CLOSURE.md`](qwen38-bringup/BUDGET_CURVE_CLOSURE.md): Falsification report for reasoning mode in software engineering.
- [`CUSTOM_QUANT_DECISION.md`](qwen38-bringup/CUSTOM_QUANT_DECISION.md): Decision matrix and triggers for custom importance matrices.
- [`VARIANTS.md`](qwen38-bringup/VARIANTS.md): Ecosystem watch list for upcoming fine-tunes and merges.

---

## 🧠 2. Recurrent State Campaign: `ops/rnn-campaign/`

Contains the complete implementation, calibration, and evaluation suite for linear-time recurrent models:

- **Mamba-2 & GDN Foundations**: `rnn_arch_matrix_gen.py`, `rnn_delta_substrate.py`, `rnn_gdn_state_probe.py`, `rnn_ruler_smoke.py`.
- **Memory Caching Sweeps**: `rnn_mc_05a.py`, `rnn_mc_05b.py`, `rnn_mc_analyze.py`, `rnn_mc_bench.py`, `rnn_mc_experiment.py`, `rnn_mc_substrate.py`.
- **Mamba-2 Lifecycle & Transports**: `rnn_06a_mamba_lifecycle.py`, `rnn_06a2_continuation.py`, `rnn_06b_base.py`, `rnn_06c_base.py`, `rnn_06d0_ceiling.py`, `rnn_06d1_recovery.py`, `rnn_06t_*.py`, `rnn_06t2_*.py`.
- **NoLiMa Semi-Synthetic Bridge**: `rnn_07a_bridge_lib.py`, `rnn_07a_bridge_long.py`, `rnn_07a_bridge_r1.py`, `rnn_07a_bridge_r1_replay.py`, `rnn_07a_bridge_short.py`.
- **TPTT Test-Time Prompt Adaptation**: `rnn_tptt_analyze.py`, `rnn_tptt_canary.py`, `rnn_tptt_experiment.py`, `rnn_tptt_gates.py`, `rnn_tptt_lifecycle.py`, `rnn_tptt_memcalib.py`.

---

## 🌐 3. High-Concurrency Serving: `ops/serving-campaign/`

Harnesses for evaluating slot topologies, batch queuing, and TTFT micro-batching:

- `lab_serve_bench.py` & `lab_serve_bench_openloop.py`: Closed-loop and open-loop concurrency load generators.
- `lab_serve_analyze.py` & `lab_serve_openloop_analyze.py`: Statistical percentile breakdown of TTFT and inter-token latency.
- `lab_serve_normalize.py` & `lab_serve_replicate.py`: Normalization against machine baseline and multi-round replication.
- `lab_serve_workload_gen.py`: Synthetic prompt workload generator for multi-tenant stress tests.
- `lab_serve_openloop_replicate.sh`: Automated replication script for open-loop queueing experiments.

---

## 🔧 4. Low-Friction Execution & Environment Runners

- **`ops/wsl/wslx.sh`**:
  Runs a bash script inside WSL from Windows: stages a CR-stripped copy, runs it under a login shell, logs its
  output and can detach it with `setsid`. See [`wsl/README.md`](wsl/README.md).
- **`ops/windows-host/host-status.ps1`**:
  Writes the `host-status.json` snapshot that the node service's fleet view reads.
- **`ops/qualified-model-fleet/embedding_reindex_server.sh`**:
  Starts, stops and reports the on-demand embedding reindex server (`start [full|hybrid|cpu] | stop | status`); see [`EMBEDDING_SERVICE.md`](../docs/EMBEDDING_SERVICE.md).
