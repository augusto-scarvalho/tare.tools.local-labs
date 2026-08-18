# tare.tools.local-labs — Operational Campaigns & Playbooks 🚀🎯

> **Overview**: `ops/` contains active and completed operational campaign playbooks, bring-up harnesses, detached execution scripts, and environment stability configurations.

---

## 📁 Operational Subdirectories

```text
ops/
├── qwen38-bringup/         # Active Qwen 3.8-27B bringup, quant frontier & budget curves
├── rnn-campaign/           # Recurrent state models (Mamba-2, GDN, TPTT, NoLiMa) runners
├── serving-campaign/       # High-concurrency serving benchmarks & slot topologies
├── fork-consolidation/     # Llama.cpp multi-tree build archaeology and branch audit
├── gpu-stability/          # NVIDIA driver profiles, registry keys & reboot recovery
├── wsl/                    # Low-friction background runner (`wslx.sh`)
└── close-outs/             # Specialized close-out probes (e.g. Fable termination probe)
```

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
  Detached background execution runner for Windows $\to$ WSL. Dispatches long-running Python/Bash jobs into detached tmux/nohup sessions, streams stdout to dedicated log files, and captures exit codes cleanly.
- **`ops/gpu-stability/`**:
  Maintains registry patches and driver profiles (`revert_gpu_prefs.ps1`, `revert_wu_noautoreboot.ps1`, `gpu_prefs_backup.reg`) to prevent Windows Update reboots or display driver timeout resets during overnight benchmarks.
- **`ops/fork-consolidation/`**:
  Audits and preserves all 5 local `llama.cpp-*` build trees (`enum_builds.sh`, `consolidate_audit.sh`, `preserve_branches.sh`).
