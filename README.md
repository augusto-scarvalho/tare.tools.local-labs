# tare.tools.local-labs

*Versão em português: [README.pt-BR.md](README.pt-BR.md).*

[![CI](https://github.com/augusto-scarvalho/tare.tools.local-labs/actions/workflows/ci.yml/badge.svg)](https://github.com/augusto-scarvalho/tare.tools.local-labs/actions/workflows/ci.yml)

The research lab and benchmark bench of tare.tools for running open-weight language models locally: one
consumer GPU (an RTX 3090 with 24 GB) and 64 GB of host RAM. It holds the benchmark harnesses, probes and
statistics used to qualify models, quantizations and inference-engine levers, and the write-ups that explain
the results.

Claims are measured against the machine's noise floor (about 2.3 % paired scatter) with distribution-free
statistics: exact sign tests, bootstrap confidence intervals and Cliff's delta. Candidates advance through
ordered gates: eligibility, then correctness, then quality, then performance.

## What is published

This repository publishes the tooling, the tests and the write-ups. The raw run records (`runs/`), the
machine-readable research backlog and its packets, agent handoffs and audit ledgers, raw evidence bundles,
scripts that only work on the maintainer's machines, third-party datasets and model weights are kept private.
Write-ups still name the run records they summarize, but those records are not in this repository.

## Relation to the other tare.tools repositories

- [tare.tools.os](https://github.com/augusto-scarvalho/tare.tools.os) is the agent product. It does not pin
  or import this repository; some of its guides link to results published here, such as the
  [locality and loading-cost study](docs/research/LOCALITY_COST_QUALIFICATION.md).
- The service on the GPU node (model gateway, shared GPU lease and ComfyUI hook) is `node/` in
  [tare.tools.os](https://github.com/augusto-scarvalho/tare.tools.os/tree/main/node).
  It was split out of this repository on 2026-10-09. Its catalog of qualified models cites the qualification work done
  here, and `tools/agents/modelctl.py` needs it installed.
- [slop.cpp](https://github.com/augusto-scarvalho/slop.cpp), a llama.cpp fork, owns the engine code, runtime
  flags and builds. This repository owns experiment design, statistics and promotion decisions;
  [`patches/`](patches/README.md) only keeps archival patch receipts.

## Requirements

- Python 3.12 (the version CI tests) and Git.
- The tests need `pytest` and `numpy` only. They use no GPU, model, network service or private record.
- Running the experiments themselves needs an NVIDIA GPU, a slop.cpp or llama.cpp build and the model files.
  Most scripts were written for WSL2 (Ubuntu 24.04) on a Windows workstation and default to a
  `~/src/slop.cpp*` and `~/models/...` layout; adjust those paths or pass the script's options.

## Install and test

```bash
git clone https://github.com/augusto-scarvalho/tare.tools.local-labs.git
cd tare.tools.local-labs
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install pytest==9.1.1 numpy==2.5.1
python -m compileall -q src tools tests benchmark_harness_qa.py
python -m pytest -q
python tests/benchmark_harness/benchmark_harness_selftest.py
```

The last command qualifies the benchmark harness in a few seconds without a GPU and ends with
`LAB-QA-001: 23/23 passed`; it rewrites `tests/benchmark_harness/benchmark_selftest_report.json`. These are
the steps CI runs. CI also runs the research-backlog gate, which is skipped when the private backlog is absent.

Material changes need a new bullet under `## Unreleased` in [`CHANGELOG.md`](CHANGELOG.md). Install the
pre-push hook that checks it with `python tools/install_git_hooks.py`; see
[`docs/CHANGELOG_POLICY.md`](docs/CHANGELOG_POLICY.md).

## Benchmark datasets

The GSM8K, HumanEval+ and MBPP+ copies the benchmarks read are not redistributed here. Regenerate them with
the exporters (they need the Hugging Face `datasets` package, or `evalplus` for HumanEval+ and MBPP+):
`tools/analysis/export_gsm8k.py` (writes `workloads/gsm8k.jsonl`), `tools/analysis/export_humaneval_plus.py`
(writes `tools/benchmarks/workloads/humaneval_plus.jsonl`) and `tools/benchmarks/export_mbpp_plus.py OUTPUT`.

## Layout

```text
src/model_lifecycle/   Python package: promotion gates and statistics, benchmark QA, collectors,
                       run planner and guard, report generators, server adapters
tools/                 benchmarks, probes, analysis and gates (index: tools/README.md)
ops/                   campaign playbooks: Qwen3.8 bring-up, recurrent models, serving, WSL runner (ops/README.md)
docs/                  write-ups, research catalog and method contracts (map: docs/README.md)
config/                qualified-model registry, backlog priority policy, choice-helper smoke cases
workloads/             MBPP+ identity record and synthetic VLM fixtures
tests/                 unit tests and the deterministic benchmark-harness qualification
patches/, reports/     archival engine patch and generated evidence tables
```

## Findings

The [research catalog](docs/RESEARCH_CATALOG.md) indexes every campaign; the main results are:

- **Quantization.** For Qwen3.8-27B, `UD-Q2_K_XL` (9.9 GB) kept coding and MATH-500 L5 quality and deep
  long-context retrieval, while `IQ2_M` lost needles at 32k and beyond
  ([quant frontier](ops/qwen38-bringup/QUANT_FRONTIER_CAMPAIGN.md)).
- **Reasoning budget.** On an exact 60-problem HumanEval+ set, Qwen3.8 in instruct mode scored 95.0 %, above
  low (93.3 %) and medium (88.3 %) reasoning; the default highest effort truncated 31 of the 60 answers
  ([bring-up report](ops/qwen38-bringup/README.md), [budget curve](ops/qwen38-bringup/BUDGET_CURVE_CLOSURE.md)).
- **KV cache.** Symmetric `q4_0`/`q4_0` KV stayed on the fused GPU path with lossless needle recall;
  asymmetric `q8_0`/`q4_0` fell back and cost about 57 % of throughput
  ([A3 KV quantization](docs/campaigns/a3-kv-quant/A3_KV_QUANT.md)).
- **Speculative decoding.** The built-in MTP head with `--spec-draft-n-max 3` gave about 2.1x decode on code
  for the recorded build and model; requalify it for every model and build.
- **Engine levers.** Pinned host KV buffers, expert prefetch, a hot-expert cache and a chunked GDN scan were
  measured on the RTX 3090; see the [fork report](docs/research/FORK.md) for which ones help and where.

The serving launcher with these settings is [`ops/qwen38-bringup/serve.sh`](ops/qwen38-bringup/serve.sh).

## License

Licensed under the [Apache License, Version 2.0](https://opensource.org/licenses/Apache-2.0).
