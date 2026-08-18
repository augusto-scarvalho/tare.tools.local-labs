#!/usr/bin/env python3
"""Normalize LAB-SERVE-001 cell outputs into matrix.csv/json + identity.json. Reads the per-cell
*.normalized.json (which embed the raw upstream summary). Raw JSONL is never rewritten."""
import csv, json, pathlib, subprocess, sys
from datetime import datetime, timezone

BASE = pathlib.Path("runs/serving/LAB-SERVE-001")
RAW, NORM, REP = BASE / "raw", BASE / "normalized", BASE / "report"
for d in (NORM, REP):
    d.mkdir(parents=True, exist_ok=True)

CELLS = [f"mtp_{m}_n{n}" for m in ("on", "off") for n in (1, 2, 4, 8)]
COLS = ["cell", "mtp", "concurrency", "num_prompts", "success_all", "token_sane",
        "token_ratio", "request_throughput", "output_throughput_tok_s",
        "ttft_ms_median", "ttft_ms_p95", "ttft_ms_p99",
        "tpot_ms_median", "tpot_ms_p95", "itl_ms_median", "e2e_ms_median", "e2e_ms_p95",
        "accept_length", "vram_peak_mb", "power_mean_w", "util_mean", "max_concurrent"]

rows = []
for cell in CELLS:
    f = RAW / f"{cell}.normalized.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text()); s = d["upstream_summary"]; v = d["validity"]; g = d.get("gpu", {})
    rows.append({
        "cell": cell, "mtp": "on" if "on" in cell else "off",
        "concurrency": d["params"]["concurrency"], "num_prompts": d["params"]["num_prompts"],
        "success_all": v["success_all"], "token_sane": v["token_accounting_sane"],
        "token_ratio": v["token_accounting_ratio"],
        "request_throughput": round(s.get("request_throughput", 0), 3),
        "output_throughput_tok_s": round(s.get("output_throughput", 0), 1),
        "ttft_ms_median": round(s.get("median_ttft_ms", 0), 1),
        "ttft_ms_p95": round(s.get("p95_ttft_ms", 0), 1), "ttft_ms_p99": round(s.get("p99_ttft_ms", 0), 1),
        "tpot_ms_median": round(s.get("median_tpot_ms", 0), 2), "tpot_ms_p95": round(s.get("p95_tpot_ms", 0), 2),
        "itl_ms_median": round(s.get("median_itl_ms", 0), 2),
        "e2e_ms_median": round(s.get("median_e2e_latency_ms", 0), 1),
        "e2e_ms_p95": round(s.get("p95_e2e_latency_ms", 0), 1),
        "accept_length": round(s.get("accept_length", 0) or 0, 3),
        "vram_peak_mb": g.get("vram_peak_mb"), "power_mean_w": g.get("power_mean_w"),
        "util_mean": g.get("util_mean"), "max_concurrent": s.get("max_concurrent_requests"),
    })

with (NORM / "matrix.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
(NORM / "matrix.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

# MTP advantage per N (derived view)
by = {(r["mtp"], r["concurrency"]): r for r in rows}
adv = []
for n in (1, 2, 4, 8):
    on, off = by.get(("on", n)), by.get(("off", n))
    if on and off:
        adv.append({"concurrency": n,
                    "thr_on": on["output_throughput_tok_s"], "thr_off": off["output_throughput_tok_s"],
                    "thr_mtp_gain_pct": round((on["output_throughput_tok_s"]/off["output_throughput_tok_s"]-1)*100, 1),
                    "tpot_on": on["tpot_ms_median"], "tpot_off": off["tpot_ms_median"],
                    "tpot_mtp_delta_ms": round(on["tpot_ms_median"] - off["tpot_ms_median"], 2)})
(NORM / "mtp_advantage.json").write_text(json.dumps(adv, indent=2), encoding="utf-8")

head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
identity = {
    "experiment_id": "LAB-SERVE-001", "kind": "serving pilot (bounded, shape-discovery)",
    "timestamp_note": "2026-08-10 session (Wave B)", "lab_head": head,
    "model_id": "fable-tc-l1.0-q4", "model_path": "/home/augus/models/merges/fable-tc-l1.0-Q4_K_M.gguf",
    "quantization": "Q4_K_M", "model_family": "qwen36-27b dense (thinking)",
    "tokenizer": "/home/augus/models/fp16/base (EXACT: same Qwen3.6-27B base as the merge)",
    "engine_commit": "lifecycle fork 068764d92 (llama.cpp-master), system_fingerprint b10159-068764d92",
    "server_profile": "deploy-fable (MTP on) / bare fable-tc-l1.0-q4 -ngl 99 (MTP off)",
    "server_flags_common": "-ngl 99 -fa on --ctx-size 8192 --jinja",
    "mtp_on_extra": "--spec-type draft-mtp --spec-draft-n-max 4", "port": 8080,
    "load_generator": "sglang.benchmark.serving (NOTE: sglang.bench_serving is a deprecated shim)",
    "sglang_version": "0.5.16", "backend": "vllm-chat", "endpoint": "/v1/chat/completions",
    "backend_rationale": "vllm-chat and sglang-oai-chat share async_request_openai_chat_completions "
                         "+ /v1/chat/completions; vllm-chat avoids sglang-specific spec-decode "
                         "assumptions (retokenized-ITL path / assert backend==sglang-oai-chat) that a "
                         "llama-server does not satisfy. Proven by a live smoke request.",
    "workload": "random dataset, input_len=1024 output_len=128, apply-chat-template, "
                "ignore_eos=True (FORCED length -> controlled token budget, isolates MTP effect)",
    "sampling": "temperature=0 (bench default), greedy", "seed": 42, "warmup_requests": 2,
    "request_rate": "inf (max-concurrency gated)", "matrix": "concurrency {1,2,4,8} x MTP {on,off}",
    "num_prompts_per_cell": "16/16/32/64 at N=1/2/4/8 (~8 waves each)",
    "arm_order": "within a server config N ran shuffled (4,1,8,2); MTP arm needs a server restart",
    "repetitions": 1, "validity_gate": "LAB-SERVE-QA-001 QUALIFIED (token ratio 1.000, all cells sane)",
}
(BASE / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
print("wrote:", NORM / "matrix.csv", NORM / "matrix.json", NORM / "mtp_advantage.json", BASE / "identity.json")
print(json.dumps(adv, indent=2))
