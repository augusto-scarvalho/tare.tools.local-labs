#!/usr/bin/env python3
"""LAB-SERVE-001 — thin adapter around `sglang.benchmark.serving` (backend vllm-chat).

Runs ONE serving cell against a running llama-server /v1/chat/completions, captures the raw
upstream JSONL + stdout + GPU telemetry, normalizes the metrics, and runs validity checks. It does
NOT reimplement arrivals / concurrency / streaming parse / percentiles — those stay upstream.

Runs in the sglang venv (has transformers + the bench module):
  /home/augus/sglang-venv/bin/python ops/lab_serve_bench.py --cell ... --outdir ...

The upstream chat request function counts `reasoning_content` toward output tokens (verified in
serving.py) and takes output_len from usage.completion_tokens, so thinking models are handled; the
token-accounting validity check below compares reported vs retokenized output as the §6 gate.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import threading
import time

BENCH = ["-m", "sglang.benchmark.serving"]


def _gpu_sample():
    r = subprocess.run(["nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    try:
        v = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return {"vram_mb": float(v[0]), "util": float(v[1]), "power_w": float(v[2]), "temp_c": float(v[3])}
    except Exception:
        return None


class GpuSampler(threading.Thread):
    def __init__(self, hz=2.0):
        super().__init__(daemon=True); self.stop_flag = False; self.dt = 1.0 / hz; self.samples = []
    def run(self):
        while not self.stop_flag:
            s = _gpu_sample()
            if s: self.samples.append(s)
            time.sleep(self.dt)
    def summary(self):
        if not self.samples: return {}
        return {"vram_peak_mb": max(s["vram_mb"] for s in self.samples),
                "vram_mean_mb": round(sum(s["vram_mb"] for s in self.samples)/len(self.samples), 1),
                "util_mean": round(sum(s["util"] for s in self.samples)/len(self.samples), 1),
                "power_mean_w": round(sum(s["power_w"] for s in self.samples)/len(self.samples), 1),
                "power_peak_w": max(s["power_w"] for s in self.samples),
                "temp_peak_c": max(s["temp_c"] for s in self.samples), "n_samples": len(self.samples)}


def run_cell(a) -> dict:
    outdir = pathlib.Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    jsonl = outdir / f"{a.tag}.jsonl"
    argv = [sys.executable, *BENCH,
            "--backend", "vllm-chat", "--base-url", a.base_url, "--model", a.model,
            "--tokenizer", a.tokenizer, "--dataset-name", "random",
            "--random-input-len", str(a.input_len), "--random-output-len", str(a.output_len),
            "--random-range-ratio", "1.0",           # fixed lengths -> deterministic token budget
            "--num-prompts", str(a.num_prompts), "--max-concurrency", str(a.concurrency),
            "--request-rate", a.request_rate, "--apply-chat-template",
            "--warmup-requests", str(a.warmup), "--seed", "42", "--disable-tqdm",
            "--output-file", str(jsonl), "--output-details"]
    if a.disable_ignore_eos:
        argv.append("--disable-ignore-eos")          # respect EOS (realistic); omit -> forced length
    gpu = GpuSampler(); gpu.start()
    t0 = time.time()
    p = subprocess.run(argv, capture_output=True, text=True)
    wall = time.time() - t0
    gpu.stop_flag = True; gpu.join(timeout=2)
    (outdir / f"{a.tag}.stdout.txt").write_text((p.stdout or "") + "\n===STDERR===\n" + (p.stderr or ""), encoding="utf-8")
    (outdir / f"{a.tag}.argv.txt").write_text(" ".join(argv), encoding="utf-8")

    summary = {}
    if jsonl.exists():
        lines = [l for l in jsonl.read_text().splitlines() if l.strip()]
        if lines:
            summary = json.loads(lines[-1])            # upstream appends one summary dict per run

    # ---- validity checks (do not trust derived metrics if these fail) ----
    total_out = summary.get("total_output_tokens") or summary.get("total_output") or 0
    total_out_re = summary.get("total_output_tokens_retokenized") or summary.get("total_output_retokenized") or 0
    completed = summary.get("completed", 0)
    tok_ratio = (total_out_re / total_out) if total_out else 0.0
    validity = {
        "returncode": p.returncode,
        "completed": completed, "requested": a.num_prompts,
        "success_all": completed == a.num_prompts and p.returncode == 0,
        "total_output_tokens": total_out, "total_output_tokens_retokenized": total_out_re,
        "token_accounting_ratio": round(tok_ratio, 3),
        "token_accounting_sane": (0.9 <= tok_ratio <= 1.1) if total_out else False,
        "ttft_ms_median_positive": (summary.get("median_ttft_ms", 0) or 0) > 0,
    }
    record = {"tag": a.tag, "wall_s": round(wall, 1),
              "params": {"base_url": a.base_url, "model": a.model, "concurrency": a.concurrency,
                         "input_len": a.input_len, "output_len": a.output_len,
                         "num_prompts": a.num_prompts, "request_rate": a.request_rate,
                         "warmup": a.warmup, "disable_ignore_eos": a.disable_ignore_eos},
              "gpu": gpu.summary(), "validity": validity, "upstream_summary": summary}
    (outdir / f"{a.tag}.normalized.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="fable-tc")
    ap.add_argument("--tokenizer", default="/home/augus/models/fp16/base")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--input-len", type=int, default=256)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--request-rate", default="inf")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--disable-ignore-eos", action="store_true")
    a = ap.parse_args()
    rec = run_cell(a)
    v = rec["validity"]
    print(json.dumps({"tag": rec["tag"], "wall_s": rec["wall_s"], "validity": v,
                      "gpu": rec["gpu"],
                      "throughput": rec["upstream_summary"].get("output_throughput"),
                      "median_ttft_ms": rec["upstream_summary"].get("median_ttft_ms"),
                      "median_tpot_ms": rec["upstream_summary"].get("median_tpot_ms")}, indent=2))
    return 0 if v["success_all"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
