#!/usr/bin/env python3
"""LAB-SERVE-001c — open-loop serving cell (realistic traffic) around `sglang.benchmark.serving`.

Differs from lab_serve_bench.py (001/001b, closed-loop forced-length) in exactly the ways 001c needs:
  * --dataset-name custom --dataset-path <pinned workload>  (real prompts, per-request output cap)
  * NO --apply-chat-template  (server templates once via --jinja; avoids double-templating)
  * NO --max-concurrency cap  (load is set by --request-rate; the server's 4 slots queue the excess,
    so queueing is observable — the whole point of 001c vs the closed-loop 001/001b)
  * --disable-ignore-eos      (natural EOS; output_len is a per-request MAX cap, not forced)
  * --request-rate <finite>   (Poisson arrivals; --seed fixes both arrivals AND prompt shuffle, so
    two arms with the same seed/dataset/rate/num-prompts get the IDENTICAL arrival schedule — §14)
  * --output-details          (per-request input_lens/output_lens/ttfts/itls/errors for stratification)

Capture only; pairing + class/length stratification + fairness live in lab_serve_openloop_analyze.py.
Runs in the sglang venv.
"""
import argparse, json, pathlib, subprocess, sys, threading, time

BENCH = ["-m", "sglang.benchmark.serving"]


def _gpu_sample():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
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
                "vram_mean_mb": round(sum(s["vram_mb"] for s in self.samples) / len(self.samples), 1),
                "util_mean": round(sum(s["util"] for s in self.samples) / len(self.samples), 1),
                "power_mean_w": round(sum(s["power_w"] for s in self.samples) / len(self.samples), 1),
                "power_peak_w": max(s["power_w"] for s in self.samples),
                "temp_peak_c": max(s["temp_c"] for s in self.samples), "n_samples": len(self.samples)}


def run_cell(a) -> dict:
    outdir = pathlib.Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    jsonl = outdir / f"{a.tag}.jsonl"
    argv = [sys.executable, *BENCH,
            "--backend", "vllm-chat", "--base-url", a.base_url, "--model", a.model,
            "--tokenizer", a.tokenizer,
            "--dataset-name", "custom", "--dataset-path", a.dataset_path,
            "--num-prompts", str(a.num_prompts), "--request-rate", str(a.request_rate),
            "--seed", str(a.seed), "--disable-ignore-eos", "--disable-tqdm",
            "--warmup-requests", str(a.warmup),
            "--output-file", str(jsonl), "--output-details"]
    # NOTE: deliberately NO --max-concurrency (open-loop) and NO --apply-chat-template (server templates).
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
            summary = json.loads(lines[-1])

    n_err = sum(1 for e in (summary.get("errors") or []) if e)
    total_out = summary.get("total_output_tokens") or summary.get("total_output") or 0
    total_out_re = summary.get("total_output_tokens_retokenized") or summary.get("total_output_retokenized") or 0
    completed = summary.get("completed", 0)
    tok_ratio = (total_out_re / total_out) if total_out else 0.0
    validity = {
        "returncode": p.returncode, "completed": completed, "requested": a.num_prompts,
        "errors": n_err,
        "success_all": completed == a.num_prompts and p.returncode == 0 and n_err == 0,
        "total_output_tokens": total_out, "total_output_tokens_retokenized": total_out_re,
        "token_accounting_ratio": round(tok_ratio, 3),
        "token_accounting_sane": (0.9 <= tok_ratio <= 1.1) if total_out else False,
    }
    record = {"tag": a.tag, "wall_s": round(wall, 1),
              "params": {"base_url": a.base_url, "model": a.model, "dataset_path": a.dataset_path,
                         "num_prompts": a.num_prompts, "request_rate": a.request_rate, "seed": a.seed,
                         "warmup": a.warmup, "open_loop": True, "natural_eos": True},
              "gpu": gpu.summary(), "validity": validity,
              # keep the full per-request arrays for the analyzer (drop generated_texts to save space)
              "details": {k: summary.get(k) for k in ("input_lens", "output_lens", "ttfts", "itls", "errors")},
              "upstream_summary": {k: v for k, v in summary.items()
                                   if k not in ("input_lens", "output_lens", "ttfts", "itls",
                                                "errors", "generated_texts")}}
    (outdir / f"{a.tag}.normalized.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="m")
    ap.add_argument("--tokenizer", default="/home/augus/models/fp16/base")
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--num-prompts", type=int, default=60)
    ap.add_argument("--request-rate", default="0.2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=2)
    a = ap.parse_args()
    rec = run_cell(a)
    v = rec["validity"]; s = rec["upstream_summary"]
    print(json.dumps({"tag": rec["tag"], "wall_s": rec["wall_s"], "validity": v,
                      "completed_reqs_per_s": (v["completed"] / rec["wall_s"]) if rec["wall_s"] else None,
                      "request_throughput": s.get("request_throughput"),
                      "output_throughput": s.get("output_throughput"),
                      "median_ttft_ms": s.get("median_ttft_ms"), "p95_ttft_ms": s.get("p95_ttft_ms"),
                      "median_tpot_ms": s.get("median_tpot_ms"),
                      "median_e2e_ms": s.get("median_e2e_latency_ms"),
                      "gpu": rec["gpu"]}, indent=2))
    return 0 if v["completed"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
