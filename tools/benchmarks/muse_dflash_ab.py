#!/usr/bin/env python3
"""Paired deterministic Muse Glimmer no-spec/DFlash throughput qualification.

Run inside WSL. Each arm gets a fresh llama-server process, the same uncached prompt,
fixed greedy sampling, and complete timing/content receipts. This keeps the deployed
fork and service unit outside the experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import signal
import statistics
import subprocess
import tempfile
import time
import urllib.request
from datetime import datetime, timezone

BIN = "/home/augus/src/slop.cpp-muse-glimmer/build/bin/llama-server"
MODEL = ("/home/augus/models/muse-glimmer-30b/meta-70bf1b61/"
         "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf")
DRAFT = ("/home/augus/models/muse-glimmer-30b/meta-70bf1b61/"
         "dflash-Muse-Glimmer-30B-Q4_K_M.gguf")
PROMPT = (
    "Explain how a red-black tree repairs an insertion that creates two consecutive "
    "red nodes. Cover the uncle-red recolor case, triangle rotation, line rotation, "
    "root handling, and why every case preserves black height. Use several precise "
    "paragraphs and finish with a compact invariant checklist.\n\nAnswer:"
)


def wait_health(port: int, timeout_s: float = 120.0) -> bool:
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3):
                return True
        except Exception:  # noqa: BLE001 - readiness retry
            time.sleep(1)
    return False


def complete(port: int, prompt: str, n_predict: int) -> dict:
    payload = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
               "top_k": 1, "seed": 42, "cache_prompt": False}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as response:
        raw = json.loads(response.read().decode("utf-8"))
    content = raw.get("content") or ""
    timings = raw.get("timings") or {}
    return {
        "wall_s": time.monotonic() - started,
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "content": content,
        "timings": timings,
    }


def gpu_snapshot() -> dict:
    query = ("memory.total,memory.used,memory.free,temperature.gpu,power.draw")
    proc = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    values = [part.strip() for part in proc.stdout.strip().split(",")]
    keys = ("memory_total_mib", "memory_used_mib", "memory_free_mib", "temperature_c",
            "power_w")
    return dict(zip(keys, values, strict=True))


def run_arm(name: str, nmax: int | None, port: int, reps: int, n_predict: int) -> dict:
    argv = [BIN, "-m", MODEL, "--alias", f"muse-{name}", "--host", "127.0.0.1",
            "--port", str(port), "--ctx-size", "32768", "-np", "1", "--gpu-layers",
            "all", "--flash-attn", "on", "--jinja", "--no-mmproj"]
    if nmax is not None:
        argv += ["-md", DRAFT, "-ngld", "all", "--spec-type", "draft-dflash",
                 "--spec-draft-n-max", str(nmax)]
    log_path = pathlib.Path(tempfile.mkstemp(prefix=f"muse-{name}-", suffix=".log")[1])
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, preexec_fn=os.setsid)
    try:
        if not wait_health(port):
            raise RuntimeError(f"{name} never became healthy: {log_path.read_text()[-3000:]}")
        residency = gpu_snapshot()
        complete(port, "Warm up. Reply with one sentence.", 64)
        rows = [complete(port, PROMPT, n_predict) for _ in range(reps)]
        tps = [float(row["timings"]["predicted_per_second"]) for row in rows]
        accepts = []
        for row in rows:
            draft_n = row["timings"].get("draft_n")
            accepted = row["timings"].get("draft_n_accepted")
            if draft_n and accepted is not None:
                accepts.append(accepted / draft_n)
        return {
            "name": name,
            "nmax": nmax,
            "argv": argv,
            "residency": residency,
            "rows": rows,
            "decode_tps_mean": statistics.mean(tps),
            "decode_tps_stdev": statistics.stdev(tps) if len(tps) > 1 else 0.0,
            "acceptance_mean": statistics.mean(accepts) if accepts else None,
            "hash_stable": len({row["sha256"] for row in rows}) == 1,
            "log_path": str(log_path),
        }
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001 - cleanup fallback
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        time.sleep(2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--n-predict", type=int, default=384)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    report = {
        "campaign": "LAB-MUSE-001-dflash-ab",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine_commit": "d775b8967a46d8beb110d444aa3b8938179e0dd8",
        "model_sha256": "4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e",
        "draft_sha256": "b2e808bf656086fe86bd0d0bd990f01d33e377537a07c02d45371517c8b264ef",
        "sampling": {"temperature": 0.0, "top_k": 1, "seed": 42,
                     "cache_prompt": False, "n_predict": args.n_predict},
        "arms": [],
    }
    for name, nmax in (("nospec", None), ("dflash-n4", 4), ("dflash-n8", 8),
                       ("dflash-n15", 15)):
        print(f"=== {name} ===", flush=True)
        arm = run_arm(name, nmax, args.port, args.reps, args.n_predict)
        report["arms"].append(arm)
        print(f"tps={arm['decode_tps_mean']:.2f} accept={arm['acceptance_mean']} "
              f"stable={arm['hash_stable']}", flush=True)
    base = report["arms"][0]
    base_hash = base["rows"][0]["sha256"]
    for arm in report["arms"]:
        arm["speedup_vs_nospec"] = arm["decode_tps_mean"] / base["decode_tps_mean"]
        arm["byte_equal_to_nospec"] = all(row["sha256"] == base_hash for row in arm["rows"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
