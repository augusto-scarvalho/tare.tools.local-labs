#!/usr/bin/env python3
"""LAB-SERVE-001b — paired-block MTP replication orchestrator.

Statistical unit = an independent SERVER-LEVEL block (one server start). For each rep and each MTP
arm (order alternated per a seeded paired schedule), it: restarts the server with a PINNED topology,
captures exact argv + /props + /slots, runs the N=1 and N=4 forced-length bench cells, probes MTP
acceptance via the native /completion timings (draft_n/draft_n_accepted) on the ON arm, stops the
server, and cools down. Runs on Windows python; shells lmctl (serve/stop) + wsl (bench + probe).

Reusable for dense (fable-tc) and MoE (qwen36-35b-mtp) via --serve-target/--base-extra.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time

WSL = ["wsl.exe", "-d", "Ubuntu-24.04", "--", "bash", "-lc"]
MTP_EXTRA = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"]
# Seeded paired-block schedule (alternate MTP order to decorrelate arm from time/thermal drift).
SCHEDULE = {1: ["on", "off"], 2: ["off", "on"], 3: ["off", "on"], 4: ["on", "off"], 5: ["on", "off"]}


def sh(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


def wsl(cmd):
    return sh(WSL + [cmd])


def serve(target, port, extra):
    r = sh([sys.executable, "lmctl.py", "serve", target, "--port", str(port), "--", *extra], timeout=300)
    argv = ""
    for line in (r.stdout + r.stderr).splitlines():
        if "argv:" in line:
            argv = line.split("argv:", 1)[1].strip()
    up = "UP in" in r.stdout
    return up, argv, r.stdout + r.stderr


def stop(port):
    sh([sys.executable, "lmctl.py", "stop", "--port", str(port)])


def topology(port):
    p = wsl(f"curl -s http://127.0.0.1:{port}/props")
    s = wsl(f"curl -s http://127.0.0.1:{port}/slots")
    try:
        props = json.loads(p.stdout); slots = json.loads(s.stdout)
        return {"total_slots": props.get("total_slots"), "n_ctx": props.get("n_ctx"),
                "default_gen_n_ctx": (props.get("default_generation_settings") or {}).get("n_ctx"),
                "num_slots": len(slots), "slot0_n_ctx": slots[0].get("n_ctx") if slots else None}
    except Exception as e:
        return {"error": str(e), "props_raw": p.stdout[:200], "slots_raw": s.stdout[:200]}


def accept_probe(port, k=3):
    """Non-invasive MTP acceptance via native /completion timings (draft_n/draft_n_accepted)."""
    accs = []
    prompt = "Summarize the theory of continuous batching in LLM inference servers in detail. " * 12
    payload = json.dumps({"prompt": prompt, "n_predict": 128, "temperature": 0})
    for _ in range(k):
        r = wsl(f"curl -s http://127.0.0.1:{port}/completion -H 'Content-Type: application/json' -d '{payload}'")
        try:
            t = json.loads(r.stdout).get("timings", {})
            dn, da = t.get("draft_n"), t.get("draft_n_accepted")
            if dn:
                accs.append({"draft_n": dn, "draft_n_accepted": da, "accept_rate": round(da / dn, 3)})
        except Exception:
            pass
    return accs


def bench_cell(tag, outdir, tokenizer, model, n, num_prompts):
    cmd = (f"cd /mnt/c/projects/local-model-lifecycle && /home/augus/sglang-venv/bin/python "
           f"ops/lab_serve_bench.py --tag {tag} --outdir {outdir} --model {model} "
           f"--tokenizer {tokenizer} --concurrency {n} --input-len 1024 --output-len 128 "
           f"--num-prompts {num_prompts} --warmup 2")
    r = wsl(cmd)
    f = pathlib.Path(outdir) / f"{tag}.normalized.json"
    dbg = pathlib.Path(outdir) / f"{tag}.debug.txt"
    dbg.parent.mkdir(parents=True, exist_ok=True)
    dbg.write_text(f"CMD: {cmd}\nRC: {r.returncode}\n---STDOUT---\n{r.stdout}\n---STDERR---\n{r.stderr}", encoding="utf-8")
    return json.loads(f.read_text()) if f.exists() else {"error": "no normalized.json", "stdout": r.stdout[-300:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve-target", required=True)
    ap.add_argument("--model", default="m")                     # OpenAI 'model' field (any)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--base-extra", required=True, help="space-sep extra serve flags (both arms)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--cooldown", type=int, default=10)
    a = ap.parse_args()
    outdir = pathlib.Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    base_extra = a.base_extra.split()
    blocks, meta = [], []

    for rep in range(1, a.reps + 1):
        for arm in SCHEDULE.get(rep, ["on", "off"]):
            extra = base_extra + (MTP_EXTRA if arm == "on" else [])
            stop(a.port); time.sleep(2)
            up, argv, _ = serve(a.serve_target, a.port, extra)
            block = {"rep": rep, "arm": arm, "up": up, "argv": argv,
                     "topology": topology(a.port) if up else None,
                     "acceptance": accept_probe(a.port) if (up and arm == "on") else None,
                     "cells": {}}
            if up:
                for n, np_ in ((1, 16), (4, 32)):
                    tag = f"rep{rep}_{arm}_n{n}"
                    rec = bench_cell(tag, outdir.as_posix(), a.tokenizer, a.model, n, np_)  # POSIX for WSL
                    s = rec.get("upstream_summary", {}); v = rec.get("validity", {})
                    to, tr = v.get("total_output_tokens"), v.get("total_output_tokens_retokenized")
                    block["cells"][f"n{n}"] = {
                        "tag": tag, "success_all": v.get("success_all"),
                        "token_ratio": v.get("token_accounting_ratio"),
                        "total_output_tokens": to, "total_output_tokens_retokenized": tr,
                        "token_exact": (to is not None and to == tr),
                        "output_throughput": round(s.get("output_throughput", 0), 2),
                        "ttft_median_ms": round(s.get("median_ttft_ms", 0), 1),
                        "ttft_p95_ms": round(s.get("p95_ttft_ms", 0), 1),
                        "tpot_median_ms": round(s.get("median_tpot_ms", 0), 2),
                        "tpot_p95_ms": round(s.get("p95_tpot_ms", 0), 2),
                        "itl_median_ms": round(s.get("median_itl_ms", 0), 3),
                        "e2e_median_ms": round(s.get("median_e2e_latency_ms", 0), 1),
                        "e2e_p95_ms": round(s.get("p95_e2e_latency_ms", 0), 1),
                        "vram_peak_mb": rec.get("gpu", {}).get("vram_peak_mb"),
                        "power_mean_w": rec.get("gpu", {}).get("power_mean_w"),
                        "power_peak_w": rec.get("gpu", {}).get("power_peak_w"),
                    }
            blocks.append(block)
            meta.append({"rep": rep, "arm": arm, "argv": argv})
            (outdir / "blocks.json").write_text(json.dumps(blocks, indent=2), encoding="utf-8")
            print(f"rep{rep} {arm}: up={up} cells={list(block['cells'].keys())}", flush=True)
            stop(a.port); time.sleep(a.cooldown)

    (outdir / "schedule.json").write_text(json.dumps({"schedule": SCHEDULE, "arm_order": meta}, indent=2))
    print("DONE. blocks ->", outdir / "blocks.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
