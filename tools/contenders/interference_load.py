#!/usr/bin/env python3
"""Bounded, stoppable load generators for LAB-OPS-002."""
from __future__ import annotations

import argparse
import mmap
import multiprocessing as mp
import os
import pathlib
import subprocess
import sys
import threading
import time


def stdin_stop(event) -> None:
    try:
        sys.stdin.readline()
    finally:
        event.set()


def cpu_worker(event) -> None:
    value = 1
    while not event.is_set():
        value = (value * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)


def run_cpu(duration: float, workers: int) -> None:
    event = mp.Event()
    children = [mp.Process(target=cpu_worker, args=(event,)) for _ in range(workers)]
    for child in children:
        child.start()
    threading.Thread(target=stdin_stop, args=(event,), daemon=True).start()
    print(f"READY cpu workers={workers}", flush=True)
    event.wait(duration)
    event.set()
    for child in children:
        child.join(timeout=5)
        if child.is_alive():
            child.terminate()


def run_ram(duration: float, gib: int) -> None:
    event = threading.Event()
    size = gib * 1024**3
    region = mmap.mmap(-1, size)
    for offset in range(0, size, 4096):
        region[offset] = (offset // 4096) & 0xFF
    threading.Thread(target=stdin_stop, args=(event,), daemon=True).start()
    print(f"READY ram bytes={size}", flush=True)
    event.wait(duration)
    region.close()


def run_disk(duration: float, source: pathlib.Path) -> None:
    event = threading.Event()
    threading.Thread(target=stdin_stop, args=(event,), daemon=True).start()
    print(f"READY disk source={source}", flush=True)
    deadline = time.monotonic() + duration
    while not event.is_set() and time.monotonic() < deadline:
        proc = subprocess.Popen(["dd", f"if={source}", "of=/dev/null", "bs=8M",
                                 "iflag=direct", "status=none"])
        while proc.poll() is None and not event.wait(0.1):
            pass
        if event.is_set() and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def run_gpu(duration: float) -> None:
    import torch

    event = threading.Event()
    torch.manual_seed(42)
    left = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
    right = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
    result = left @ right
    torch.cuda.synchronize()
    threading.Thread(target=stdin_stop, args=(event,), daemon=True).start()
    allocated = torch.cuda.memory_allocated()
    print(f"READY gpu allocated_bytes={allocated}", flush=True)
    deadline = time.monotonic() + duration
    while not event.is_set() and time.monotonic() < deadline:
        result = left @ right
        torch.cuda.synchronize()
    del result, left, right


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("cpu", "ram", "disk", "gpu"))
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--cpu-workers", type=int, default=12)
    parser.add_argument("--ram-gib", type=int, default=8)
    parser.add_argument("--disk-source", type=pathlib.Path, default=pathlib.Path(
        "/home/augus/models/fable-fusion-711/"
        "Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf"))
    args = parser.parse_args()
    if args.kind == "cpu":
        run_cpu(args.duration, args.cpu_workers)
    elif args.kind == "ram":
        run_ram(args.duration, args.ram_gib)
    elif args.kind == "disk":
        run_disk(args.duration, args.disk_source)
    else:
        run_gpu(args.duration)
    print(f"STOPPED {args.kind}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

