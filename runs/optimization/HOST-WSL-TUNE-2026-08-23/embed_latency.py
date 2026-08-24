#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
import urllib.request
from datetime import datetime, timezone


def request(url: str, text: str) -> tuple[list[float], float]:
    payload = json.dumps({"input": [text], "model": "nomic-embed-text-v1.5"}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.load(response)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return result["data"][0]["embedding"], elapsed_ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--reps", type=int, default=7)
    args = parser.parse_args()
    url = "http://127.0.0.1:8081/v1/embeddings"
    text = ("search_document: Stability and throughput validation for a local inference server. "
            "The fixed input is repeated to make tokenization and output deterministic. ") * 8
    request(url, text)
    vectors = []
    latencies = []
    for _ in range(args.reps):
        vector, elapsed_ms = request(url, text)
        vectors.append(vector)
        latencies.append(elapsed_ms)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reps": args.reps,
        "dimension": len(vectors[0]),
        "latency_ms": latencies,
        "latency_ms_median": statistics.median(latencies),
        "reference_vector": vectors[0],
        "all_vectors_identical": all(vector == vectors[0] for vector in vectors),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "dimension", "latency_ms_median", "all_vectors_identical"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
