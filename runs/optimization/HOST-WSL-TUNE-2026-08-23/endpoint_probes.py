#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import urllib.request
from datetime import datetime, timezone


PROBES = (
    "Complete deterministically and briefly: The sum of 17 and 25 is",
    "Continue this delimiter-separated sequence with one item: alpha|beta|gamma|",
    'Complete this Python function correctly:\ndef add(a, b):\n    """Return their sum."""\n',
)


def completion(base_url: str, prompt: str) -> dict:
    payload = {
        "prompt": prompt,
        "n_predict": 64,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 0,
        "cache_prompt": False,
        "stream": False,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.load(response)
    content = result.get("content", "")
    return {
        "content": content,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "tokens_predicted": int(result.get("tokens_predicted") or 0),
        "timings": result.get("timings") or {},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    rows = [completion(args.base_url, prompt) for prompt in PROBES]
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "qualified": all(row["tokens_predicted"] > 0 for row in rows),
        "hashes": [row["content_sha256"] for row in rows],
        "probes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"qualified": report["qualified"], "hashes": report["hashes"]}, indent=2))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
