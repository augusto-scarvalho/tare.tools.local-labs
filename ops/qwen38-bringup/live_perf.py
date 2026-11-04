#!/usr/bin/env python3
"""Small repeated decode check against the active OpenAI-compatible endpoint."""
from __future__ import annotations

import json
import statistics
import time
import urllib.request


URL = "http://127.0.0.1:8080/v1/chat/completions"
PROMPT = (
    "Write a Python class TaskScheduler with methods add_task(name, priority, deps), "
    "remove_task(name), run_order(), and detect_cycle(). Use type hints. Output only code."
)


def request() -> dict:
    payload = {
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 256,
        "temperature": 0.0,
        "top_k": 1,
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as response:
        data = json.load(response)
    data["client_wall_s"] = time.monotonic() - started
    return data


def main() -> int:
    request()  # unscored warmup
    rows = []
    for rep in range(1, 6):
        result = request()
        timings = result.get("timings", {})
        rows.append({
            "rep": rep,
            "predicted_n": timings.get("predicted_n"),
            "predicted_tps": timings.get("predicted_per_second"),
            "draft_n": timings.get("draft_n"),
            "draft_n_accepted": timings.get("draft_n_accepted"),
            "wall_s": round(result["client_wall_s"], 3),
        })
    rates = [row["predicted_tps"] for row in rows]
    drafted = sum(row["draft_n"] or 0 for row in rows)
    accepted = sum(row["draft_n_accepted"] or 0 for row in rows)
    print(json.dumps({
        "rows": rows,
        "median_predicted_tps": statistics.median(rates),
        "mean_predicted_tps": statistics.mean(rates),
        "mtp_drafted": drafted,
        "mtp_accepted": accepted,
        "mtp_acceptance": accepted / drafted if drafted else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
