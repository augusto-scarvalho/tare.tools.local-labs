#!/usr/bin/env python3
"""LAB-CLOSE-002 termination matrix for Fable-Fusion-711."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
import urllib.request
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.model_lifecycle.analysis.benchmark_qa import flag_truncated  # noqa: E402


PROMPTS = (
    "What is the capital of France? Answer in one word.",
    "Write a Python one-liner that returns the square of n. Only the code.",
    "Say hello in exactly three words.",
    "Is 17 prime? Answer 'yes' or 'no' and one short sentence.",
)

ARMS = (
    {"id": "instruct-greedy-512", "think": False, "cap": 512, "temperature": 0.0},
    {"id": "thinking-greedy-512", "think": True, "cap": 512, "temperature": 0.0},
    {"id": "thinking-sampled-512", "think": True, "cap": 512, "temperature": 0.6},
    {"id": "instruct-greedy-2048", "think": False, "cap": 2048, "temperature": 0.0},
    {"id": "thinking-greedy-2048", "think": True, "cap": 2048, "temperature": 0.0},
    {"id": "thinking-sampled-2048", "think": True, "cap": 2048, "temperature": 0.6},
    {"id": "thinking-explicit-stop-2048", "think": True, "cap": 2048,
     "temperature": 0.0, "stop": ["</think>"]},
    {"id": "instruct-ignore-eos-512", "think": False, "cap": 512,
     "temperature": 0.0, "ignore_eos": True},
)


def ask(base_url: str, arm: dict, prompt_index: int, prompt: str) -> dict:
    payload = {
        "model": "fable-fusion-711-termination",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": arm["cap"], "temperature": arm["temperature"],
        "top_k": 1 if arm["temperature"] == 0 else 40,
        "top_p": 1.0 if arm["temperature"] == 0 else 0.95,
        "seed": 42, "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": arm["think"]},
    }
    if arm.get("stop"):
        payload["stop"] = arm["stop"]
    if arm.get("ignore_eos"):
        payload["ignore_eos"] = True
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=900) as response:
        raw = json.loads(response.read().decode("utf-8"))
    choice = raw["choices"][0]
    message = choice["message"]
    timings = raw.get("timings") or {}
    predicted_n = int(timings.get("predicted_n") or raw.get("usage", {}).get("completion_tokens") or 0)
    probe = {"task_id": f"{arm['id']}:{prompt_index}",
             "finish_reason": choice.get("finish_reason"), "answer_tokens": predicted_n}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    explicit_stop = bool(arm.get("stop"))
    truncated = bool(flag_truncated([probe], max_tokens=arm["cap"]))
    natural_stop = choice.get("finish_reason") == "stop" and not explicit_stop and not arm.get("ignore_eos")
    return {
        "task_id": probe["task_id"], "arm": arm["id"], "prompt_index": prompt_index,
        "prompt": prompt, "thinking": arm["think"], "max_tokens": arm["cap"],
        "temperature": arm["temperature"], "stop": arm.get("stop"),
        "ignore_eos": bool(arm.get("ignore_eos")),
        "finish_reason": choice.get("finish_reason"), "predicted_n": predicted_n,
        "truncated": truncated, "natural_stop": natural_stop,
        "completed_content": bool(content.strip()), "reasoning_present": bool(reasoning.strip()),
        "wall_s": round(time.monotonic() - started, 3),
        "content": content, "reasoning_content": reasoning, "raw": raw,
    }


def summarize(rows: list[dict]) -> dict:
    arms = []
    for arm in ARMS:
        group = [row for row in rows if row["arm"] == arm["id"]]
        arms.append({"arm": arm["id"], "n": len(group),
                     "natural_stops": sum(row["natural_stop"] for row in group),
                     "finish_stop": sum(row["finish_reason"] == "stop" for row in group),
                     "finish_length": sum(row["finish_reason"] == "length" for row in group),
                     "completed_content": sum(row["completed_content"] for row in group),
                     "predicted_tokens": [row["predicted_n"] for row in group]})
    instruct = [row for row in rows if row["arm"].startswith("instruct-greedy-")]
    thinking = [row for row in rows if row["arm"].startswith("thinking-greedy-")
                or row["arm"].startswith("thinking-sampled-")]
    repeated_length_prompts = [index for index in range(len(PROMPTS))
                               if all(any(row["prompt_index"] == index and row["truncated"]
                                          for row in thinking if row["max_tokens"] == cap)
                                      for cap in (512, 2048))]
    instruct_safe = len(instruct) == 8 and all(row["natural_stop"] for row in instruct)
    thinking_rate = sum(row["natural_stop"] for row in thinking) / max(1, len(thinking))
    thinking_eligible = (len(thinking) == 16 and thinking_rate >= 0.95
                         and not repeated_length_prompts)
    return {"arms": arms, "instruct_natural_termination_rate":
            sum(row["natural_stop"] for row in instruct) / max(1, len(instruct)),
            "instruct_bounded_safe": instruct_safe,
            "thinking_natural_termination_rate": thinking_rate,
            "thinking_repeated_length_prompt_indices": repeated_length_prompts,
            "thinking_agentic_eligible": thinking_eligible,
            "verdict": ("ELIGIBLE" if thinking_eligible else
                        "DISQUALIFIED_FOR_THINKING_ENABLED_AGENTIC_ROLE")}


def selfcheck() -> None:
    assert len(ARMS) * len(PROMPTS) == 32
    probe = [{"task_id": "x", "finish_reason": "length", "answer_tokens": 12}]
    assert flag_truncated(probe, max_tokens=512) == ["x"]
    print("Fable termination matrix self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8092")
    parser.add_argument("--output-dir", type=pathlib.Path, default=ROOT / "runs" / "close-outs" /
                        "LAB-CLOSE-002-FABLE-TERMINATION-2026-08-22")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipts_path = args.output_dir / "receipts.jsonl"
    if receipts_path.exists():
        raise RuntimeError(f"refusing to overwrite existing receipts: {receipts_path}")
    rows = []
    with receipts_path.open("x", encoding="utf-8", buffering=1) as stream:
        for arm_index, arm in enumerate(ARMS):
            order = list(range(len(PROMPTS)))
            order = order[arm_index % len(order):] + order[:arm_index % len(order)]
            for prompt_index in order:
                row = ask(args.base_url, arm, prompt_index, PROMPTS[prompt_index])
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"{row['arm']} p{prompt_index}: finish={row['finish_reason']} "
                      f"tokens={row['predicted_n']} content={row['completed_content']} "
                      f"wall={row['wall_s']:.1f}s", flush=True)
    report = {"campaign": "LAB-CLOSE-002", "timestamp": datetime.now(timezone.utc).isoformat(),
              "artifact_sha256": "c796c2c011eaa0edf06395ff49cda5bfd4843ad52b86b58a83296dfc33849e4e",
              "engine_commit": "5e7f6271c06b9104862ab799278a1b7f1323a449",
              "qualified": len(rows) == 32, "summary": summarize(rows),
              "receipts_sha256": hashlib.sha256(receipts_path.read_bytes()).hexdigest()}
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
