#!/usr/bin/env python3
"""Nonce-controlled prefill/decode sentinel for two GGUFs on one endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path


PARAGRAPH = (
    "Autoregressive inference separates prompt processing from token-by-token "
    "generation. Prompt processing exposes parallel matrix multiplication, while "
    "decoding repeatedly streams model weights and cached state. Measurements must "
    "therefore report first-token latency and steady decode throughput separately. "
)


def stream_chat(base_url: str, prompt: str, max_tokens: int, nonce: int) -> dict:
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": f"Nonce {nonce:08d}.\n\n{prompt}"}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 4242,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    first = None
    pieces: list[str] = []
    prompt_tokens = completion_tokens = 0
    finish_reason = None
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            datum = line[5:].strip()
            if datum == "[DONE]":
                break
            event = json.loads(datum)
            usage = event.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens") or prompt_tokens
            completion_tokens = usage.get("completion_tokens") or completion_tokens
            for choice in event.get("choices") or []:
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                if piece and first is None:
                    first = time.monotonic()
                pieces.append(piece)
    ended = time.monotonic()
    ttft = first - started if first is not None else None
    decode_s = ended - first if first is not None else None
    text = "".join(pieces)
    return {
        "nonce": nonce,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": ttft,
        "total_s": ended - started,
        "prefill_tps": prompt_tokens / ttft if ttft else None,
        "decode_tps": (completion_tokens - 1) / decode_s
        if decode_s and completion_tokens > 1 else None,
        "finish_reason": finish_reason,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
    }


def median(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return statistics.median(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8092")
    parser.add_argument("--label", required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--nonce-base", type=int, default=202608220)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prefill_prompt = "Read the passage and name the two inference phases.\n\n" + PARAGRAPH * 40
    decode_prompt = (
        "Explain why prompt processing and autoregressive decoding have different "
        "performance bottlenecks on a single consumer GPU. Use complete sentences."
    )
    rows: list[dict] = []
    for rep in range(args.reps):
        for kind, prompt, max_tokens in (
            ("prefill", prefill_prompt, 16), ("decode", decode_prompt, 256)
        ):
            nonce = args.nonce_base + rep * 10 + (0 if kind == "prefill" else 1)
            row = stream_chat(args.base_url, prompt, max_tokens, nonce)
            row.update({"label": args.label, "rep": rep, "kind": kind})
            rows.append(row)
            print(
                f"{args.label} {kind} r{rep}: ttft={row['ttft_s']:.3f}s "
                f"prefill={row['prefill_tps']:.2f} t/s decode={row['decode_tps']:.2f} t/s"
            )

    result = {
        "label": args.label,
        "reps": args.reps,
        "nonce_base": args.nonce_base,
        "controls": {"temperature": 0.0, "seed": 4242, "thinking": False},
        "summary": {},
        "rows": rows,
    }
    for kind in ("prefill", "decode"):
        subset = [row for row in rows if row["kind"] == kind]
        result["summary"][kind] = {
            "median_ttft_s": median(subset, "ttft_s"),
            "median_prefill_tps": median(subset, "prefill_tps"),
            "median_decode_tps": median(subset, "decode_tps"),
            "median_total_s": median(subset, "total_s"),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
