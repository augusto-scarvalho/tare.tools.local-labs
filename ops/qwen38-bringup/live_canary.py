#!/usr/bin/env python3
"""Endpoint canary for the revision-pinned HauhauCS Qwen3.8 candidate."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


BASE = "http://127.0.0.1:8080"


def post(path: str, payload: dict, timeout: int = 600) -> dict:
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def token_count(text: str) -> int:
    return len(post("/tokenize", {"content": text, "add_special": False})["tokens"])


def chat(prompt: str, max_tokens: int = 128) -> tuple[dict, float]:
    started = time.monotonic()
    response = post(
        "/v1/chat/completions",
        {
            "model": "qwen38-hauhaucs-aggressive",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_k": 1,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return response, time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filler-lines", type=int, default=1950)
    args = parser.parse_args()
    short, short_wall = chat("Responda somente com o resultado inteiro de 17 * 19.", 32)
    short_text = short["choices"][0]["message"].get("content") or ""

    needle = "A SENHA DO CANARIO E JADE-7319."
    filler = "Registro técnico neutro: o sistema permanece estável.\n"
    # 12 tokenizer tokens per filler line on this exact model; 1,950 lines plus
    # instructions stays near 23.4k, leaving ample generation headroom in a 32k slot.
    repeats = args.filler_lines
    prefix = filler * int(repeats * 0.70)
    suffix = filler * (repeats - int(repeats * 0.70))
    long_prompt = (
        "Leia o registro e, ao final, responda somente com a senha, sem explicação.\n"
        + prefix + needle + "\n" + suffix
        + "Fim do registro. Qual é a senha do canário?"
    )
    long_tokens = token_count(long_prompt)
    long_response, long_wall = chat(long_prompt, 48)
    long_text = long_response["choices"][0]["message"].get("content") or ""

    result = {
        "short": {
            "expected": "323",
            "answer": short_text.strip(),
            "pass": short_text.strip() == "323",
            "wall_s": round(short_wall, 3),
            "timings": short.get("timings", {}),
        },
        "long_context": {
            "prompt_tokens_via_tokenize": long_tokens,
            "needle": "JADE-7319",
            "answer": long_text.strip(),
            "pass": "JADE-7319" in long_text,
            "wall_s": round(long_wall, 3),
            "timings": long_response.get("timings", {}),
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["short"]["pass"] and result["long_context"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
