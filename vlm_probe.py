"""Send an image + prompt to a llama-server vision endpoint and print the reply.

The M0 accept probe. llama-server speaks the OpenAI chat schema with image content as a
base64 data URI, so no SDK is needed -- just urllib.

    python vlm_test.py error_dialog.png "Transcribe every line of text in this dialog."
    python vlm_test.py ui_mockup.png "Describe this UI: every field, button, and link." --port 8092
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib
import time
import urllib.request


def ask(port: int, image: pathlib.Path, prompt: str, max_tokens: int) -> tuple[str, float]:
    b64 = base64.b64encode(image.read_bytes()).decode()
    mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
    payload = {
        "model": "vlm",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=180) as r:
        body = json.load(r)
    ch = body["choices"][0]
    msg = ch["message"]
    text = msg.get("content") or ""
    # Thinking VLMs (Gemma-4) spend the budget on reasoning_content FIRST, then emit
    # content. If max_tokens is too low the answer never leaves reasoning -> content empty.
    # Surface that instead of printing nothing, and flag a truncated think.
    if not text.strip():
        rc = msg.get("reasoning_content") or ""
        if rc.strip():
            fr = ch.get("finish_reason")
            text = (f"[content empty, finish_reason={fr}; showing reasoning_content — "
                    f"raise --max-tokens]\n{rc}")
    return text, time.monotonic() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=pathlib.Path)
    ap.add_argument("prompt")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--max-tokens", type=int, default=1024)  # headroom for thinking VLMs
    a = ap.parse_args()
    text, dt = ask(a.port, a.image, a.prompt, a.max_tokens)
    print(f"--- {a.image.name}  ({dt:.1f}s) ---")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
