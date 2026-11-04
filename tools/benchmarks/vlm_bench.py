"""VLM performance benchmark -- fills the M0 gap (M0 only had single-request latency-feel on 2
images, no reps, no stats, and never separated the vision-encode cost from decode).

Two metrics that actually decide daily-driver + inform the M-A §71 levers:
  * visual TTFT   -- time to first token WITH an image (= image-encode + prefill + 1 decode).
  * text TTFT     -- same prompt, NO image. The DELTA (visual - text) ~= the image-encode cost,
                     the vision-specific tax the §71 notes want measured separately from decode.
  * decode t/s    -- steady-state generation speed = (gen_tokens - 1) / (t_last - t_first).

Measured by STREAMING (SSE) so we timestamp the first and last token directly -- no dependence on
the server's timings JSON. N reps, warmup discarded, unique nonce per rep (cache_prompt=False) so
prompt-cache reuse can't inflate TTFT (A4 discipline). Serve a model first, then:
    python lmctl.py serve qwen3-vl-8b
    python vlm_bench.py --tag qwen3-vl-8b
    python lmctl.py stop
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib
import statistics as st
import time
import urllib.request

FIX = pathlib.Path(__file__).parent / "runs" / "m-a-align"
OUT = pathlib.Path(__file__).parent / "runs" / "m-a-perf"
IMAGE = FIX / "bank_login.png"   # a realistic mid-size UI (460x420) -- typical agent screenshot
PROMPT = ("Describe this screen in thorough detail: every visible element, its layout, colors, "
          "and apparent purpose. Be comprehensive and specific.")


def _b64(p: pathlib.Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def one_rep(port: int, with_image: bool, max_tokens: int, nonce: str) -> dict:
    content: list = [{"type": "text", "text": PROMPT + f"  (ref {nonce})"}]
    if with_image:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_b64(IMAGE)}"}})
    payload = {"model": "vlm", "temperature": 0, "max_tokens": max_tokens, "stream": True,
               "stream_options": {"include_usage": True}, "cache_prompt": False,
               "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    t_first = t_last = None
    n_chunks = 0
    gen_tokens = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                gen_tokens = obj["usage"].get("completion_tokens")
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or delta.get("reasoning_content") or ""
            if piece:
                now = time.monotonic()
                if t_first is None:
                    t_first = now
                t_last = now
                n_chunks += 1
    ttft = (t_first - t0) if t_first else float("nan")
    ntok = gen_tokens if gen_tokens else n_chunks
    decode_span = (t_last - t_first) if (t_first and t_last and t_last > t_first) else float("nan")
    tps = (ntok - 1) / decode_span if (decode_span == decode_span and ntok and ntok > 1) else float("nan")
    return {"ttft_s": ttft, "decode_tps": tps, "gen_tokens": ntok,
            "total_s": time.monotonic() - t0}


def summ(xs: list[float]) -> str:
    xs = [x for x in xs if x == x]  # drop NaN
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return f"{xs[0]:.2f}"
    m, sd = st.mean(xs), st.pstdev(xs)
    return f"{m:.2f} +/- {sd:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=200)
    a = ap.parse_args()

    print(f"\n=== VLM perf bench :: {a.tag} (reps={a.reps}, max_tokens={a.max_tokens}) ===")
    results = {"tag": a.tag, "reps": a.reps, "max_tokens": a.max_tokens, "runs": {}}
    for mode, with_image in (("image", True), ("text", False)):
        reps = []
        # warmup (rep -1), discarded
        one_rep(a.port, with_image, a.max_tokens, nonce="warm")
        for i in range(a.reps):
            reps.append(one_rep(a.port, with_image, a.max_tokens, nonce=f"{a.tag}-{mode}-{i}-{time.monotonic():.4f}"))
        results["runs"][mode] = reps
        ttfts = [r["ttft_s"] for r in reps]
        tpss = [r["decode_tps"] for r in reps]
        toks = [r["gen_tokens"] for r in reps]
        print(f"  {mode:5}:  TTFT {summ(ttfts):>16} s   decode {summ(tpss):>16} t/s   "
              f"(gen~{int(st.mean([t for t in toks if t])) if any(toks) else 0} tok)")

    # image-encode tax = visual TTFT - text TTFT (means)
    vi = [r["ttft_s"] for r in results["runs"]["image"] if r["ttft_s"] == r["ttft_s"]]
    tx = [r["ttft_s"] for r in results["runs"]["text"] if r["ttft_s"] == r["ttft_s"]]
    if vi and tx:
        tax = st.mean(vi) - st.mean(tx)
        results["image_encode_tax_s"] = round(tax, 3)
        print(f"  >> image-encode tax (visual TTFT - text TTFT) = {tax*1000:.0f} ms")

    OUT.mkdir(parents=True, exist_ok=True)
    outf = OUT / f"PERF_{a.tag}.json"
    outf.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  raw -> {outf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
