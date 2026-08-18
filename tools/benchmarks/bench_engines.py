"""Head-to-head: SGLang vs the llama.cpp fork on gpt-oss-20b, measured IDENTICALLY.

This is NOT a clean single-variable A/B and it does not pretend to be. The two engines
cannot share weights -- SGLang serves gpt-oss in its native MXFP4, the fork serves the
GGUF Q4_K_M -- so the number confounds engine x quantisation. What it DOES control is the
measurement: the same client, the same prompts, the same clock for both. It answers the
deployment question ("what actually serves gpt-oss fastest on this box"), not a mechanism.

Two design choices carry the honesty:

  * PREFILL is timed to the FIRST TOKEN OF ANY KIND, not the first content token. gpt-oss
    is a reasoning model; timing to first *content* would fold its reasoning generation
    into the "prefill" number and flatter whichever engine reasons less. First-any-token
    with a long prompt and a tiny budget is a clean prefill boundary for both.

  * The engines run in SEQUENTIAL blocks, not interleaved -- they cannot coexist (both
    want :8080 and neither fits in VRAM beside the other). Interleaving is the project's
    rule against order/thermal drift; its absence here is a stated limitation, tolerable
    only because engine deltas are expected to dwarf it. If a delta comes back small,
    it does NOT survive this design and must be re-run with per-round server swaps.

    python bench_engines.py --engine sglang   --rounds 6
    python bench_engines.py --engine llamacpp --rounds 6
    python bench_engines.py --compare
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from model_lifecycle.analysis.robust import bootstrap_ci, sign_test_p    # noqa: E402
from model_lifecycle.collectors.host import sample                       # noqa: E402
from model_lifecycle.control_plane.guard import (                        # noqa: E402
    Envelope, precheck, start_watch)
from model_lifecycle.models import MODELS                                # noqa: E402
from model_lifecycle.servers.llama_cpp import (                          # noqa: E402
    LlamaCppAdapter, ServerProfile)

DISTRO = "Ubuntu-24.04"
SGLANG_PY = "/home/augus/sglang-venv/bin/python"
# The MXFP4 HF dir is SGLang-specific (not a GGUF, no block_count in the registry sense),
# so it stays a local constant. The GGUF side comes from the shared registry.
SGLANG_MODEL = "/home/augus/models/gpt-oss-20b-mxfp4"
LOCAL_BIN = "/home/augus/src/llama.cpp-local/build/bin/llama-server"
GGUF = MODELS["gpt-oss-20b-q4"].path
OUT = pathlib.Path(__file__).parent / "runs" / "bench-gptoss"
PORT = 8080

# A long, fixed prompt so prefill dominates the first-token latency. Built deterministically
# from a paragraph repeated to ~1k tokens; the exact text does not matter, only that it is
# identical for both engines and long enough that a handful of reasoning tokens before the
# first emission are negligible against it.
_PARA = ("Memory bandwidth, not arithmetic throughput, is the binding constraint on "
         "single-stream token generation for large language models on consumer GPUs. "
         "Each generated token requires streaming the model weights through the memory "
         "system once, so the achievable rate is set by how fast bytes move, not by how "
         "many multiply-accumulates the cores can retire. ")
PREFILL_PROMPT = ("Read the following passage carefully.\n\n" + _PARA * 24 +
                  "\n\nNow answer in one short sentence: what is the binding constraint?")
DECODE_PROMPT = ("Explain, in about 200 words, why memory bandwidth rather than raw "
                 "compute usually limits token generation on a single consumer GPU.")


def stream_probe(base_url: str, prompt: str, *, max_tokens: int,
                 timeout_s: float = 300.0) -> dict:
    """One streaming completion. Records first-token-of-ANY-kind (prefill boundary),
    first-content-token, total wall, and server-reported token counts."""
    body = json.dumps({
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    t_any = t_content = None
    prompt_n = completion_n = reasoning_chars = 0
    finish = None
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if p == "[DONE]":
                    break
                try:
                    ev = json.loads(p)
                except ValueError:
                    continue
                if u := ev.get("usage"):
                    completion_n = u.get("completion_tokens", completion_n)
                    prompt_n = u.get("prompt_tokens", prompt_n)
                for ch in ev.get("choices") or []:
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    d = ch.get("delta") or {}
                    rc = d.get("reasoning_content") or ""
                    piece = d.get("content") or ""
                    if (rc or piece) and t_any is None:
                        t_any = time.monotonic() - t0       # prefill boundary
                    reasoning_chars += len(rc)
                    if piece and t_content is None:
                        t_content = time.monotonic() - t0
        total = time.monotonic() - t0
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "total_s": time.monotonic() - t0}
    return {"ok": True, "t_first_any": t_any, "t_first_content": t_content,
            "total_s": total, "prompt_n": prompt_n, "completion_n": completion_n,
            "reasoning_chars": reasoning_chars, "finish_reason": finish}


def prefill_tps(pr: dict) -> float | None:
    if not pr.get("ok") or not pr.get("t_first_any") or not pr.get("prompt_n"):
        return None
    return pr["prompt_n"] / pr["t_first_any"]


def decode_tps(pr: dict) -> float | None:
    if not pr.get("ok") or pr.get("t_first_any") is None or not pr.get("completion_n"):
        return None
    window = pr["total_s"] - pr["t_first_any"]
    return pr["completion_n"] / window if window > 0 else None


# --- SGLang: a minimal adapter, same lifecycle shape as LlamaCppAdapter ---------------
class SGLangServer:
    def __init__(self):
        self.proc: subprocess.Popen | None = None

    def argv(self, *, mem_fraction: float, ctx: int) -> list[str]:
        return ["wsl", "-d", DISTRO, "--", SGLANG_PY, "-m", "sglang.launch_server",
                "--model-path", SGLANG_MODEL, "--host", "0.0.0.0", "--port", str(PORT),
                "--mem-fraction-static", str(mem_fraction), "--context-length", str(ctx),
                "--reasoning-parser", "gpt-oss", "--attention-backend", "flashinfer"]

    def start(self, *, mem_fraction: float, ctx: int) -> float | None:
        self.proc = subprocess.Popen(self.argv(mem_fraction=mem_fraction, ctx=ctx),
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace")
        # /health_generate 200s only once the scheduler can actually generate -- the
        # equivalent of waiting for llama.cpp's post-load /health, not mere liveness.
        t0 = time.monotonic()
        url = f"http://127.0.0.1:{PORT}/health_generate"
        while time.monotonic() - t0 < 1800:
            if self.proc.poll() is not None:
                return None
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        return time.monotonic() - t0
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(2.0)
        return None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        # SGLang forks a scheduler + detokenizer + workers; the parent dying does not take
        # them down. Kill the tree by its command signature inside the distro.
        subprocess.run(["wsl", "-d", DISTRO, "--", "pkill", "-9", "-f",
                        "sglang.launch_server"], capture_output=True, timeout=30)
        subprocess.run(["wsl", "-d", DISTRO, "--", "pkill", "-9", "-f", "sglang::"],
                       capture_output=True, timeout=30)


def run_sglang(rounds: int) -> list[dict]:
    env = Envelope()
    if reason := precheck(env):
        print(f"REFUSING: {reason}"); return []
    srv = SGLangServer()
    print("starting SGLang (gpt-oss-20b MXFP4, mem-fraction 0.75) ...", flush=True)
    load = srv.start(mem_fraction=0.75, ctx=8192)
    if load is None:
        print("SGLang never healthy"); srv.stop(); return []
    print(f"  up in {load:.1f}s", flush=True)
    watch = start_watch(env); watch.mark_healthy()
    recs = []
    try:
        for r in range(rounds):
            pf = stream_probe(f"http://127.0.0.1:{PORT}", PREFILL_PROMPT, max_tokens=16)
            dc = stream_probe(f"http://127.0.0.1:{PORT}", DECODE_PROMPT, max_tokens=512)
            watch.observe(sample())
            recs.append({"engine": "sglang", "round": r, "prefill": pf, "decode": dc,
                         "load_seconds": load,
                         "prefill_tps": prefill_tps(pf), "decode_tps": decode_tps(dc)})
            print(f"  r{r}: prefill {prefill_tps(pf)} t/s  decode {decode_tps(dc)} t/s",
                  flush=True)
    finally:
        srv.stop(); time.sleep(15)
    return recs


def run_llamacpp(rounds: int) -> list[dict]:
    env = Envelope()
    if reason := precheck(env):
        print(f"REFUSING: {reason}"); return []
    adapter = LlamaCppAdapter(server_bin=LOCAL_BIN,
                              env={"GGML_CUDA_REGISTER_HOST": "1"})   # pinning on
    profile = ServerProfile(model_path=GGUF, port=PORT, ctx_size=8192,
                            cache_type_k="q8_0", cache_type_v="q8_0",
                            extra_args=("--jinja",))
    print("starting llama.cpp fork (gpt-oss-20b GGUF Q4_K_M, pinning on) ...", flush=True)
    h = adapter.start(profile)
    if not adapter.wait_until_healthy(h, timeout_s=1800):
        print("llama.cpp never healthy"); adapter.stop(h); return []
    print(f"  up in {h.load_seconds:.1f}s", flush=True)
    watch = start_watch(env); watch.mark_healthy()
    recs = []
    try:
        for r in range(rounds):
            pf = stream_probe(h.base_url, PREFILL_PROMPT, max_tokens=16)
            dc = stream_probe(h.base_url, DECODE_PROMPT, max_tokens=512)
            watch.observe(sample())
            recs.append({"engine": "llamacpp", "round": r, "prefill": pf, "decode": dc,
                         "load_seconds": h.load_seconds,
                         "prefill_tps": prefill_tps(pf), "decode_tps": decode_tps(dc)})
            print(f"  r{r}: prefill {prefill_tps(pf)} t/s  decode {decode_tps(dc)} t/s",
                  flush=True)
    finally:
        adapter.stop(h); adapter.force_stop(h); time.sleep(15)
    return recs


def _med(v):
    s = sorted(v); n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def compare():
    def load(engine):
        f = OUT / f"{engine}.json"
        return json.load(open(f)) if f.exists() else []
    sg, lc = load("sglang"), load("llamacpp")
    if not sg or not lc:
        print("need both runs first: --engine sglang, then --engine llamacpp")
        return
    print("gpt-oss-20b — SGLang (MXFP4) vs llama.cpp fork (GGUF Q4, pinned)")
    print("  measurement identical; engine x quant is confounded, by necessity.\n")
    for metric in ("prefill_tps", "decode_tps"):
        s = [r[metric] for r in sg if r.get(metric)]
        l = [r[metric] for r in lc if r.get(metric)]
        if not s or not l:
            print(f"  {metric}: missing data"); continue
        sm, lm = _med(s), _med(l)
        ratio = sm / lm if lm else float("nan")
        print(f"  {metric:12}  sglang {sm:8.1f}   llamacpp {lm:8.1f}   "
              f"sglang/llamacpp {ratio:.2f}x")
    print(f"\n  load: sglang {_med([r['load_seconds'] for r in sg]):.1f}s  "
          f"llamacpp {_med([r['load_seconds'] for r in lc]):.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["sglang", "llamacpp"])
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.compare:
        compare(); return
    recs = run_sglang(a.rounds) if a.engine == "sglang" else run_llamacpp(a.rounds)
    if recs:
        (OUT / f"{a.engine}.json").write_text(json.dumps(recs, indent=2), encoding="utf-8")
        print(f"wrote runs/bench-gptoss/{a.engine}.json ({len(recs)} rounds)")


if __name__ == "__main__":
    main()
