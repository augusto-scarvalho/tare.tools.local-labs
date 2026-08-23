"""LAB-ENGINE-001/002: same-snapshot BF16 comparison on the local RTX 3090.

Run one engine at a time. The frozen design is in
runs/engines/LAB-ENGINE-001-002-2026-08-22/DECISION_PACKET.md.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request


DISTRO = "Ubuntu-24.04"
PORT = 8092
MODEL_DIR = "/home/augus/models/engine-parity-qwen3-4b"
GGUF = f"{MODEL_DIR}/Qwen3-4B-BF16.gguf"
LLAMA_SERVER = "/home/augus/src/slop.cpp-local/build/bin/llama-server"
SGLANG_PY = "/home/augus/sglang-venv/bin/python"
VLLM_PY = "/home/augus/vllm-venv/bin/python"
OUT = pathlib.Path(__file__).resolve().parents[2] / "runs" / "engines" / \
    "LAB-ENGINE-001-002-2026-08-22"

PARA = (
    "Memory bandwidth is often the limiting resource for autoregressive generation on a consumer "
    "GPU. Each new token requires reading model weights and cached state, while prompt processing "
    "can exploit more parallel matrix operations. Controlled measurements must separate startup, "
    "prompt processing, first-token latency, and steady decode throughput. "
)
PREFILL_PROMPT = (
    "Read this repeated technical passage and answer with one sentence naming the primary resource.\n\n"
    + PARA * 30
)
DECODE_PROMPT = (
    "Write a concise technical explanation of why memory bandwidth can limit autoregressive token "
    "generation on a single consumer GPU. Use complete sentences and no bullet list."
)


def _wsl(*args: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(["wsl", "-d", DISTRO, "--", *args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout)


def commands(engine: str) -> tuple[list[str], str]:
    if engine == "llamacpp":
        inner = [LLAMA_SERVER, "-m", GGUF, "--host", "0.0.0.0", "--port", str(PORT),
                 "-ngl", "999", "--ctx-size", "8192", "--parallel", "1",
                 "--cache-type-k", "f16", "--cache-type-v", "f16", "--jinja",
                 "--seed", "4242", "-fa", "on"]
        return ["wsl", "-d", DISTRO, "--", *inner], "llama-server"
    if engine == "sglang":
        inner = [SGLANG_PY, "-m", "sglang.launch_server", "--model-path", MODEL_DIR,
                 "--host", "0.0.0.0", "--port", str(PORT), "--context-length", "8192",
                 "--mem-fraction-static", "0.70", "--dtype", "bfloat16"]
        return ["wsl", "-d", DISTRO, "--", *inner], "sglang.launch_server"
    if engine == "vllm":
        inner = [VLLM_PY, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL_DIR,
                 "--host", "0.0.0.0", "--port", str(PORT), "--max-model-len", "8192",
                 "--gpu-memory-utilization", "0.70", "--dtype", "bfloat16",
                 "--served-model-name", "local", "--seed", "4242"]
        return ["wsl", "-d", DISTRO, "--", "env", "VLLM_WSL2_ENABLE_PIN_MEMORY=1",
                *inner], "vllm.entrypoints.openai.api_server"
    raise ValueError(engine)


def kill_engine(signature: str) -> None:
    # The launchers fork workers which can survive their immediate parent.
    _wsl("pkill", "-9", "-f", signature, timeout=30)


def wait_ready(proc: subprocess.Popen, timeout_s: float = 900.0) -> float | None:
    started = time.monotonic()
    url = f"http://127.0.0.1:{PORT}/v1/models"
    while time.monotonic() - started < timeout_s:
        if proc.poll() is not None:
            return None
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return time.monotonic() - started
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(2)
    return None


def query_vram() -> int | None:
    p = _wsl("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             timeout=15)
    try:
        return int(p.stdout.strip().splitlines()[0]) if p.returncode == 0 else None
    except (ValueError, IndexError):
        return None


class VramMonitor:
    def __init__(self) -> None:
        self.samples: list[dict] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            value = query_vram()
            if value is not None:
                self.samples.append({"elapsed_s": time.monotonic(), "used_mib": value})
            self.stop_event.wait(0.25)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=5)


def stream_chat(prompt: str, max_tokens: int, *, nonce: int,
                timeout_s: float = 300.0) -> dict:
    # Put the nonce first: a suffix would still let radix/prefix caches reuse almost all prefill.
    prompt = f"Control nonce {nonce:04d}.\n\n{prompt}"
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 4242,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    t_first = None
    prompt_tokens = completion_tokens = 0
    content: list[str] = []
    reasoning: list[str] = []
    finish_reason = None
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                datum = line[5:].strip()
                if datum == "[DONE]":
                    break
                try:
                    event = json.loads(datum)
                except json.JSONDecodeError:
                    continue
                usage = event.get("usage") or {}
                prompt_tokens = usage.get("prompt_tokens") or prompt_tokens
                completion_tokens = usage.get("completion_tokens") or completion_tokens
                for choice in event.get("choices") or []:
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or ""
                    thought = delta.get("reasoning_content") or ""
                    if (piece or thought) and t_first is None:
                        t_first = time.monotonic() - t0
                    content.append(piece)
                    reasoning.append(thought)
        total_s = time.monotonic() - t0
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "total_s": time.monotonic() - t0}
    decode_s = total_s - t_first if t_first is not None else None
    return {
        "ok": bool(t_first is not None and completion_tokens > 0 and (content or reasoning)),
        "t_first_s": t_first,
        "total_s": total_s,
        "decode_window_s": decode_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prefill_tps": prompt_tokens / t_first if prompt_tokens and t_first else None,
        "decode_tps": completion_tokens / decode_s if completion_tokens and decode_s else None,
        "finish_reason": finish_reason,
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
    }


def median(records: list[dict], probe: str, field: str) -> float | None:
    vals = [r[probe].get(field) for r in records if r[probe].get(field) is not None]
    return statistics.median(vals) if vals else None


def run(engine: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    argv, signature = commands(engine)
    kill_engine(signature)
    log_path = OUT / f"{engine}-server.log"
    print(f"starting {engine}: {' '.join(argv)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, text=True)
        load_s = wait_ready(proc)
        if load_s is None:
            kill_engine(signature)
            (OUT / f"{engine}.json").write_text(json.dumps({
                "engine": engine, "valid": False, "error": "server_not_ready",
                "argv": argv, "server_log": str(log_path),
            }, indent=2), encoding="utf-8")
            return 2
        print(f"ready in {load_s:.2f}s", flush=True)
        resident_mib = query_vram()
        records: list[dict] = []
        with VramMonitor() as monitor:
            # Warm-up is retained separately but excluded from medians.
            warmup = {
                "prefill": stream_chat(PREFILL_PROMPT, 16, nonce=0),
                "decode": stream_chat(DECODE_PROMPT, 256, nonce=0),
            }
            for idx in range(5):
                rec = {
                    "round": idx,
                    "prefill": stream_chat(PREFILL_PROMPT, 16, nonce=idx + 1),
                    "decode": stream_chat(DECODE_PROMPT, 256, nonce=idx + 1),
                }
                records.append(rec)
                print(
                    f"r{idx} pf={rec['prefill'].get('prefill_tps')} "
                    f"dc={rec['decode'].get('decode_tps')}", flush=True,
                )
        peak_mib = max((s["used_mib"] for s in monitor.samples), default=resident_mib)
        valid = all(r[p].get("ok") for r in records for p in ("prefill", "decode"))
        result = {
            "engine": engine,
            "valid": valid,
            "argv": argv,
            "load_s": load_s,
            "resident_mib": resident_mib,
            "peak_mib": peak_mib,
            "warmup": warmup,
            "records": records,
            "medians": {
                "prefill_tps": median(records, "prefill", "prefill_tps"),
                "prefill_ttft_s": median(records, "prefill", "t_first_s"),
                "decode_tps": median(records, "decode", "decode_tps"),
                "decode_ttft_s": median(records, "decode", "t_first_s"),
                "decode_total_s": median(records, "decode", "total_s"),
            },
            "vram_samples": monitor.samples,
            "server_log": str(log_path),
        }
        (OUT / f"{engine}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result["medians"], indent=2), flush=True)
    try:
        proc.terminate()
        proc.wait(timeout=15)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        proc.kill()
    finally:
        kill_engine(signature)
    return 0 if valid else 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=("llamacpp", "sglang", "vllm"))
    args = parser.parse_args()
    raise SystemExit(run(args.engine))
