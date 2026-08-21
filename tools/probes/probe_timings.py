"""Does llama-server hand us exact prefill/generation timings in the stream?

If it does, every rate in this platform should come from it instead of from wall-clock:
the server knows precisely where prefill ended, and that boundary is invisible from
outside on a thinking model that never emits content.
"""
import json, subprocess, sys, time, urllib.request

BIN = "/home/augus/src/slop.cpp-base/build/bin/llama-server"
MODEL = "/home/augus/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
p = subprocess.Popen(["wsl", "-d", "Ubuntu-24.04", "--", BIN, "-m", MODEL,
                      "--host", "0.0.0.0", "--port", "8099", "-ncmoe", "8",
                      "-c", "2048", "-ctk", "q8_0", "-ctv", "q8_0"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:8099/health", timeout=3).read()
            break
        except Exception:
            time.sleep(2)
    else:
        print("server never healthy"); sys.exit(1)

    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 16, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request("http://127.0.0.1:8099/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    keys, timings = set(), None
    with urllib.request.urlopen(req, timeout=180) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except ValueError:
                continue
            keys |= set(ev.keys())
            if "timings" in ev:
                timings = ev["timings"]
    print("TOP-LEVEL KEYS SEEN:", sorted(keys))
    print("TIMINGS PRESENT:", timings is not None)
    if timings:
        print(json.dumps(timings, indent=2))
finally:
    p.kill()
    subprocess.run(["wsl", "-d", "Ubuntu-24.04", "--", "pkill", "-9", "-f", "llama-server"])
