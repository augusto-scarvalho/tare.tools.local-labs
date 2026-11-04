#!/usr/bin/env python3
import json, sys, time, urllib.request, os

PORT = 8080
LS = [8000, 16000, 24000, 30000]
DS = [0.10, 0.25, 0.50, 0.75, 0.90]

FILLER = ("The quarterly maintenance log for sector 7 records routine checks with no anomalies "
          "and nominal readings across all subsystems. ") # ~22 tokens/unit

def niah(L, depth, code):
    nw = int(L / 1.35)
    pool = (FILLER * (nw // 22 + 2)).split()
    words = pool[:nw]
    needle = f"IMPORTANT REMEMBER THIS: the Grover facility access code is {code}.".split()
    pos = max(1, min(len(words) - 1, int(len(words) * depth)))
    words = words[:pos] + needle + words[pos:]
    return (" ".join(words) +
            "\n\nQuestion: what is the Grover facility access code? Reply with ONLY the code, nothing else.")

def ask(prompt):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 24,
        "temperature": 0.0,
        "top_k": 1,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    t0 = time.time()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}
    )
    r = json.load(urllib.request.urlopen(req, timeout=300))
    el = time.time() - t0
    content = r["choices"][0]["message"].get("content") or ""
    timings = r.get("timings", {})
    usage = r.get("usage", {})
    return content.strip(), el, timings, usage

def main():
    print(f"{'L_target':>9} {'prompt_n':>9} {'d':>5} {'found':>6} {'latency_s':>10} {'prefill_tps':>12} {'decode_tps':>11}", flush=True)
    rows = []
    for L in LS:
        for d in DS:
            code = f"ZK{50000 + (L//1000)*11 + int(d*100)*17}Q"
            content, el, t, u = ask(niah(L, d, code))
            found = code in content
            pn = u.get("prompt_tokens", t.get("prompt_n", 0))
            ptps = t.get("prompt_per_second", (pn / el) if el else 0)
            dtps = t.get("predicted_per_second", 0)
            rows.append({"L": L, "prompt_tokens": pn, "depth": d, "found": found, "latency_s": el, "prefill_tps": ptps, "decode_tps": dtps})
            print(f"{L:>9} {pn:>9} {d:>5.2f} {str(found):>6} {el:>10.2f} {ptps:>12.1f} {dtps:>11.2f}", flush=True)

    print("\n== Summary Table ==", flush=True)
    summary = []
    for L in LS:
        r = [x for x in rows if x["L"] == L]
        acc = sum(1 for x in r if x["found"]) / len(r)
        mean_lat = sum(x["latency_s"] for x in r) / len(r)
        mean_ptps = sum(x["prefill_tps"] for x in r) / len(r)
        print(f"Depth ~{L:>6} (prompt_tokens {r[0]['prompt_tokens']:>6}): NIAH Accuracy = {acc*100:>5.1f}% | Avg Latency = {mean_lat:>5.1f}s | Prefill = {mean_ptps:>6.1f} t/s", flush=True)
        summary.append({
            "target_context": L,
            "actual_prompt_tokens": r[0]["prompt_tokens"],
            "accuracy": acc,
            "mean_latency_s": round(mean_lat, 2),
            "mean_prefill_tps": round(mean_ptps, 1)
        })
    os.makedirs("runs/qwen38-niah", exist_ok=True)
    with open("runs/qwen38-niah/niah_summary.json", "w") as f:
        json.dump({"summary": summary, "raw_rows": rows}, f, indent=2)
    print("Saved runs/qwen38-niah/niah_summary.json", flush=True)

if __name__ == "__main__":
    main()
