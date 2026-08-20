#!/usr/bin/env python3
import json, sys, re, time, urllib.request, os

PORT = 8080
PROBLEMS = "/mnt/c/projects/local-model-lifecycle/workloads/gsm8k.jsonl"
SUBSET = 30

def extract_gold(a):
    a = str(a).split('####')[-1]
    m = re.findall(r'-?\d[\d,]*\.?\d*', a)
    return m[-1].replace(',', '') if m else None

def extract_pred(text):
    t = text.split('</think>')[-1]
    m = re.search(r'####\s*(-?\d[\d,]*\.?\d*)', t)
    if m:
        return m.group(1).replace(',', '')
    nums = re.findall(r'-?\d[\d,]*\.?\d*', t)
    return nums[-1].replace(',', '') if nums else None

def is_equal(a, b):
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-5
    except Exception:
        return str(a).strip() == str(b).strip()

def ask(prompt):
    inst = f"Solve this math problem. Show your work briefly, then end with the final answer on its own line in the form: #### <number>\n\n{prompt}"
    body = {
        "messages": [{"role": "user", "content": inst}],
        "max_tokens": 512,
        "temperature": 0.0,
        "top_k": 1,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    t0 = time.time()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    r = json.load(urllib.request.urlopen(req, timeout=120))
    el = time.time() - t0
    content = r["choices"][0]["message"].get("content") or ""
    timings = r.get("timings", {})
    usage = r.get("usage", {})
    return content, el, timings, usage

def main():
    probs = []
    for line in open(PROBLEMS, encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            probs.append({"task_id": d["task_id"], "prompt": d["prompt"], "gold": extract_gold(d["answer"])})
    probs = probs[:SUBSET]
    
    print(f"== Evaluating GSM8K on Qwen 3.8-27B (N={len(probs)}) ==", flush=True)
    correct = 0
    total_tokens = 0
    total_time = 0.0
    results = []
    
    for idx, p in enumerate(probs):
        ans, el, timings, usage = ask(p["prompt"])
        pred_val = extract_pred(ans)
        ok = is_equal(pred_val, p["gold"])
        if ok:
            correct += 1
        gen_tokens = usage.get("completion_tokens", timings.get("predicted_n", len(ans.split())))
        tps = gen_tokens / el if el > 0 else 0
        total_tokens += gen_tokens
        total_time += el
        print(f"[{idx+1:02d}/{SUBSET}] {p['task_id']:<10} Gold={p['gold']:<8} Pred={str(pred_val):<8} Match={str(ok):<5} ({tps:5.1f} t/s)", flush=True)
        results.append({
            "task_id": p["task_id"],
            "gold": p["gold"],
            "pred": pred_val,
            "correct": ok,
            "tokens": gen_tokens,
            "seconds": round(el, 2)
        })
        
    acc = correct / len(probs)
    mean_tps = total_tokens / total_time if total_time > 0 else 0
    print("\n== Final GSM8K Accuracy ==", flush=True)
    print(f"Accuracy: {correct}/{len(probs)} = {acc*100:.1f}% | Avg Generation Speed: {mean_tps:.1f} tok/s", flush=True)
    
    os.makedirs("runs/qwen38-gsm8k", exist_ok=True)
    with open("runs/qwen38-gsm8k/gsm8k_eval_summary.json", "w") as f:
        json.dump({
            "model": "Qwen3.8-27B",
            "accuracy": acc,
            "correct": correct,
            "total": len(probs),
            "mean_gen_tps": round(mean_tps, 2),
            "results": results
        }, f, indent=2)
    print("Saved runs/qwen38-gsm8k/gsm8k_eval_summary.json", flush=True)

if __name__ == "__main__":
    main()
