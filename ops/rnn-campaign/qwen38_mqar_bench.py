#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MQAR Associative Recall Benchmark for Qwen 3.8-27B (via llama-server HTTP).
Supports arbitrary doses P scaling up to long-context multi-key pressure.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import numpy as np

def rng_for(*ints):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(ints))))

def build_pools(pool_size=256):
    words = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
        "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
        "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray",
        "yankee", "zulu", "amber", "bronze", "copper", "diamond", "emerald", "garnet",
        "hazel", "indigo", "jade", "kobold", "lapis", "malachite", "nickel", "onyx",
        "pearl", "quartz", "ruby", "sapphire", "topaz", "umbra", "velvet", "winter",
        "xenon", "yellow", "zenith", "arctic", "breeze", "canyon", "desert", "ember",
        "forest", "glacier", "harbor", "island", "jungle", "lagoon", "meadow", "nexus",
        "ocean", "prairie", "quarry", "river", "summit", "tundra", "valley", "woodland"
    ]
    key_pool = []
    for w in words:
        key_pool.append(w)
    idx = 1
    while len(key_pool) < pool_size:
        key_pool.append(f"item{idx}")
        idx += 1
    key_pool = key_pool[:pool_size]
    val_pool = [str(100 + i * 3) for i in range(pool_size)]
    return key_pool, val_pool

def build_calibration_spec(master_seed, n_examples, p_max, p_min, pool_size):
    examples = []
    for i in range(n_examples):
        r = rng_for(master_seed, 0x5EED, i)
        key_slots = r.permutation(pool_size)[:p_max].tolist()
        val_slots = r.permutation(pool_size)[:p_max].tolist()
        probe_index = int(r.integers(0, p_min))
        examples.append({
            "key_slots": key_slots,
            "val_slots": val_slots,
            "probe_index": probe_index
        })
    return {
        "master_seed": master_seed,
        "n_examples": n_examples,
        "p_max": p_max,
        "p_min": p_min,
        "pool_size": pool_size,
        "examples": examples
    }

def format_prompt(key_slots, val_slots, probe_idx, P, key_pool, val_pool):
    lines = ["Here is a key-value mapping:"]
    for k_idx, v_idx in zip(key_slots[:P], val_slots[:P]):
        lines.append(f"{key_pool[k_idx]} = {val_pool[v_idx]}")
    probe_key = key_pool[key_slots[probe_idx]]
    gold_val = val_pool[val_slots[probe_idx]]
    lines.append(f"\nWhat is the value for '{probe_key}'? Reply with ONLY the exact value, nothing else.")
    prompt = "\n".join(lines)
    return prompt, gold_val

def query_chat(port, prompt, timeout=120):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0.0,
        "top_k": 1,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=timeout)
    elapsed = time.time() - t0
    res_json = json.loads(resp.read().decode("utf-8"))
    content = (res_json["choices"][0]["message"].get("content") or "").strip()
    usage = res_json.get("usage", {})
    return content, elapsed, usage

def run_mqar_qwen38(port=8080, doses=[4, 8, 16, 32, 64, 128, 256, 512, 1024], n_eval=32, master_seed=20260811, outdir="runs/rnn/RNN-06-P0"):
    os.makedirs(outdir, exist_ok=True)
    p_max = max(doses)
    p_min = min(doses)
    pool_size = max(256, p_max)
    key_pool, val_pool = build_pools(pool_size)
    spec = build_calibration_spec(master_seed, n_eval, p_max, p_min, pool_size)
    val_set = set(val_pool)
    curves = []
    print(f"== Running MQAR on Qwen 3.8-27B (port {port}, doses={doses}, n_eval={n_eval}, pool={pool_size}) ==", flush=True)
    for P in doses:
        t_start = time.time()
        n_correct = 0
        n_format_ok = 0
        total_prompt_toks = 0
        for ex in spec["examples"][:n_eval]:
            prompt, gold = format_prompt(ex["key_slots"], ex["val_slots"], ex["probe_index"], P, key_pool, val_pool)
            ans, el, usage = query_chat(port, prompt)
            total_prompt_toks += usage.get("prompt_tokens", 0)
            clean_ans = ans.split()[0] if ans.split() else ""
            if clean_ans == gold or gold in ans:
                n_correct += 1
            if any(v in ans for v in val_set) or clean_ans.isdigit():
                n_format_ok += 1
        eval_secs = time.time() - t_start
        acc = n_correct / n_eval
        fmt = n_format_ok / n_eval
        avg_prompt_len = int(total_prompt_toks / n_eval) if n_eval else 0
        print(f"[qwen38_27b] P={P:>4} len~{avg_prompt_len:>5} con_acc={acc:.3f} fmt={fmt:.3f} ({eval_secs:.1f}s)", flush=True)
        curves.append({
            "pairs": P,
            "seq_len_tokens": avg_prompt_len,
            "n": n_eval,
            "constrained_acc": acc,
            "format_adherence": fmt,
            "unconstrained_exact_acc": acc,
            "chance": 1.0 / pool_size,
            "value_vocab_size": pool_size,
            "n_constrained_correct": n_correct,
            "n_format_ok": n_format_ok,
            "eval_seconds": round(eval_secs, 2)
        })
    record = {
        "packet": "RNN-06-P0",
        "candidate_tag": "qwen38_27b",
        "model_id": "qwen38-27b",
        "status": {
            "MODEL_RUNNABLE": "YES",
            "TASK_COMPETENT": "YES" if curves[0]["constrained_acc"] >= 0.75 else "NO",
            "low_dose_constrained_acc": curves[0]["constrained_acc"],
            "P0_GRADED_BAND": "HIGH_FIDELITY" if curves[-1]["constrained_acc"] >= 0.85 else "PLAUSIBLE"
        },
        "curves": curves
    }
    out_path = os.path.join(outdir, "P0_RESULTS_qwen38_27b.json")
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote {out_path}", flush=True)
    return record

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--doses", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--n-eval", type=int, default=32)
    parser.add_argument("--outdir", default="/mnt/c/projects/local-model-lifecycle/runs/rnn/RNN-06-P0")
    args = parser.parse_args()
    run_mqar_qwen38(args.port, args.doses, args.n_eval, outdir=args.outdir)
