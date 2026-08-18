"""VLM visual-recognition QUALITY benchmark on MMStar (Lin-Chen/MMStar).

M0 judged description quality on 2 hand-made fixtures (vibes). This scores it OBJECTIVELY on a
real, modern, leakage-controlled validation set: MMStar = 1500 multiple-choice items, 6 balanced
ability categories (coarse/fine perception, instance/logical reasoning, math, science&tech). MCQ
answer is a single letter, so scoring is exact-match accuracy -- the visual analog of how we score
GSM8K/HumanEval for text. Ranks the 3 served VLMs on real recognition quality, not one-shot feel.

Runs UNDER WSL sglang-venv (has datasets + PIL) hitting the llama-server on 127.0.0.1:8092:
    python lmctl.py serve qwen3-vl-8b                      # from Windows
    wsl ... /home/augus/sglang-venv/bin/python /mnt/c/.../vlm_vqa_bench.py --tag qwen3-vl-8b
    python lmctl.py stop
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import re
import time
import urllib.request
from collections import Counter, defaultdict

OUT = pathlib.Path("/mnt/c/projects/local-model-lifecycle/runs/m-a-vqa")

INSTR = ("\nAnswer with ONLY the single option letter (A, B, C, or D). "
         "Do not explain. Output just the letter.")


def load_subset(n_per_cat: int, seed: int):
    from datasets import load_dataset
    ds = load_dataset("Lin-Chen/MMStar", split="val").shuffle(seed=seed)
    per = defaultdict(list)
    for row in ds:
        c = row["category"]
        if len(per[c]) < n_per_cat:
            per[c].append(row)
    items = [r for rows in per.values() for r in rows]
    return items


def img_b64(pil) -> str:
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


CHOICE = re.compile(r"[ABCD]")


def parse_choice(text: str) -> str | None:
    t = (text or "").strip()
    if not t:
        return None
    if t.upper() in ("A", "B", "C", "D"):
        return t.upper()
    m = re.search(r"(?i)\b(?:answer|option|choice)\b[^A-D]{0,12}([ABCD])\b", t)
    if m:
        return m.group(1).upper()
    m = re.search(r"^\(?([ABCD])[.):\s]", t.strip())
    if m:
        return m.group(1).upper()
    # last resort: the last standalone A-D in the string (thinking models conclude at the end)
    hits = CHOICE.findall(t.upper())
    return hits[-1] if hits else None


def ask(port: int, b64: str, question: str, max_tokens: int) -> tuple[str, float]:
    payload = {"model": "vlm", "temperature": 0, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content": [
                   {"type": "text", "text": question + INSTR},
                   {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.load(r)
    msg = body["choices"][0]["message"]
    text = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()
    return text, time.monotonic() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--n-per-cat", type=int, default=25)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260806)
    a = ap.parse_args()

    items = load_subset(a.n_per_cat, a.seed)
    print(f"\n=== MMStar quality bench :: {a.tag}  (n={len(items)}, {a.n_per_cat}/cat) ===")
    correct = Counter(); total = Counter(); unparsed = 0; recs = []
    t_start = time.monotonic()
    for i, row in enumerate(items):
        try:
            text, dt = ask(a.port, img_b64(row["image"]), row["question"], a.max_tokens)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i+1}/{len(items)}] ERROR {e}"); continue
        pred = parse_choice(text)
        gold = row["answer"].strip().upper()
        ok = (pred == gold)
        cat = row["category"]
        total[cat] += 1
        if ok:
            correct[cat] += 1
        if pred is None:
            unparsed += 1
        recs.append({"index": row["index"], "category": cat, "gold": gold, "pred": pred,
                     "ok": ok, "wall_s": round(dt, 2), "raw": text[:200]})
        if (i + 1) % 25 == 0:
            acc = sum(correct.values()) / max(1, sum(total.values()))
            print(f"  ...{i+1}/{len(items)}  running acc={acc:.3f}")

    n = sum(total.values()); c = sum(correct.values())
    print(f"\n  OVERALL {a.tag}: {c}/{n} = {c/n:.3f}   (unparsed={unparsed}, "
          f"{time.monotonic()-t_start:.0f}s)")
    for cat in sorted(total):
        print(f"    {cat:26} {correct[cat]:3}/{total[cat]:<3} = {correct[cat]/total[cat]:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    outf = OUT / f"MMSTAR_{a.tag}.json"
    outf.write_text(json.dumps({"tag": a.tag, "n": n, "overall_acc": round(c/n, 4),
                                "per_cat": {k: [correct[k], total[k]] for k in total},
                                "unparsed": unparsed, "records": recs}, indent=2))
    print(f"  raw -> {outf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
