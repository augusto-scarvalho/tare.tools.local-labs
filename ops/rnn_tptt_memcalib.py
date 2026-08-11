#!/usr/bin/env python
"""
RNN-08b §13 — BASE-ONLY memory-axis calibration. Predeclared grid + deterministic selection rule.
Evaluate in cost order; SELECT the first config whose BASE accuracy is strictly inside (0.20, 0.90).
No TPTT results used. Generation-free teacher-forced NLL scoring.
"""
import argparse, json, os, random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-0.5B"
DEV = "cuda"
WINDOW = (0.20, 0.90)
N = 16
GRID = [(1024, "single"), (1024, "multi_key_4"), (1024, "multi_distractor_8"),
        (2048, "single"), (2048, "multi_key_4"), (2048, "multi_distractor_8"),
        (4096, "single"), (4096, "multi_key_4"), (4096, "multi_distractor_8")]


@torch.no_grad()
def answer_nll(model, ctx_ids, ans_ids):
    ids = torch.tensor([ctx_ids + ans_ids], device=DEV)
    labels = torch.tensor([[-100] * len(ctx_ids) + ans_ids], device=DEV)
    return float(model(input_ids=ids, labels=labels).loss)


def build_sample(tok, ctx_len, difficulty, rnd):
    filler = tok("The weather is mild and the road is long. Nothing here matters. ",
                 add_special_tokens=False)["input_ids"]
    def kv(key, val):
        return tok(f" The magic number for {key} is {val}. ", add_special_tokens=False)["input_ids"]
    keys = ["".join(rnd.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(5)) for _ in range(4)]
    vals = [str(rnd.randint(1000000, 9999999)) for _ in range(4)]
    if difficulty == "single":
        pairs = [(keys[0], vals[0])]; ask = 0
        cands = [vals[0], str(rnd.randint(1000000, 9999999))]
    elif difficulty == "multi_key_4":
        pairs = list(zip(keys, vals)); ask = rnd.randint(0, 3)
        cands = list(vals)
    else:  # multi_distractor_8
        pairs = [(keys[0], vals[0])]; ask = 0
        cands = [vals[0]] + [str(rnd.randint(1000000, 9999999)) for _ in range(8)]
    q = tok(f"\nWhat is the magic number for {keys[ask]}? Answer:", add_special_tokens=False)["input_ids"]
    needles = [kv(k, v) for (k, v) in pairs]
    budget = ctx_len - sum(len(n) for n in needles) - len(q)
    body = []
    while len(body) < max(0, budget):
        body += filler
    body = body[:max(0, budget)]
    # insert needles at spread depths
    ctx = []
    step = max(1, len(body) // (len(needles) + 1))
    pos = 0
    for i, n in enumerate(needles):
        ctx += body[pos:pos + step] + n
        pos += step
    ctx += body[pos:] + q
    ctx = ctx[:ctx_len]
    correct = vals[ask]
    cand_ids = [tok(" " + c, add_special_tokens=False)["input_ids"] for c in cands]
    correct_idx = cands.index(correct)
    return ctx, cand_ids, correct_idx


def acc_for(model, tok, ctx_len, difficulty):
    rnd = random.Random(20250810 + ctx_len + hash(difficulty) % 1000)
    correct = 0
    for _ in range(N):
        ctx, cand_ids, ci = build_sample(tok, ctx_len, difficulty, rnd)
        nlls = [answer_nll(model, ctx, c) for c in cand_ids]
        if nlls.index(min(nlls)) == ci:
            correct += 1
    return round(correct / N, 3)


def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(DEV).eval()
    evaluated, selected = [], None
    for ctx_len, diff in GRID:
        a = acc_for(model, tok, ctx_len, diff)
        chance = {"single": 0.5, "multi_key_4": 0.25, "multi_distractor_8": 1 / 9}[diff]
        row = dict(context=ctx_len, difficulty=diff, base_acc=a, chance=round(chance, 3),
                   in_window=bool(WINDOW[0] < a < WINDOW[1]))
        evaluated.append(row)
        print(f"  ctx={ctx_len} diff={diff} base_acc={a} chance={round(chance,3)} in_window={row['in_window']}")
        if row["in_window"]:
            selected = dict(context=ctx_len, difficulty=diff, base_acc=a, N=N)
            break
    out = dict(model=MODEL, grid=GRID, window=WINDOW, n_per_config=N,
               selection_rule="first config in cost order with 0.20 < base_acc < 0.90",
               evaluated=evaluated,
               MEMORY_AXIS=("QUALIFIED" if selected else "NOT_QUALIFIED"),
               selected=selected)
    json.dump(out, open(os.path.join(args.outdir, "rnn08b_memcalib.json"), "w"), indent=2)
    print(json.dumps(dict(MEMORY_AXIS=out["MEMORY_AXIS"], selected=selected), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    run(ap.parse_args())
