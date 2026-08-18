#!/usr/bin/env python3
"""LAB-SERVE-001c — pinned realistic workload generator (deterministic, no live prompts).

Emits a custom-format JSONL dataset consumable by `sglang.benchmark.serving --dataset-name custom`
plus a provenance manifest. Two disjoint classes (input-length bands are non-overlapping so a request
can be classified downstream purely from its measured prompt_len):

  INTERACTIVE : prompt 512-4096 tok,  output cap 512   (Q&A / short reasoning / tool-select / short code)
  CODING      : prompt 8192-16384 tok, output cap 2048 (large code context + a modification/explain task)

The assistant "completion" turn exists ONLY to set the per-request output cap: custom.py takes
output_len = len(tokenizer.encode(completion)). The model generates its OWN content up to that cap with
natural EOS (bench run with --disable-ignore-eos, no --apply-chat-template so the server templates once).

Determinism: content is selected by index from fixed corpora and padded to hit token bands with a
fixed filler; a single seeded RNG (seed=1729) only chooses target lengths within each band. Re-running
reproduces byte-identical output. Tokenizer = the exact model tokenizer dir passed via --tokenizer.

Run (in sglang venv, has transformers):
  /home/augus/sglang-venv/bin/python ops/lab_serve_workload_gen.py \
     --tokenizer /home/augus/models/fp16/base --outdir runs/serving/LAB-SERVE-001c/workload
"""
import argparse, hashlib, json, pathlib, random
from transformers import AutoTokenizer

SEED = 1729
LICENSE = "CC0-1.0 (synthetic, authored for this lab; no third-party text)"
SOURCE = "lab_serve_workload_gen.py (deterministic synthetic generator)"

# --- fixed corpora (authored here; no external/licensed text) -------------------------------------
INTERACTIVE_TASKS = [
    "Explain what a Bloom filter is and when you would prefer it over a hash set.",
    "Given a REST endpoint returning paginated JSON, outline a robust retry-with-backoff strategy.",
    "What are the trade-offs between a B-tree and an LSM-tree for a write-heavy key-value store?",
    "Summarize the CAP theorem and give one concrete example per pairing.",
    "A user reports intermittent 502s behind a load balancer. List the first five things you check.",
    "Which tool would you call to look up the current weather for a city, and what arguments does it need?",
    "Describe the difference between optimistic and pessimistic locking with a banking example.",
    "Write a short regex that matches an ISO-8601 date and explain each group.",
    "When should you choose gRPC over REST for service-to-service communication?",
    "Explain how a bloom-free counting sketch (Count-Min) estimates item frequencies.",
    "Given three candidate database indexes, how do you decide which the query planner will use?",
    "Outline a strategy to migrate a monolith to services without a big-bang rewrite.",
]
CODING_TASKS = [
    "Refactor the `process_batch` function below to stream results instead of buffering the whole list, "
    "and explain what you changed.",
    "There is an off-by-one bug somewhere in the pagination logic below. Find it and give a corrected diff.",
    "Add structured logging and a retry decorator to the network calls in the module below.",
    "The code below deadlocks under load. Identify the cause and propose a fix.",
    "Write unit tests covering the edge cases of the `parse_config` function in the module below.",
]
# realistic-looking filler (code-ish for coding, prose-ish for interactive), tiled deterministically
CODE_FILLER = (
    "def handler_IDX(payload, ctx):\n"
    "    # validate and normalize the incoming record before dispatch\n"
    "    record = normalize(payload.get('data', {}), schema=ctx.schema_IDX)\n"
    "    if not record.get('id'):\n"
    "        raise ValueError('missing id in record %r' % record)\n"
    "    total = 0\n"
    "    for item in record.get('items', []):\n"
    "        total += item['qty'] * price_of(item['sku'], ctx.catalog)\n"
    "    ctx.metrics.observe('order_total', total, tags={'region': ctx.region})\n"
    "    return {'id': record['id'], 'total': total, 'status': 'ok'}\n\n"
)
PROSE_FILLER = (
    "In a distributed system the ordering of events across nodes cannot be assumed without an explicit "
    "mechanism such as a logical clock or a consensus protocol; the following context describes the "
    "constraints and prior decisions that shaped the current design so that any recommendation accounts "
    "for them rather than restating first principles. "
)


def pad_to_tokens(base_text, filler, target_tok, tok):
    """Deterministically grow base_text with tiled filler until it tokenizes to >= target_tok, then
    trim by tokens to exactly target_tok (decode of a token prefix)."""
    text = base_text
    i = 0
    while len(tok.encode(text)) < target_tok:
        text += filler.replace("IDX", str(i))
        i += 1
    ids = tok.encode(text)[:target_tok]
    return tok.decode(ids)


def completion_of_len(cap_tok, tok):
    text = ""
    i = 0
    while len(tok.encode(text)) < cap_tok:
        text += PROSE_FILLER
        i += 1
    ids = tok.encode(text)[:cap_tok]
    return tok.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-interactive", type=int, default=80)
    ap.add_argument("--n-coding", type=int, default=40)
    ap.add_argument("--interactive-min", type=int, default=512)
    ap.add_argument("--interactive-max", type=int, default=4096)
    ap.add_argument("--interactive-caps", default="128,256,512,1024",
                    help="comma-list of per-request output caps, assigned cyclically by item index")
    ap.add_argument("--coding-min", type=int, default=8192)
    ap.add_argument("--coding-max", type=int, default=9216)
    ap.add_argument("--coding-caps", default="512,1024,1536")
    a = ap.parse_args()
    inter_caps = [int(x) for x in a.interactive_caps.split(",")]
    coding_caps = [int(x) for x in a.coding_caps.split(",")]
    out = pathlib.Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    rng = random.Random(SEED)

    # precompute a completion (= output-cap setter) for every distinct cap value used
    comp = {c: completion_of_len(c, tok) for c in set(inter_caps) | set(coding_caps)}

    items, manifest = [], []
    # INTERACTIVE: prompt bands spread across [min,max]; per-request cap cycles through inter_caps
    for k in range(a.n_interactive):
        task = INTERACTIVE_TASKS[k % len(INTERACTIVE_TASKS)]
        target = rng.randint(a.interactive_min, a.interactive_max)
        cap = inter_caps[k % len(inter_caps)]
        prompt = pad_to_tokens(task + "\n\nContext:\n", PROSE_FILLER, target, tok)
        plen = len(tok.encode(prompt))
        wid = f"INT-{k:03d}"
        items.append({"workload_id": wid, "class": "interactive",
                      "conversations": [{"role": "user", "content": prompt},
                                        {"role": "assistant", "content": comp[cap]}]})
        manifest.append({"workload_id": wid, "class": "interactive", "source": SOURCE, "license": LICENSE,
                         "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                         "prompt_tok": plen, "output_cap_tok": cap,
                         "target_band": [a.interactive_min, a.interactive_max]})
    # CODING: prompt bands spread across [min,max]; per-request cap cycles through coding_caps
    for k in range(a.n_coding):
        task = CODING_TASKS[k % len(CODING_TASKS)]
        target = rng.randint(a.coding_min, a.coding_max)
        cap = coding_caps[k % len(coding_caps)]
        prompt = pad_to_tokens(task + "\n\n```python\n", CODE_FILLER, target, tok) + "\n```\n"
        plen = len(tok.encode(prompt))
        wid = f"COD-{k:03d}"
        items.append({"workload_id": wid, "class": "coding",
                      "conversations": [{"role": "user", "content": prompt},
                                        {"role": "assistant", "content": comp[cap]}]})
        manifest.append({"workload_id": wid, "class": "coding", "source": SOURCE, "license": LICENSE,
                         "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                         "prompt_tok": plen, "output_cap_tok": cap,
                         "target_band": [a.coding_min, a.coding_max]})

    # write dataset (custom-format JSONL) — keep the workload_id/class as extra keys (ignored by loader)
    ds = out / "workload_001c.jsonl"
    with ds.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    (out / "workload_manifest.json").write_text(json.dumps({
        "seed": SEED, "tokenizer": a.tokenizer, "n_interactive": a.n_interactive, "n_coding": a.n_coding,
        "classes": {"interactive": {"prompt_band": [a.interactive_min, a.interactive_max], "output_caps": inter_caps},
                    "coding": {"prompt_band": [a.coding_min, a.coding_max], "output_caps": coding_caps}},
        "dataset_sha256": hashlib.sha256(ds.read_bytes()).hexdigest(),
        "items": manifest}, indent=2), encoding="utf-8")
    # quick stats
    ip = [m["prompt_tok"] for m in manifest if m["class"] == "interactive"]
    cp = [m["prompt_tok"] for m in manifest if m["class"] == "coding"]
    print(json.dumps({"dataset": str(ds), "n": len(items),
                      "interactive_prompt_tok": {"min": min(ip), "max": max(ip)},
                      "coding_prompt_tok": {"min": min(cp), "max": max(cp)},
                      "dataset_sha256": hashlib.sha256(ds.read_bytes()).hexdigest()[:16]}, indent=2))


if __name__ == "__main__":
    main()
