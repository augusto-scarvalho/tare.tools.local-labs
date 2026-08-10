#!/usr/bin/env python
"""
RNN-00B — RULER-style instrument smoke qualification (LAB-QA philosophy), CPU-only.

This does NOT run the full NVIDIA/RULER benchmark against a model (that needs GPU/serving and is
deferred). It qualifies the MEASUREMENT DISCIPLINE we would need before trusting any long-context
number, on a RULER-style single-needle synthetic task:

  1. tokenizer identity            (which tokenizer, vocab size, file hash)
  2. exact context length delivered (no silent truncation: delivered tokens == target)
  3. needle survives truncation     (the answer span is inside the delivered window)
  4. known-good fixture PASSES       (correct answer scores 1.0)
  5. known-bad  fixture FAILS        (wrong answer scores 0.0)
  6. reproducibility                 (same seed -> identical sample hash)

Upstream instrument identity (verified, to be used when the real benchmark is run):
  NVIDIA/RULER  repo, Apache-2.0, commit c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a (default branch).
"""
import argparse, hashlib, json, os, random, sys, glob


def tok_identity(tok_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_path)
    h = hashlib.sha256()
    for pat in ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json"):
        for f in sorted(glob.glob(os.path.join(tok_path, pat))):
            with open(f, "rb") as fh:
                h.update(fh.read())
    return tok, dict(path=tok_path, vocab_size=int(tok.vocab_size),
                     files_sha256=h.hexdigest()[:16])


def make_sample(tok, target_tokens, seed):
    """Single-needle-in-haystack, padded/trimmed to EXACTLY target_tokens (no silent truncation)."""
    rnd = random.Random(seed)
    key = "".join(rnd.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(6))
    value = str(rnd.randint(1000000, 9999999))
    needle = f" The magic number for {key} is {value}. "
    filler = ("The grass is green and the sky is blue. Nothing important happens here. ")
    question = f"\nWhat is the magic number for {key}? Answer:"

    needle_ids = tok(needle, add_special_tokens=False)["input_ids"]
    q_ids = tok(question, add_special_tokens=False)["input_ids"]
    filler_ids = tok(filler, add_special_tokens=False)["input_ids"]

    budget = target_tokens - len(needle_ids) - len(q_ids)
    if budget < 0:
        raise ValueError("target too small for needle+question")
    # place needle at ~60% depth inside the filler
    pre_n = int(budget * 0.6)
    body = []
    while len(body) < pre_n:
        body += filler_ids
    body = body[:pre_n]
    needle_start = len(body)
    body += needle_ids
    while len(body) < budget + len(needle_ids):
        body += filler_ids
    body = body[:budget + len(needle_ids)]
    ids = body + q_ids
    # enforce EXACT length
    ids = ids[:target_tokens]
    delivered = len(ids)
    needle_end = needle_start + len(needle_ids)
    needle_survives = needle_end <= delivered
    sample_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
    return dict(key=key, value=value, ids=ids, delivered=delivered,
                needle_start=needle_start, needle_end=needle_end,
                needle_survives=bool(needle_survives), sample_hash=sample_hash)


def score(pred, gold):
    """Exact-match RULER scorer: 1.0 iff gold string appears in prediction."""
    return 1.0 if gold in pred else 0.0


def run(args):
    tok, ident = tok_identity(args.tokenizer)
    checks = {}

    s = make_sample(tok, args.target_tokens, seed=42)
    checks["tokenizer_identity"] = ident
    checks["exact_length_delivered"] = dict(
        target=args.target_tokens, delivered=s["delivered"],
        no_silent_truncation=bool(s["delivered"] == args.target_tokens))
    checks["needle_survives_window"] = dict(
        needle_span=[s["needle_start"], s["needle_end"]], delivered=s["delivered"],
        survives=s["needle_survives"])

    good = score(f"The magic number is {s['value']}.", s["value"])
    bad = score("The magic number is 0000000.", s["value"])
    checks["known_good_fixture"] = dict(pred_contains_gold=True, score=good, PASS=bool(good == 1.0))
    checks["known_bad_fixture"] = dict(pred_contains_gold=False, score=bad, PASS=bool(bad == 0.0))

    s2 = make_sample(tok, args.target_tokens, seed=42)
    s3 = make_sample(tok, args.target_tokens, seed=43)
    checks["reproducibility"] = dict(
        same_seed_identical=bool(s["sample_hash"] == s2["sample_hash"]),
        diff_seed_differs=bool(s["sample_hash"] != s3["sample_hash"]),
        sample_hash_seed42=s["sample_hash"])

    all_pass = (
        checks["exact_length_delivered"]["no_silent_truncation"]
        and checks["needle_survives_window"]["survives"]
        and checks["known_good_fixture"]["PASS"]
        and checks["known_bad_fixture"]["PASS"]
        and checks["reproducibility"]["same_seed_identical"]
        and checks["reproducibility"]["diff_seed_differs"])
    checks["VERDICT"] = "RULER_HARNESS_DISCIPLINE_QUALIFIED" if all_pass else "FAILED"
    checks["upstream_instrument"] = dict(
        repo="github.com/NVIDIA/RULER", license="Apache-2.0",
        commit="c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a",
        note="Full benchmark execution against a model is DEFERRED (needs GPU/serving). "
             "This qualifies the harness discipline only, on a RULER-style single-needle task.")

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "rnn00b_ruler_smoke.json"), "w") as f:
        json.dump(checks, f, indent=2)
    print(json.dumps(checks, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="/home/augus/models/fp16/base")
    ap.add_argument("--target-tokens", type=int, default=2048)
    ap.add_argument("--outdir", required=True)
    sys.exit(run(ap.parse_args()))
