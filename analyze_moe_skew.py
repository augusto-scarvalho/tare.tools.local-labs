#!/usr/bin/env python3
"""§E5 precondition — does Qwen3.6-35B-A3B routing have enough SKEW for a hot-expert cache?

The --moe-cache-slots N cache keeps the N hottest experts per layer resident in VRAM so they
skip the per-token H2D copy. Its ceiling is the fraction of decode expert-ACCESSES those N
experts capture (the max hit rate). Load-balanced MoEs (Qwen3 uses an aux-loss to spread
tokens) may route near-uniformly -> top-N captures little -> the cache cannot help regardless
of how it is built. Computed from the decode rows of the trace, no GPU needed."""
import sys, csv
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "/home/augus/models/qwen36-35b-moe-trace.csv"
per_layer = {}      # layer -> Counter(expert_id -> access count), decode rows only (pos>=0)
tot_tokens = 0
with open(path) as f:
    for row in csv.reader(f):
        if len(row) < 3:
            continue
        pos, layer = int(row[0]), int(row[1])
        if pos < 0:                       # skip prefill rows
            continue
        c = per_layer.setdefault(layer, Counter())
        for e in row[2:]:
            c[int(e)] += 1

layers = sorted(per_layer)
n_used = 8                                # experts active per token for this model
print(f"layers={len(layers)}  n_expert=256  n_used={n_used}")
print(f"{'topN':>6} {'mean hit%':>10} {'min':>7} {'max':>7}   (fraction of decode accesses captured)")
for N in (4, 8, 16, 32, 64, 128):
    hits = []
    for l in layers:
        c = per_layer[l]
        total = sum(c.values())
        top = sum(v for _, v in c.most_common(N))
        hits.append(top / total if total else 0.0)
    mean = sum(hits) / len(hits)
    print(f"{N:>6} {mean*100:>9.1f}% {min(hits)*100:>6.1f}% {max(hits)*100:>6.1f}%")

# Reference points: uniform routing would give topN hit = N/256. Skew ratio = actual/uniform.
print()
for N in (8, 32):
    hits = [sum(v for _, v in per_layer[l].most_common(N)) / sum(per_layer[l].values()) for l in layers]
    mean = sum(hits) / len(hits)
    uni = N / 256
    print(f"top{N}: mean hit {mean*100:.1f}%  vs uniform {uni*100:.1f}%  -> skew x{mean/uni:.2f}")
print()
print("Verdict guide: for the cache to pay, top-N (N = a few x n_used, fitting spare VRAM)")
print("must capture MOST accesses. Near-uniform (skew ~1x) => cache cannot help this model.")
