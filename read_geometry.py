#!/usr/bin/env python3
"""Read the KV-relevant geometry from a GGUF so we can DEDUCE max context per lever config.
KV bytes/token = 2 (K+V) * n_layer * n_head_kv * head_dim * bytes_per_elem.
Run:  PYTHONPATH=/home/augus/src/llama.cpp-master/gguf-py python3 read_geometry.py <gguf...>"""
import sys
from gguf import GGUFReader, GGUFValueType

def g(r, *names):
    for k in names:
        f = r.get_field(k)
        if f is None:
            continue
        t = f.types[-1] if f.types else None
        if t == GGUFValueType.STRING:
            return bytes(f.parts[f.data[-1]]).decode("utf-8", "replace")
        v = f.parts[f.data[-1]]
        try:
            return v[0].item() if hasattr(v, "__len__") else v.item()
        except Exception:
            return v.tolist() if hasattr(v, "tolist") else v
    return None

for path in sys.argv[1:]:
    r = GGUFReader(path)
    arch = g(r, "general.architecture")
    p = f"{arch}."
    nl   = g(r, p+"block_count")
    nhkv = g(r, p+"attention.head_count_kv")
    nh   = g(r, p+"attention.head_count")
    ekk  = g(r, p+"attention.key_length")
    evv  = g(r, p+"attention.value_length")
    nembd= g(r, p+"embedding_length")
    nctx = g(r, p+"context_length")
    rope = g(r, p+"rope.freq_base")
    ropescale = g(r, p+"rope.scaling.type")
    head_dim = ekk or (nembd // nh if (nembd and nh) else None)
    vdim = evv or head_dim
    print(f"\n== {path.split('/')[-1]} ==  arch={arch}")
    print(f"  n_layer={nl}  n_head={nh}  n_head_kv={nhkv}  head_dim(K)={head_dim} head_dim(V)={vdim}")
    print(f"  n_ctx_train={nctx}  rope_freq_base={rope}  rope_scaling={ropescale}")
    if nl and nhkv and head_dim:
        per_tok = 2 * nl * nhkv * (head_dim)  # elements/token (K+V, symmetric head_dim assumed)
        for fmt, byt in (("f16", 2.0), ("q8_0", 1.0625), ("q4_0", 0.5625)):
            mb_per_1k = per_tok * byt * 1000 / (1024*1024)
            print(f"  KV {fmt:5}: {per_tok*byt:8.1f} B/tok  = {mb_per_1k:6.1f} MB per 1k ctx"
                  f"  -> 128k ctx = {mb_per_1k*128:7.0f} MB")
