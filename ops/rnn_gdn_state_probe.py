#!/usr/bin/env python
"""
RNN-01/02/03 — Gated DeltaNet (Qwen3.5) recurrent-state archaeology + checkpoint/restore.

Packet: Local AI Lab RNN Foundation R0/R1.  Frugal, CPU-only, deterministic.

We build a STRUCTURALLY-FAITHFUL tiny surrogate: a Qwen3_5TextModel whose per-linear-layer
Gated-DeltaNet head dims are set EXACTLY to the real Qwen3.5-0.8B config (linear_num_*_heads,
linear_*_head_dim, linear_conv_kernel_dim, head_dim, full_attention_interval), reducing only
num_hidden_layers / hidden_size / vocab_size to keep it tiny. Consequently the per-layer state
TENSORS (conv_states, recurrent_states) are shape/dtype-identical to real Qwen3.5-0.8B; byte
accounting scales by the real layer counts. Checkpoint/restore determinism is a property of the
cache mechanics and is weight-independent, so the conclusion transfers to the real model.

Outputs (into --outdir):
  RNN_STATE_INVENTORY.json   per-layer cache tensor inventory (surrogate) + analytic real-model scaling
  RNN_STATE_SIZES.csv        flat per-layer sizes
  rnn01_inventory.txt        human dump
  rnn02_checkpoint_restore.json   determinism verdicts
  rnn03_branching.json       optional branching verdicts
"""
import argparse, json, os, sys, copy, csv, platform

import torch


# ---- real Qwen3.5-0.8B linear/attn head geometry (verified from HF config.json) ----
REAL_08B = dict(
    hidden_size=1024, num_hidden_layers=24, num_attention_heads=8, num_key_value_heads=2,
    head_dim=256, full_attention_interval=4, intermediate_size=3584, vocab_size=248320,
    linear_num_key_heads=16, linear_num_value_heads=16,
    linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
)
# real Qwen3.6-27B geometry (verified from local fp16/base text_config + HF)
REAL_27B = dict(
    hidden_size=5120, num_hidden_layers=64, num_attention_heads=24, num_key_value_heads=4,
    head_dim=256, full_attention_interval=4, intermediate_size=17408, vocab_size=248320,
    linear_num_key_heads=16, linear_num_value_heads=48,
    linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
)


def build_surrogate(match, n_layers, hidden, vocab):
    """Qwen3_5TextConfig with REAL per-layer GDN head dims from `match`, tiny elsewhere."""
    try:
        from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
    except Exception:
        from transformers.models.qwen3_5 import Qwen3_5TextConfig, Qwen3_5TextModel
    cfg = Qwen3_5TextConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=n_layers,
        num_attention_heads=match["num_attention_heads"],
        num_key_value_heads=match["num_key_value_heads"],
        head_dim=match["head_dim"],
        max_position_embeddings=4096,
        full_attention_interval=match["full_attention_interval"],
        linear_num_key_heads=match["linear_num_key_heads"],
        linear_num_value_heads=match["linear_num_value_heads"],
        linear_key_head_dim=match["linear_key_head_dim"],
        linear_value_head_dim=match["linear_value_head_dim"],
        linear_conv_kernel_dim=match["linear_conv_kernel_dim"],
        attention_bias=False,
        tie_word_embeddings=False,
    )
    model = Qwen3_5TextModel(cfg)
    model.eval()
    return cfg, model


def tensor_attrs(obj):
    """Return {name: tensor} for all torch.Tensor attributes on a cache-layer object."""
    out = {}
    names = set()
    if hasattr(obj, "__dict__"):
        names |= set(vars(obj).keys())
    for n in ("conv_states", "recurrent_states", "keys", "values", "key_cache", "value_cache"):
        names.add(n)
    for n in sorted(names):
        try:
            v = getattr(obj, n)
        except Exception:
            continue
        if isinstance(v, torch.Tensor) and v.numel() > 0:
            out[n] = v
    return out


def inventory(cache, layer_types):
    layers = getattr(cache, "layers", None)
    if layers is None:
        layers = [cache.layers[i] for i in range(len(layer_types))]
    rows = []
    for i, lyr in enumerate(layers):
        lt = layer_types[i] if i < len(layer_types) else "?"
        tens = tensor_attrs(lyr)
        if not tens:
            rows.append(dict(layer=i, layer_type=lt, tensor="<none>", shape=None,
                             dtype=None, numel=0, bytes=0))
        for name, v in tens.items():
            rows.append(dict(layer=i, layer_type=lt, tensor=name,
                             shape=list(v.shape), dtype=str(v.dtype).replace("torch.", ""),
                             numel=int(v.numel()), bytes=int(v.numel() * v.element_size())))
    return rows


def analytic_state_bytes(cfg_dict, batch=1, recurrent_dtype_bytes=4, conv_dtype_bytes=2):
    """Per-request GDN state bytes for a real config (conv + recurrent only; excludes full-attn KV)."""
    kd = cfg_dict["linear_key_head_dim"]; vd = cfg_dict["linear_value_head_dim"]
    nk = cfg_dict["linear_num_key_heads"]; nv = cfg_dict["linear_num_value_heads"]
    key_dim = kd * nk; value_dim = vd * nv
    conv_dim = key_dim * 2 + value_dim
    K = cfg_dict["linear_conv_kernel_dim"]
    n_layers = cfg_dict["num_hidden_layers"]
    interval = cfg_dict["full_attention_interval"]
    layer_types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                   for i in range(n_layers)]
    n_linear = layer_types.count("linear_attention")
    n_full = layer_types.count("full_attention")
    conv_numel = batch * conv_dim * K
    rec_numel = batch * nv * kd * vd
    per_linear_bytes = conv_numel * conv_dtype_bytes + rec_numel * recurrent_dtype_bytes
    return dict(
        n_layers=n_layers, n_linear=n_linear, n_full=n_full,
        key_dim=key_dim, value_dim=value_dim, conv_dim=conv_dim, conv_kernel=K,
        conv_state_shape=[batch, conv_dim, K], recurrent_state_shape=[batch, nv, kd, vd],
        conv_numel=conv_numel, recurrent_numel=rec_numel,
        per_linear_layer_bytes=per_linear_bytes,
        total_gdn_state_bytes=per_linear_bytes * n_linear,
        total_gdn_state_MiB=round(per_linear_bytes * n_linear / (1024**2), 3),
        note=("conv assumed model-dtype(2B); recurrent assumed fp32(4B) per mamba_ssm_dtype. "
              "Full-attention KV cache is separate and grows with sequence length."),
    )


def max_abs_diff(a, b):
    return float((a.float() - b.float()).abs().max().item())


def cache_recurrent_snapshot(cache, layer_types):
    """Deep snapshot of per-layer conv/recurrent tensors (for cross-path comparison)."""
    snap = {}
    layers = cache.layers
    for i, lt in enumerate(layer_types):
        t = tensor_attrs(layers[i])
        snap[i] = {k: v.detach().clone() for k, v in t.items()}
    return snap


def run(args):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device(args.device)
    dtype = torch.float32

    cfg, model = build_surrogate(REAL_08B, args.layers, args.hidden, args.vocab)
    model = model.to(dev, dtype=dtype)
    layer_types = list(cfg.layer_types)

    os.makedirs(args.outdir, exist_ok=True)
    meta = dict(
        torch=torch.__version__, python=platform.python_version(),
        device=str(dev), dtype="float32", seed=0,
        surrogate_config=dict(hidden_size=cfg.hidden_size, num_hidden_layers=cfg.num_hidden_layers,
                              vocab_size=cfg.vocab_size, layer_types=layer_types,
                              linear_num_key_heads=cfg.linear_num_key_heads,
                              linear_num_value_heads=cfg.linear_num_value_heads,
                              linear_key_head_dim=cfg.linear_key_head_dim,
                              linear_value_head_dim=cfg.linear_value_head_dim,
                              linear_conv_kernel_dim=cfg.linear_conv_kernel_dim,
                              head_dim=cfg.head_dim, full_attention_interval=4),
    )

    Lp, Lc = args.prefix_len, args.cont_len
    g = torch.Generator().manual_seed(1234)
    full_ids = torch.randint(0, cfg.vocab_size, (1, Lp + Lc), generator=g).to(dev)
    prefix, cont = full_ids[:, :Lp], full_ids[:, Lp:]

    # ---------- RNN-01: inventory ----------
    with torch.no_grad():
        out = model(input_ids=prefix, use_cache=True)
    cache = out.past_key_values
    inv_rows = inventory(cache, layer_types)
    real08 = analytic_state_bytes(REAL_08B)
    real27 = analytic_state_bytes(REAL_27B)
    surro = analytic_state_bytes(dict(REAL_08B, num_hidden_layers=cfg.num_hidden_layers))

    with open(os.path.join(args.outdir, "RNN_STATE_INVENTORY.json"), "w") as f:
        json.dump(dict(meta=meta, cache_type=type(cache).__name__,
                       cache_layer_type=type(cache.layers[0]).__name__,
                       inventory=inv_rows,
                       analytic=dict(surrogate=surro, qwen35_0_8b=real08, qwen36_27b=real27)),
                  f, indent=2)
    with open(os.path.join(args.outdir, "RNN_STATE_SIZES.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "layer_type", "tensor", "shape",
                                          "dtype", "numel", "bytes"])
        w.writeheader()
        for r in inv_rows:
            r = dict(r); r["shape"] = json.dumps(r["shape"]); w.writerow(r)
    with open(os.path.join(args.outdir, "rnn01_inventory.txt"), "w") as f:
        f.write(f"cache_type={type(cache).__name__} layer_obj={type(cache.layers[0]).__name__}\n")
        for r in inv_rows:
            f.write(f"L{r['layer']:02d} {r['layer_type']:16s} {r['tensor']:16s} "
                    f"{str(r['shape']):24s} {r['dtype']} numel={r['numel']} bytes={r['bytes']}\n")
        f.write(f"\nsurrogate total GDN state: {surro['total_gdn_state_MiB']} MiB "
                f"({surro['n_linear']} linear layers)\n")
        f.write(f"REAL Qwen3.5-0.8B total GDN state: {real08['total_gdn_state_MiB']} MiB "
                f"({real08['n_linear']} linear layers)\n")
        f.write(f"REAL Qwen3.6-27B total GDN state: {real27['total_gdn_state_MiB']} MiB "
                f"({real27['n_linear']} linear layers)\n")

    # ---------- RNN-02: checkpoint / restore ----------
    # Path A (continuous): prefix -> cache_A ; checkpoint S=deepcopy(cache_A) ; continue cont -> hidden_A
    with torch.no_grad():
        outA = model(input_ids=prefix, use_cache=True)
    cacheA = outA.past_key_values
    S = copy.deepcopy(cacheA)                              # <-- checkpoint
    # serialize the raw GDN state tensors to disk to prove a portable checkpoint
    disk = {f"L{i}_{k}": v for i, d in cache_recurrent_snapshot(cacheA, layer_types).items()
            for k, v in d.items()}
    ckpt_path = os.path.join(args.outdir, "rnn02_state_checkpoint.pt")
    torch.save(disk, ckpt_path)
    with torch.no_grad():
        contA = model(input_ids=cont, past_key_values=cacheA, use_cache=True)
    hidden_A = contA.last_hidden_state
    stateA = cache_recurrent_snapshot(cacheA, layer_types)

    # "destroy" runtime state: drop refs + run a decoy forward with a throwaway cache
    del cacheA, outA
    decoy = torch.randint(0, cfg.vocab_size, (1, Lp), generator=torch.Generator().manual_seed(7)).to(dev)
    with torch.no_grad():
        _ = model(input_ids=decoy, use_cache=True)

    # Path B (restore S): continue the SAME cont -> hidden_B
    cacheB = copy.deepcopy(S)                              # <-- restore
    with torch.no_grad():
        contB = model(input_ids=cont, past_key_values=cacheB, use_cache=True)
    hidden_B = contB.last_hidden_state
    stateB = cache_recurrent_snapshot(cacheB, layer_types)

    hidden_diff = max_abs_diff(hidden_A, hidden_B)
    state_diff = 0.0
    for i in stateA:
        for k in stateA[i]:
            state_diff = max(state_diff, max_abs_diff(stateA[i][k], stateB[i][k]))

    # disk round-trip check
    loaded = torch.load(ckpt_path)
    disk_diff = max(max_abs_diff(loaded[k], disk[k]) for k in disk)

    def verdict(d):
        if d == 0.0:
            return "BIT_EXACT"
        if d <= 1e-4:
            return f"NUMERICALLY_EQUIVALENT_WITHIN_MEASURED_TOLERANCE(<=1e-4, observed {d:.2e})"
        return f"DIFFERENT(observed {d:.2e})"

    # ---------- extra rigor: cached-incremental decode vs full recompute ----------
    with torch.no_grad():
        full = model(input_ids=full_ids, use_cache=False)
    hidden_full_cont = full.last_hidden_state[:, Lp:, :]
    recompute_diff = max_abs_diff(hidden_A, hidden_full_cont)

    rnn02 = dict(
        experiment="checkpoint/restore + cached-vs-recompute",
        decoding="deterministic (no sampling); CPU fp32; eager",
        prefix_len=Lp, cont_len=Lc,
        restore_hidden_max_abs_diff=hidden_diff,
        restore_state_max_abs_diff=state_diff,
        restore_verdict=verdict(max(hidden_diff, state_diff)),
        disk_roundtrip_max_abs_diff=disk_diff,
        disk_roundtrip_verdict=verdict(disk_diff),
        cached_vs_fullrecompute_hidden_max_abs_diff=recompute_diff,
        cached_vs_fullrecompute_verdict=verdict(recompute_diff),
        tolerance_rationale=(
            "Restore/disk paths execute the identical op sequence on cloned tensors -> expected "
            "bit-exact (0.0). Cached-incremental vs full-recompute run DIFFERENT kernels "
            "(recurrent_gated_delta_rule per-step vs chunk_gated_delta_rule prefill) and full vs "
            "partial softmax attention, so fp non-associativity can produce a small nonzero diff; "
            "1e-4 fp32 threshold reflects that, not a semantic difference."),
    )
    with open(os.path.join(args.outdir, "rnn02_checkpoint_restore.json"), "w") as f:
        json.dump(rnn02, f, indent=2)

    # ---------- RNN-03 (optional): branching ----------
    rnn03 = None
    if args.branching:
        contX = cont
        g2 = torch.Generator().manual_seed(999)
        contY = torch.randint(0, cfg.vocab_size, (1, Lc), generator=g2).to(dev)
        cA = copy.deepcopy(S)
        cB = copy.deepcopy(S)
        with torch.no_grad():
            hX = model(input_ids=contX, past_key_values=cA, use_cache=True).last_hidden_state
            hY = model(input_ids=contY, past_key_values=cB, use_cache=True).last_hidden_state
        branch_sep = max_abs_diff(hX, hY)                  # should be > 0 (branches differ)
        # independently restore each branch and reproduce
        cA2 = copy.deepcopy(S); cB2 = copy.deepcopy(S)
        with torch.no_grad():
            hX2 = model(input_ids=contX, past_key_values=cA2, use_cache=True).last_hidden_state
            hY2 = model(input_ids=contY, past_key_values=cB2, use_cache=True).last_hidden_state
        rnn03 = dict(
            branch_separation_max_abs_diff=branch_sep,
            branch_A_reproduce_max_abs_diff=max_abs_diff(hX, hX2),
            branch_B_reproduce_max_abs_diff=max_abs_diff(hY, hY2),
            branches_independent=bool(branch_sep > 0),
            branch_A_reproducible=verdict(max_abs_diff(hX, hX2)),
            branch_B_reproducible=verdict(max_abs_diff(hY, hY2)),
        )
        with open(os.path.join(args.outdir, "rnn03_branching.json"), "w") as f:
            json.dump(rnn03, f, indent=2)

    print("=== RNN-01 inventory ===")
    print(open(os.path.join(args.outdir, "rnn01_inventory.txt")).read())
    print("=== RNN-02 ===")
    print(json.dumps(rnn02, indent=2))
    if rnn03:
        print("=== RNN-03 ===")
        print(json.dumps(rnn03, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--prefix-len", type=int, default=24)
    ap.add_argument("--cont-len", type=int, default=8)
    ap.add_argument("--branching", action="store_true")
    run(ap.parse_args())
