#!/usr/bin/env python
"""
RNN-05B substrate: a nested Linear-Attention / DeltaNet / Gated-DeltaNet family with a COMPLETE,
checkpointable module state (recurrent matrix + causal-conv boundary), plus Memory-Caching aggregation.

Authority (see runs/rnn/RNN-05B-delta-gdn/RNN05B_DELTA_SEMANTICS.md):
  * GDN_KERNEL.md  (ggml/llama.cpp GDN recurrence; out = S^T x; scalar decay d_t=exp(g_t))
  * scratchpad/modeling_qwen3_next.py  (HF Qwen3-Next):
      - torch_recurrent_gated_delta_rule  -> ported here as `delta_scan` (SEQUENTIAL ground truth)
      - torch_chunk_gated_delta_rule      -> ported here as `delta_chunked` (chunk-parallel; fast path)
  * RNN-01 RNN_STATE_INVENTORY.json: real Qwen linear layer cache = {recurrent_states, conv_states}.

Nested family (one recurrence, two switches):
  LA : decay=1,          u_t = v_t                 (additive; S = sum_i k_i (x) v_i)
  DN : decay=1,          u_t = beta_t(v_t - S^T k_t)
  GDN: decay=exp(g_t),   u_t = beta_t(v_t - S^T k_t)
Read (all): o_t = S^T q_t (after the write). q,k L2-normalized; q scaled by 1/sqrt(d_k). g_t<=0 per-head scalar.

Parity oracle (FLA absent in the venv): our SEQUENTIAL scan vs our CHUNK-PARALLEL path (no shared code).
No CUDA/Triton (packet §4). Pure PyTorch, vectorized over batch.

Runnable: python rnn_delta_substrate.py --selftest --out <json>
  -> reference parity (LA/DN/GDN), full-module checkpoint/restore, request isolation, collapsibility.
"""
import argparse, json, hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse the RNN-04 pure aggregation functions (unit-tested there) for Memory Caching
from rnn_mc_substrate import read_states, agg_moving_average, agg_grm, gate_softmax  # noqa: F401

MODES = ("la", "dn", "gdn")


# ============================ recurrences ============================
def delta_scan(mode, q, k, v, g, beta, S0=None):
    """SEQUENTIAL ground truth. Ported line-for-line from Qwen torch_recurrent_gated_delta_rule.
    q,k,v: [B,L,d]; g,beta: [B,L] (per-head scalars). q,k assumed already L2-normalized; q already scaled.
    Returns (o [B,L,dv], S_final [B,dk,dv]). mode in {la,dn,gdn}."""
    B, L, dk = k.shape
    dv = v.shape[-1]
    S = torch.zeros(B, dk, dv, device=k.device, dtype=k.dtype) if S0 is None else S0.clone()
    outs = []
    use_gate = (mode == "gdn")
    use_delta = (mode in ("dn", "gdn"))
    for t in range(L):
        q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]                       # [B,d]
        if use_gate:
            S = S * g[:, t].exp().view(B, 1, 1)                         # decay whole state
        if use_delta:
            kv = (S * k_t.unsqueeze(-1)).sum(dim=1)                     # S^T k_t -> [B,dv]
            u = (v_t - kv) * beta[:, t].unsqueeze(-1)                   # beta*(v - S^T k)
        else:
            u = v_t                                                     # LA: plain value write
        S = S + k_t.unsqueeze(-1) * u.unsqueeze(1)                      # S += k (x) u
        outs.append((S * q_t.unsqueeze(-1)).sum(dim=1))                 # S^T q_t
    return torch.stack(outs, dim=1), S


def delta_chunked(q, k, v, g, beta, chunk_size=32, S0=None):
    """CHUNK-PARALLEL path for the delta family (DN via g=0). Ported from Qwen torch_chunk_gated_delta_rule
    (single head; scale/l2norm done by the caller). Returns (o [B,L,dv], S_final)."""
    B, L, dk = k.shape
    dv = v.shape[-1]
    pad = (chunk_size - L % chunk_size) % chunk_size
    pv = lambda x: F.pad(x, (0, 0, 0, pad))                            # pad [B,L,d] along L
    q, k, v = pv(q), pv(k), pv(v)
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    T = L + pad
    nc = T // chunk_size
    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    ch = lambda x: x.reshape(B, nc, chunk_size, x.shape[-1])
    q, k, v, k_beta, v_beta = ch(q), ch(k), ch(v), ch(k_beta), ch(v_beta)
    g = g.reshape(B, nc, chunk_size).cumsum(dim=-1)                     # cumulative log-decay within chunk
    eye = torch.eye(chunk_size, dtype=q.dtype, device=q.device)
    incl = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)  # >= diag
    decay = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp()).tril()  # [B,nc,C,C], exp(g_t-g_r), r<=t
    attn = -((k_beta @ k.transpose(-1, -2)) * decay).masked_fill(incl, 0)
    for i in range(1, chunk_size):                                     # unit lower-tri inverse (fwd subst)
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + eye
    value = attn @ v_beta                                              # effective per-token writes U
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    S = torch.zeros(B, dk, dv, device=q.device, dtype=q.dtype) if S0 is None else S0.clone().to(q.dtype)
    out = torch.zeros_like(value)
    for i in range(nc):
        q_i, k_i, u_i = q[:, i], k[:, i], value[:, i]
        a_intra = (q_i @ k_i.transpose(-1, -2)) * decay[:, i]          # intra-chunk (causal incl diag)
        v_prime = k_cumdecay[:, i] @ S
        u_new = u_i - v_prime
        a_inter = (q_i * g[:, i, :, None].exp()) @ S                   # read carried state, decayed
        out[:, i] = a_inter + a_intra @ u_new
        S = S * g[:, i, -1, None, None].exp() + \
            (k_i * (g[:, i, -1, None] - g[:, i]).exp()[..., None]).transpose(-1, -2) @ u_new
    return out.reshape(B, T, dv)[:, :L], S


def la_chunked(q, k, v, chunk_size=32, S0=None):
    """CHUNK-PARALLEL Linear Attention (additive; matches RNN-04 _seg_linear generalized to carried state).
    q already scaled + L2-normalized like the others (kept consistent for a fair family). Returns (o, S)."""
    B, L, dk = k.shape
    dv = v.shape[-1]
    S = torch.zeros(B, dk, dv, device=q.device, dtype=q.dtype) if S0 is None else S0.clone()
    outs = []
    for a in range(0, L, chunk_size):
        b = min(a + chunk_size, L)
        qc, kc, vc = q[:, a:b], k[:, a:b], v[:, a:b]
        n = b - a
        tril = torch.tril(torch.ones(n, n, dtype=torch.bool, device=q.device))
        intra = (qc @ kc.transpose(-1, -2)).masked_fill(~tril, 0.0) @ vc
        inter = torch.einsum('bkd,bsk->bsd', S, qc)
        outs.append(inter + intra)
        S = S + torch.einsum('bik,bid->bkd', kc, vc)
    return torch.cat(outs, dim=1), S


def run_recurrence(mode, q, k, v, g, beta, chunk_size=32, S0=None, path="chunked"):
    """Dispatch. path in {scan, chunked}. LA uses la_chunked; DN uses delta_chunked(g=0); GDN delta_chunked."""
    if path == "scan":
        return delta_scan(mode, q, k, v, g, beta, S0)
    if mode == "la":
        return la_chunked(q, k, v, chunk_size, S0)
    gg = torch.zeros_like(g) if mode == "dn" else g
    return delta_chunked(q, k, v, gg, beta, chunk_size, S0)


# ============================ block (projections + stateful conv) ============================
class DeltaBlock(nn.Module):
    """Projects x -> q,k,v,g,beta with a single depthwise causal conv over [q;k;v] (kernel K), SiLU, then
    L2norm(q,k) and scale q by 1/sqrt(d_k). g=-softplus(w_g x) (GDN) ; beta=sigmoid(w_b x). The conv keeps a
    boundary buffer so the COMPLETE module state = {recurrent S, conv_state} (mirrors Qwen conv_states)."""

    def __init__(self, d_model, d_k, d_v, conv_k=4):
        super().__init__()
        self.d_k, self.d_v, self.conv_k = d_k, d_v, conv_k
        self.conv_dim = d_k + d_k + d_v
        self.in_proj = nn.Linear(d_model, self.conv_dim, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, conv_k, groups=self.conv_dim, bias=False)
        self.w_g = nn.Linear(d_model, 1)                                # per-head scalar log-decay (GDN)
        self.w_b = nn.Linear(d_model, 1)                                # per-head scalar beta
        # GDN/Mamba-style init: decay ~= 1 at start (near-lossless) so the model can RETAIN, then learn to
        # forget. Bias -4.5 -> softplus~0.011 -> g~-0.011 -> decay~0.989/token; small weight keeps it gentle.
        # Without this, default init gives decay~0.5/token = catastrophic forgetting -> GDN can't do recall.
        nn.init.constant_(self.w_g.bias, -4.5)
        with torch.no_grad():
            self.w_g.weight.mul_(0.3)
        self.w_u = nn.Linear(d_model, d_k, bias=False)                  # MC GRM gate connector (reader)
        self.out = nn.Linear(d_v, d_model, bias=False)
        self.scale = d_k ** -0.5

    def _causal_conv(self, z, conv_state=None):
        """z: [B,L,C] -> conv output [B,L,C] + new_state [B,C,K-1]. If conv_state given, left-context is it
        (continuation); else zero left pad (fresh)."""
        zt = z.transpose(1, 2)                                          # [B,C,L]
        if conv_state is None:
            zp = F.pad(zt, (self.conv_k - 1, 0))
        else:
            zp = torch.cat([conv_state, zt], dim=-1)
        y = self.conv1d(zp)                                            # valid conv, length L
        new_state = zp[..., -(self.conv_k - 1):].detach() if (self.conv_k > 1) else \
            torch.zeros(z.shape[0], self.conv_dim, 0, device=z.device, dtype=z.dtype)
        return y.transpose(1, 2), new_state

    def _finish(self, x, conved):
        q, k, v = torch.split(conved, [self.d_k, self.d_k, self.d_v], dim=-1)
        q = F.normalize(q, dim=-1) * self.scale
        k = F.normalize(k, dim=-1)
        g = -F.softplus(self.w_g(x)).squeeze(-1)                        # [B,L], <=0
        beta = torch.sigmoid(self.w_b(x)).squeeze(-1)                   # [B,L]
        return q, k, v, g, beta

    def proj(self, x, conv_state=None):
        """Full projection with an optional conv boundary buffer. Returns q,k,v,g,beta,new_conv_state."""
        conved, new_state = self._causal_conv(self.in_proj(x), conv_state)   # conv over projections
        conved = F.silu(conved)                                              # activation after conv (Qwen order)
        q, k, v, g, beta = self._finish(x, conved)
        return q, k, v, g, beta, new_state


class MQARDeltaModel(nn.Module):
    def __init__(self, vocab, d_model=128, d_k=24, d_v=24, conv_k=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.blk = DeltaBlock(d_model, d_k, d_v, conv_k)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)
        self.d_k, self.d_v = d_k, d_v

    def project(self, ids, conv_state=None):
        x = self.emb(ids)
        q, k, v, g, beta, ns = self.blk.proj(x, conv_state)
        return x, q, k, v, g, beta, ns

    def readout(self, x, o):
        return self.head(self.norm(x + self.blk.out(o)))

    def forward(self, ids, mode, path="chunked", chunk_size=32):
        x, q, k, v, g, beta, _ = self.project(ids)
        o, _ = run_recurrence(mode, q, k, v, g, beta, chunk_size, None, path)
        return self.readout(x, o)


# ============================ complete-state checkpoint (packet §7) ============================
def full_state_bytes(S, conv_state):
    return int(S.numel() // S.shape[0] * S.element_size() +
               conv_state.numel() // conv_state.shape[0] * conv_state.element_size())


@torch.no_grad()
def checkpoint_restore_full_module(model, ids, mode, split, chunk_size=32):
    """HARD gate (packet §7). Prefix -> serialize COMPLETE state {S, conv_state} -> destroy -> restore ->
    feed ONLY continuation tokens (conv rebuilt from restored conv_state, recurrence from restored S) ->
    compare against the uninterrupted run. No precompute of future q/k/v from the prefix."""
    B, L = ids.shape
    pre, cont = ids[:, :split], ids[:, split:]

    # --- uninterrupted reference ---
    x_full, q, k, v, g, beta, _ = model.project(ids)
    o_full, S_full = run_recurrence(mode, q, k, v, g, beta, chunk_size, None, "scan")
    y_full = model.readout(x_full, o_full)

    # --- prefix pass: build COMPLETE state ---
    x_pre, qp, kp, vp, gp, bp, conv_state = model.project(pre, None)
    o_pre, S_pre = run_recurrence(mode, qp, kp, vp, gp, bp, chunk_size, None, "scan")

    # --- serialize -> destroy -> restore (bit-exact round trip of BOTH components) ---
    import numpy as np
    blobS = S_pre.detach().cpu().numpy().tobytes()
    blobC = conv_state.detach().cpu().numpy().tobytes()
    del S_pre, conv_state
    S_re = torch.from_numpy(np.frombuffer(blobS, dtype=np.float32).copy()).view(B, model.d_k, model.d_v).to(ids.device)
    C_re = torch.from_numpy(np.frombuffer(blobC, dtype=np.float32).copy()).view(B, model.blk.conv_dim,
                                                                                model.blk.conv_k - 1).to(ids.device)

    # --- continuation: ONLY continuation tokens; conv uses restored boundary; recurrence from restored S ---
    x_c, qc, kc, vc, gc, bc, _ = model.project(cont, C_re)
    o_c, S_c = run_recurrence(mode, qc, kc, vc, gc, bc, chunk_size, S_re, "scan")
    y_c = model.readout(x_c, o_c)

    reload_bitexact = max(float((torch.from_numpy(np.frombuffer(blobS, dtype=np.float32).copy()).view_as(S_re)
                                 .to(ids.device) - S_re).abs().max()), 0.0)
    out_maxabs = float((y_full[:, split:] - y_c).abs().max())
    state_maxabs = float((S_full - S_c).abs().max())
    tol = 1e-4
    status = "BIT_EXACT" if (out_maxabs == 0.0 and state_maxabs == 0.0) else \
             ("NUMERICALLY_EQUIVALENT" if (out_maxabs < tol and state_maxabs < tol) else "FAILED")
    return dict(mode=mode, split=split, reload_bitexact=reload_bitexact,
                continuation_out_maxabs=out_maxabs, continuation_state_maxabs=state_maxabs, tol=tol,
                complete_state_bytes_per_req=full_state_bytes(S_full, C_re),
                components=["recurrent_state S", "conv_state (kernel-1 boundary)"],
                FULL_MODULE_CHECKPOINT_RESTORE=status)


# ============================ parity gate (packet §5-6) ============================
@torch.no_grad()
def reference_parity(model, ids, mode, chunk_size=32):
    """scan (ground truth) vs chunked, at: single step, small seq, chunked seq, incremental decode."""
    x, q, k, v, g, beta, _ = model.project(ids)
    B, L, _ = q.shape
    res = {}
    # (1) single step
    o1s, S1s = run_recurrence(mode, q[:, :1], k[:, :1], v[:, :1], g[:, :1], beta[:, :1], chunk_size, None, "scan")
    o1c, S1c = run_recurrence(mode, q[:, :1], k[:, :1], v[:, :1], g[:, :1], beta[:, :1], chunk_size, None, "chunked")
    res["single_step"] = max(float((o1s - o1c).abs().max()), float((S1s - S1c).abs().max()))
    # (2) small seq (< chunk)
    sl = min(8, L)
    oss, Sss = run_recurrence(mode, q[:, :sl], k[:, :sl], v[:, :sl], g[:, :sl], beta[:, :sl], chunk_size, None, "scan")
    osc, Ssc = run_recurrence(mode, q[:, :sl], k[:, :sl], v[:, :sl], g[:, :sl], beta[:, :sl], chunk_size, None, "chunked")
    res["small_seq"] = max(float((oss - osc).abs().max()), float((Sss - Ssc).abs().max()))
    # (3) chunked full seq
    ofs, Sfs = run_recurrence(mode, q, k, v, g, beta, chunk_size, None, "scan")
    ofc, Sfc = run_recurrence(mode, q, k, v, g, beta, chunk_size, None, "chunked")
    res["chunked_full"] = max(float((ofs - ofc).abs().max()), float((Sfs - Sfc).abs().max()))
    # (4) incremental recurrent decode: feed one token at a time (scan carrying S) == full scan
    S = None
    dec = []
    for t in range(L):
        ot, S = run_recurrence(mode, q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1], g[:, t:t + 1], beta[:, t:t + 1],
                               chunk_size, S, "scan")
        dec.append(ot)
    dec = torch.cat(dec, dim=1)
    res["incremental_decode_vs_scan"] = max(float((dec - ofs).abs().max()), float((S - Sfs).abs().max()))
    tol = 1e-4
    status = "PASS" if all(vv < tol for vv in res.values()) else "FAIL"
    # SCOPE (audit §3): both paths are LOCAL ports of the pinned Qwen recurrence (scan <- torch_recurrent,
    # chunked <- torch_chunk). FLA/upstream executable is NOT invoked -> this is dual-implementation parity,
    # not upstream-executable parity.
    return dict(mode=mode, chunk_size=chunk_size, tol=tol, maxabs=res, PARITY=status,
                REFERENCE_PARITY_SCOPE="LOCAL_PORTS_ONLY",
                LOCAL_DUAL_IMPLEMENTATION_PARITY=status,
                UPSTREAM_EXECUTABLE_PARITY="NOT_QUALIFIED")


# ============================ request isolation (packet §8) ============================
@torch.no_grad()
def request_isolation(model, idsA, idsB, mode, chunk_size=32):
    """B's output must be identical whether A ran before or not (no global singleton state); and two
    branches restored from the SAME prefix checkpoint reproduce the reference."""
    yB_alone = model(idsB, mode, "scan", chunk_size)
    _ = model(idsA, mode, "scan", chunk_size)                          # run A in between
    yB_after = model(idsB, mode, "scan", chunk_size)
    iso = float((yB_alone - yB_after).abs().max())
    # branch restore: checkpoint a shared prefix, restore twice, continue with two different suffixes
    L = idsB.shape[1]
    split = L // 2
    r1 = checkpoint_restore_full_module(model, idsB, mode, split, chunk_size)
    r2 = checkpoint_restore_full_module(model, torch.cat([idsB[:, :split], idsA[:, split:]], 1), mode, split, chunk_size)
    branch_ok = (r1["FULL_MODULE_CHECKPOINT_RESTORE"] in ("BIT_EXACT", "NUMERICALLY_EQUIVALENT") and
                 r2["FULL_MODULE_CHECKPOINT_RESTORE"] in ("BIT_EXACT", "NUMERICALLY_EQUIVALENT"))
    return dict(mode=mode, B_output_invariance_maxabs=iso,
                REQUEST_STATE_ISOLATION="PASS" if iso == 0.0 else "FAIL",
                BRANCH_RESTORE="PASS" if branch_ok else "FAIL")


# ============================ single-blob fork (audit §6) ============================
@torch.no_grad()
def single_blob_fork(model, ids, mode, split, chunk_size=32):
    """Prove two continuations can be restored from ONE serialized complete-state blob {S, conv_state}
    (not two separate prefix recomputations). Serialize once, restore twice, feed two DIFFERENT suffixes;
    each must match its own uninterrupted full run, and the two branches must actually diverge."""
    import numpy as np
    B, L = ids.shape
    pre = ids[:, :split]
    contA = ids[:, split:]
    contB = ids.flip(0)[:, split:]                                     # a different, deterministic suffix
    # build the complete-state blob ONCE from the shared prefix
    _, qp, kp, vp, gp, bp, conv_state = model.project(pre, None)
    _, S_pre = run_recurrence(mode, qp, kp, vp, gp, bp, chunk_size, None, "scan")
    blobS = S_pre.detach().cpu().numpy().tobytes()
    blobC = conv_state.detach().cpu().numpy().tobytes()
    del S_pre, conv_state, qp, kp, vp, gp, bp

    def restore():
        S = torch.from_numpy(np.frombuffer(blobS, dtype=np.float32).copy()).view(B, model.d_k, model.d_v).to(ids.device)
        C = torch.from_numpy(np.frombuffer(blobC, dtype=np.float32).copy()).view(
            B, model.blk.conv_dim, model.blk.conv_k - 1).to(ids.device)
        return S, C

    def cont(suffix):
        S, C = restore()                                              # SAME blob each time
        x, q, k, v, g, b, _ = model.project(suffix, C)
        o, _ = run_recurrence(mode, q, k, v, g, b, chunk_size, S, "scan")
        return model.readout(x, o)

    yA, yB = cont(contA), cont(contB)
    fullA = model(torch.cat([pre, contA], 1), mode, "scan", chunk_size)[:, split:]
    fullB = model(torch.cat([pre, contB], 1), mode, "scan", chunk_size)[:, split:]
    errA, errB = float((yA - fullA).abs().max()), float((yB - fullB).abs().max())
    diverge = float((yA - yB).abs().max())
    tol = 1e-4
    ok = (errA < tol and errB < tol and diverge > tol)
    return dict(mode=mode, split=split, branchA_err_vs_full=errA, branchB_err_vs_full=errB,
                branch_divergence=diverge, tol=tol,
                SINGLE_BLOB_FORK_BRANCHING="PASS" if ok else "FAIL",
                note="two suffixes restored from ONE {S,conv} blob; each matches its full run; branches differ")


# ============================ collapsibility (packet §15) ============================
@torch.no_grad()
def collapsibility(model, ids, mode, n_seg=4, chunk_size=32):
    """Does a simple composition of INDEPENDENT segment states reproduce the true final state? For LA yes
    (additive); for DN/GDN expected no. Diagnostic distance only (not semantic equivalence)."""
    x, q, k, v, g, beta, _ = model.project(ids)
    B, L, _ = q.shape
    seg = L // n_seg
    # true final state over the whole sequence (continuous)
    _, S_true = run_recurrence(mode, q, k, v, g, beta, chunk_size, None, "scan")
    # independent per-segment states (each reset to zero)
    states = []
    for i in range(n_seg):
        a, b = i * seg, (i + 1) * seg if i < n_seg - 1 else L
        _, Si = run_recurrence(mode, q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b], chunk_size, None, "scan")
        states.append(Si)
    stack = torch.stack(states, dim=0)                                 # [n_seg,B,dk,dv]
    denom = S_true.abs().mean().clamp_min(1e-8)
    comps = {
        "sum": stack.sum(0),
        "mean": stack.mean(0),
        "last": states[-1],
    }
    rel = {name: float((c - S_true).abs().mean() / denom) for name, c in comps.items()}
    # best least-squares scalar-weighted combination (still a *linear* composition of independent states)
    A = stack.reshape(n_seg, -1).T                                     # [BDkDv, n_seg]
    tgt = S_true.reshape(-1, 1)
    w = torch.linalg.lstsq(A, tgt).solution
    rel["weighted_lstsq"] = float((A @ w - tgt).abs().mean() / denom)
    collapses = rel["sum"] < 1e-3
    return dict(mode=mode, n_seg=n_seg, rel_error_vs_final_state=rel,
                ADDITIVE_COLLAPSE="YES" if collapses else "NO",
                note="rel = mean|comp - S_final| / mean|S_final|; LA additive -> sum~0; DN/GDN expected >0")


# ============================ state inventory (packet §6) ============================
@torch.no_grad()
def state_inventory(model, mode, L=256, seg=32, B=1):
    """Enumerate ALL sequence-owned mutable state of the COMPLETE module (not just the matrix). Mirrors the
    real Qwen linear-layer cache {recurrent_states, conv_states} (RNN-01)."""
    blk = model.blk
    dk, dv, K = model.d_k, model.d_v, blk.conv_k
    n_ckpt = (L + seg - 1) // seg
    inv = [
        dict(name="recurrent_state_S", owner="delta recurrence", shape=[B, dk, dv], dtype="float32",
             bytes=dk * dv * 4, init="zeros", update_location="delta_scan/delta_chunked/la_chunked",
             reset_semantics="reset to 0 (independent) or carried (continuous)",
             serialization="raw fp32 tobytes -> restore view", batch_request_dim=0,
             sequence_owned=True, note="the [d_k,d_v] matrix; additive for LA, non-additive for DN/GDN"),
        dict(name="conv_state", owner="depthwise causal conv (in-proj q;k;v)", shape=[B, blk.conv_dim, K - 1],
             dtype="float32", bytes=blk.conv_dim * (K - 1) * 4, init="zeros left-pad",
             update_location="DeltaBlock._causal_conv", reset_semantics="last (K-1) projected cols",
             serialization="raw fp32 tobytes -> restore view", batch_request_dim=0, sequence_owned=True,
             note="the (kernel-1) boundary window; RNN-05A did NOT serialize this -> its FULL_MODULE gap"),
        dict(name="position/offset", owner="none (RoPE/positional not used in toy)", shape=[], dtype="n/a",
             bytes=0, init="n/a", update_location="n/a", reset_semantics="n/a", serialization="n/a",
             batch_request_dim=None, sequence_owned=False, note="no explicit position state in this substrate"),
        dict(name="normalization_buffers", owner="LayerNorm(final) / F.normalize(q,k)", shape=[], dtype="n/a",
             bytes=0, init="n/a", update_location="stateless", reset_semantics="n/a", serialization="n/a",
             batch_request_dim=None, sequence_owned=False,
             note="LayerNorm is parametric-but-not-sequence-owned; q/k L2norm is stateless -> NOT cache state"),
    ]
    live_bytes = sum(i["bytes"] for i in inv if i["sequence_owned"])
    return dict(mode=mode, d_k=dk, d_v=dv, conv_dim=blk.conv_dim, conv_kernel=K, seq_len=L, seg=seg,
                n_checkpoints=n_ckpt, complete_live_state_bytes_per_req=live_bytes,
                historical_matrix_cache_bytes_at_full=dk * dv * 4 * (n_ckpt - 1),
                sequence_owned_components=[i["name"] for i in inv if i["sequence_owned"]],
                maps_to_qwen="{recurrent_state_S -> recurrent_states, conv_state -> conv_states} per RNN-01",
                inventory=inv)


# ============================ selftest CLI ============================
def _mk_model(vocab=200, seed=0):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return MQARDeltaModel(vocab).to(dev)


def run_selftest(out_path, chunk_size=32):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _mk_model()
    g = torch.Generator(device="cpu").manual_seed(1)
    ids = torch.randint(0, 200, (4, 96), generator=g).to(dev)
    idsB = torch.randint(0, 200, (4, 96), generator=torch.Generator().manual_seed(2)).to(dev)
    R = {"torch": torch.__version__, "device": dev, "chunk_size": chunk_size}
    R["parity"] = {mode: reference_parity(m, ids, mode, chunk_size) for mode in MODES}
    R["full_module_checkpoint"] = {mode: checkpoint_restore_full_module(m, ids, mode, 48, chunk_size) for mode in MODES}
    R["request_isolation"] = {mode: request_isolation(m, ids, idsB, mode, chunk_size) for mode in MODES}
    R["collapsibility"] = {mode: collapsibility(m, ids, mode, 4, chunk_size) for mode in MODES}
    R["PARITY_ALL"] = "PASS" if all(R["parity"][x]["PARITY"] == "PASS" for x in MODES) else "FAIL"
    R["FULL_MODULE_ALL"] = "PASS" if all(
        R["full_module_checkpoint"][x]["FULL_MODULE_CHECKPOINT_RESTORE"] in ("BIT_EXACT", "NUMERICALLY_EQUIVALENT")
        for x in MODES) else "FAIL"
    R["ISOLATION_ALL"] = "PASS" if all(
        R["request_isolation"][x]["REQUEST_STATE_ISOLATION"] == "PASS" and
        R["request_isolation"][x]["BRANCH_RESTORE"] == "PASS" for x in MODES) else "FAIL"
    summary = {k: R[k] for k in ["PARITY_ALL", "FULL_MODULE_ALL", "ISOLATION_ALL"]}
    summary["collapse"] = {mode: R["collapsibility"][mode]["ADDITIVE_COLLAPSE"] for mode in MODES}
    print(json.dumps(summary, indent=2))
    if out_path:
        json.dump(R, open(out_path, "w"), indent=2)
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--forktest", default=None, metavar="JSON",
                    help="single-blob fork branching test (audit §6); CPU-safe, seconds")
    ap.add_argument("--inventory", default=None, metavar="JSON", help="emit RNN05B_STATE_INVENTORY.json (CPU)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--dk", type=int, default=64)
    a = ap.parse_args()
    if a.inventory:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""                        # CPU only (no GPU contention)
        torch.manual_seed(0)
        m = MQARDeltaModel(200, d_model=128, d_k=a.dk, d_v=a.dk, conv_k=4)
        inv = {mode: state_inventory(m, mode, L=256, seg=32) for mode in MODES}
        json.dump(inv, open(a.inventory, "w"), indent=2)
        print(json.dumps({k: inv["gdn"][k] for k in ["complete_live_state_bytes_per_req",
                          "sequence_owned_components", "maps_to_qwen"]}, indent=2))
    elif a.forktest:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""                        # CPU (seconds; no training)
        torch.manual_seed(0)
        m = MQARDeltaModel(200, d_model=128, d_k=a.dk, d_v=a.dk, conv_k=4)
        gcpu = torch.Generator().manual_seed(3)
        ids = torch.randint(0, 200, (6, 96), generator=gcpu)
        R = {mode: single_blob_fork(m, ids, mode, 48, a.chunk) for mode in MODES}
        R["SINGLE_BLOB_FORK_ALL"] = "PASS" if all(
            R[x]["SINGLE_BLOB_FORK_BRANCHING"] == "PASS" for x in MODES) else "FAIL"
        json.dump(R, open(a.forktest, "w"), indent=2)
        print(json.dumps({"SINGLE_BLOB_FORK_ALL": R["SINGLE_BLOB_FORK_ALL"],
                          **{m: R[m]["SINGLE_BLOB_FORK_BRANCHING"] for m in MODES}}, indent=2))
    elif a.selftest:
        run_selftest(a.out, a.chunk)
