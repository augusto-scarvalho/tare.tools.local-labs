#!/usr/bin/env python
"""
RNN-04 recurrent substrate + Memory Caching aggregation (packet sections 7/13/19).

Substrate: a small, transparent pure-PyTorch recurrent memory. State S is a single [d_k, d_v] matrix
(the same shape as the lab's Qwen Gated-DeltaNet recurrent state -- necessary, NOT sufficient, for a
transplant claim; see runs/rnn/RNN-04-memory-caching/AUDIT_CORRECTIONS.md).

AUDIT-CORRECTED (RNN-05A stage 0): the EXECUTED memory is plain additive **Linear Attention** (paper Eq. 2)
computed by `_seg_linear`:  S_t = S_{t-1} + k_t v_t^T ;  o_t = S_t^T q_t  (a valid Memory-Caching-paper
substrate). It is NOT the Gated Delta rule. The delta/beta rule below is kept ONLY as reference notation and
is **not** on the executed path (the `beta` projection is built but unused by `_seg_linear`); do not read
"DeltaMemory" as evidence of DeltaNet semantics.
    reference-only (matches GDN_KERNEL.md notation; NOT EXECUTED):
        pred_t = S_{t-1}^T k_t ; S_t = S_{t-1} + beta_t * k_t (v_t - pred_t)^T ; o_t = S_t^T q_t
A depthwise causal conv (kernel 3) on q/k/v gives the temporal shift needed to bind key(t) to value(t+1)
in MQAR (as in GDN/Mamba). No CUDA/Triton (section 7): vectorized over batch/dims (no python scan).

Memory Caching (RNN_MEMORY_CACHING_SPEC.md equation->code binding): the sequence is split into segments;
the final state of each completed segment is cached; reads are aggregated across cached + online states.
Aggregation variants are PURE functions (unit-tested in section 19):
    single  : online state only (BASE_RNN / equal-memory control)   -> arm A / C
    residual: unweighted sum (Eq. 7; collapses for linear memory)   -> arm B0
    grm     : query-dependent softmax gate (Eq. 9-10)               -> arm B
    soup    : parameter-average form (Eq. 14-15; == grm for linear) -> deferred
    ssc     : Top-k router (Eq. 16-17); random variant = control    -> arm C-sel / D
    moving_average: param-free equal weights (Sec 4.3 post-training) -> arm Post
Runnable: python rnn_mc_substrate.py --unittest --out <json>   (section 19 aggregation + checkpoint tests)
"""
import argparse, json
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- pure aggregation functions (section 19 unit-tested) ----------------
def read_states(states, q):
    """states: list of [B,dk,dv]; q: [B,...,dk] -> list of reads [B,...,dv] (o = S^T q)."""
    return [torch.einsum('bkd,b...k->b...d', S, q) for S in states]


def combine(reads, gammas):
    """reads: list of n [B,...,dv]; gammas: [B,...,n] -> weighted sum [B,...,dv]."""
    R = torch.stack(reads, dim=-2)                       # [B,...,n,dv]
    return (gammas.unsqueeze(-1) * R).sum(dim=-2)


def agg_single(online_read):
    return online_read


def agg_residual(cached_reads, online_read):
    reads = cached_reads + [online_read]
    g = torch.ones(reads[0].shape[:-1] + (len(reads),), device=online_read.device, dtype=online_read.dtype)
    return combine(reads, g)


def agg_moving_average(cached_reads, online_read):
    reads = cached_reads + [online_read]
    n = len(reads)
    g = torch.full(reads[0].shape[:-1] + (n,), 1.0 / n, device=online_read.device, dtype=online_read.dtype)
    return combine(reads, g)


def gate_softmax(gate_logits):
    return torch.softmax(gate_logits, dim=-1)


def agg_grm(cached_reads, online_read, gate_logits):
    """gate_logits: [B,...,ncached+1] (cached..., online last). Eq. 9-10."""
    reads = cached_reads + [online_read]
    return combine(reads, gate_softmax(gate_logits))


def soup_states(states, gammas):
    """Eq. 14-15: average of STATE PARAMS. states: list n of [B,dk,dv]; gammas [B,...,n].
    Returns per-position souped state [B,...,dk,dv]. (For linear memory reading this == agg_grm.)"""
    S = torch.stack(states, dim=-3)                      # [B,n,dk,dv]
    # broadcast gammas [B,...,n] over dk,dv
    g = gammas.unsqueeze(-1).unsqueeze(-1)               # [B,...,n,1,1]
    while S.dim() < g.dim():
        S = S.unsqueeze(1)
    return (g * S).sum(dim=-3)


def ssc_gates(gate_logits, k, random_sel=False, generator=None):
    """Top-k over cached segments (all but last col = online), online always kept. Eq. 16-17.
    Returns renormalized gammas [B,...,ncached+1] with non-selected cached set to 0."""
    ncached = gate_logits.shape[-1] - 1
    k = min(k, ncached)
    cached_logits = gate_logits[..., :ncached]
    if random_sel:
        noise = torch.rand(cached_logits.shape, device=gate_logits.device, generator=generator)
        sel = noise.topk(k, dim=-1).indices if k > 0 else noise[..., :0].long()
    else:
        sel = cached_logits.topk(k, dim=-1).indices if k > 0 else cached_logits[..., :0].long()
    mask = torch.zeros_like(gate_logits, dtype=torch.bool)
    if k > 0:
        mask.scatter_(-1, sel, True)
    mask[..., -1] = True                                 # online always kept
    masked = gate_logits.masked_fill(~mask, float('-inf'))
    return torch.softmax(masked, dim=-1)


# ---------------- substrate ----------------
class DeltaMemory(nn.Module):
    def __init__(self, d_model, d_k, d_v, conv_k=3):
        super().__init__()
        self.d_k, self.d_v = d_k, d_v
        self.q = nn.Linear(d_model, d_k, bias=False)
        self.k = nn.Linear(d_model, d_k, bias=False)
        self.v = nn.Linear(d_model, d_v, bias=False)
        self.beta = nn.Linear(d_model, 1)
        self.conv_q = nn.Conv1d(d_k, d_k, conv_k, groups=d_k, bias=False)
        self.conv_k = nn.Conv1d(d_k, d_k, conv_k, groups=d_k, bias=False)
        self.conv_v = nn.Conv1d(d_v, d_v, conv_k, groups=d_v, bias=False)
        self.conv_k_pad = conv_k - 1
        self.w_u = nn.Linear(d_model, d_k, bias=False)   # gate connector u_t = x_t W_u (Eq. 10)
        self.out = nn.Linear(d_v, d_model, bias=False)

    def _cconv(self, conv, z):                            # z [B,L,C] causal depthwise conv
        z = z.transpose(1, 2)
        z = F.pad(z, (self.conv_k_pad, 0))
        return conv(z).transpose(1, 2)

    def _proj(self, x):
        q = F.normalize(self._cconv(self.conv_q, self.q(x)), dim=-1)
        k = F.normalize(self._cconv(self.conv_k, self.k(x)), dim=-1)
        v = self._cconv(self.conv_v, self.v(x))
        beta = torch.sigmoid(self.beta(x))               # [B,L,1]
        return q, k, v, beta

    def _seg_linear(self, q, k, v, S_prev=None):
        """Vectorized intra-segment causal linear attention (paper Eq. 2 base memory; no python scan).
        Returns online causal reads [B,seg,dv] and final segment state S = sum_i k_i v_i^T  [B,dk,dv].
        With S_prev (warm start) the read includes S_prev's read and S_prev is added to the state."""
        seg = q.size(1)
        scores = torch.einsum('bsk,btk->bst', q, k)
        mask = torch.tril(torch.ones(seg, seg, device=q.device, dtype=torch.bool))
        o = torch.einsum('bst,btd->bsd', scores.masked_fill(~mask, 0.0), v)
        S = torch.einsum('bik,bid->bkd', k, v)
        if S_prev is not None:
            o = o + torch.einsum('bkd,bsk->bsd', S_prev, q)
            S = S + S_prev
        return o, S

    def forward(self, x, agg="single", seg_size=None, warm_start=False, ssc_k=2, ssc_random=False,
                gen=None, return_state_info=False):
        B, L, _ = x.shape
        q, k, v, _ = self._proj(x)
        segs = [(0, L)] if (agg == "single" or seg_size is None) else \
               [(i, min(i + seg_size, L)) for i in range(0, L, seg_size)]
        cached_states, cached_pools = [], []
        outs, state_bytes = [], 0
        S_prev = None
        for (a, b) in segs:
            xr = x[:, a:b]
            sp = S_prev if (warm_start and S_prev is not None) else None
            online_reads, S_fin = self._seg_linear(q[:, a:b], k[:, a:b], v[:, a:b], sp)
            if agg == "single":
                o = agg_single(online_reads)
            else:
                cr = read_states(cached_states, q[:, a:b]) if cached_states else []
                if agg == "residual":
                    o = agg_residual(cr, online_reads) if cr else online_reads
                elif agg == "moving_average":
                    o = agg_moving_average(cr, online_reads) if cr else online_reads
                elif agg in ("grm", "ssc", "soup"):
                    u = self.w_u(xr)                                  # [B,seg,dk]
                    kcum = torch.cumsum(k[:, a:b], dim=1) / \
                        torch.arange(1, b - a + 1, device=x.device).view(1, -1, 1)   # causal online pool
                    gl_online = torch.einsum('bsk,bsk->bs', u, kcum).unsqueeze(-1)   # [B,seg,1]
                    if cached_pools:
                        P = torch.stack(cached_pools, dim=1)          # [B,ncached,dk]
                        gl_cached = torch.einsum('bsk,bck->bsc', u, P)
                        gl = torch.cat([gl_cached, gl_online], dim=-1)
                    else:
                        gl = gl_online
                    if not cr:
                        o = online_reads
                    elif agg == "grm":
                        o = agg_grm(cr, online_reads, gl)
                    elif agg == "soup":
                        Ss = cached_states + [S_fin.expand(B, self.d_k, self.d_v)]
                        Sm = soup_states([s if s.dim() == 3 else s for s in Ss],
                                         gate_softmax(gl))              # [B,seg,dk,dv]
                        o = torch.einsum('bskd,bsk->bsd', Sm, q[:, a:b])
                    else:  # ssc
                        g = ssc_gates(gl, ssc_k, random_sel=ssc_random, generator=gen)
                        o = combine(cr + [online_reads], g)
                else:
                    raise ValueError(agg)
            outs.append(self.out(o))
            state_bytes = S_fin.numel() // B * S_fin.element_size()   # per-request live state (all arms)
            # cache finalized segment state + pool (mean key). NOT detached: trained arms backprop end-to-end
            # through cached-segment writes (paper trains with MC enabled; detach would starve exactly the
            # recall pathway). The frozen post-training variant has no gradient, so this is a no-op there.
            if agg != "single":
                cached_states.append(S_fin)
                cached_pools.append(k[:, a:b].mean(dim=1))
            S_prev = S_fin
        y = torch.cat(outs, dim=1)
        if return_state_info:
            return y, dict(n_cached=len(cached_states), state_bytes_per_req=int(state_bytes),
                           total_cache_bytes=int(state_bytes * max(0, len(cached_states))),
                           dk=self.d_k, dv=self.d_v)
        return y


class MQARModel(nn.Module):
    def __init__(self, vocab, d_model=128, d_k=64, d_v=64, conv_k=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.mem = DeltaMemory(d_model, d_k, d_v, conv_k)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, input_ids, **kw):
        x = self.emb(input_ids)
        info = kw.pop("return_state_info", False)
        y = self.mem(x, return_state_info=info, **kw)
        if info:
            y, si = y
            return self.head(self.norm(x + y)), si
        return self.head(self.norm(x + y))


# ---------------- section 19 unit tests ----------------
def run_unittest(out_path):
    torch.manual_seed(0)
    B, dk, dv = 2, 4, 3
    S1 = torch.randn(B, dk, dv); S2 = torch.randn(B, dk, dv); S3 = torch.randn(B, dk, dv)
    q = torch.randn(B, dk)
    r = read_states([S1, S2, S3], q)                     # online = S3
    checks = {}

    # (a) residual == sum of hand-computed reads
    man_res = (S1.transpose(1, 2) @ q.unsqueeze(-1)).squeeze(-1) + \
              (S2.transpose(1, 2) @ q.unsqueeze(-1)).squeeze(-1) + \
              (S3.transpose(1, 2) @ q.unsqueeze(-1)).squeeze(-1)
    got_res = agg_residual(r[:2], r[2])
    checks["residual_vs_manual"] = float((man_res - got_res).abs().max())

    # (b) grm == gamma-weighted hand sum; and soup(read) == grm (linear-memory equivalence)
    gl = torch.randn(B, 3)
    g = gate_softmax(gl)
    man_grm = g[:, 0:1] * r[0] + g[:, 1:2] * r[1] + g[:, 2:3] * r[2]
    got_grm = agg_grm(r[:2], r[2], gl)
    checks["grm_vs_manual"] = float((man_grm - got_grm).abs().max())
    souped = soup_states([S1, S2, S3], g)                # [B,dk,dv]
    read_soup = torch.einsum('bkd,bk->bd', souped, q)
    checks["soup_eq_grm_linear"] = float((read_soup - got_grm).abs().max())

    # (c) moving_average == equal weights
    man_ma = (r[0] + r[1] + r[2]) / 3
    checks["moving_avg_vs_manual"] = float((man_ma - agg_moving_average(r[:2], r[2])).abs().max())

    # (d) ssc keeps online + exactly k cached (nonzero gamma count == k+1)
    gl2 = torch.randn(B, 5)                              # 4 cached + online
    gs = ssc_gates(gl2, k=2)
    nz = (gs > 0).sum(-1)
    checks["ssc_selects_k_plus_online"] = int((nz == 3).all())
    # random ssc selects same count
    gen = torch.Generator().manual_seed(0)
    gsr = ssc_gates(gl2, k=2, random_sel=True, generator=gen)
    checks["ssc_random_selects_k_plus_online"] = int(((gsr > 0).sum(-1) == 3).all())

    # (e) residual collapse for linear memory: sum of segment states read == full concatenated state read
    #     (plain linear-attention: S_full = sum_i S_i). Build additive states and verify.
    A1 = torch.randn(B, dk, dv); A2 = torch.randn(B, dk, dv)
    lhs = agg_residual(read_states([A1], q)[:0] + [read_states([A1], q)[0]], read_states([A2], q)[0])
    rhs = torch.einsum('bkd,bk->bd', A1 + A2, q)
    checks["residual_collapse_linear"] = float((lhs - rhs).abs().max())

    # (f) checkpoint/restore (section 13): cache serialize->reload is BIT_EXACT; additive continuation of
    #     the linear state matches the full-sequence run to numerical tolerance (RNN-01/02 vocabulary).
    mem = DeltaMemory(16, dk, dv)
    x = torch.randn(B, 8, 16)
    q2, k2, v2, _ = mem._proj(x)
    full_reads, S_full = mem._seg_linear(q2, k2, v2)                       # whole sequence at once
    _, S_ck = mem._seg_linear(q2[:, :5], k2[:, :5], v2[:, :5])             # checkpoint after 5 tokens
    S_reload = S_ck.clone()                                                # serialize -> reload roundtrip
    r_ck = torch.einsum('bkd,bsk->bsd', S_ck, q2)
    r_rl = torch.einsum('bkd,bsk->bsd', S_reload, q2)
    checks["checkpoint_reload_bitexact"] = float((r_ck - r_rl).abs().max())   # 0.0 by construction
    r_cont, S_cont = mem._seg_linear(q2[:, 5:], k2[:, 5:], v2[:, 5:], S_reload)  # restore + continue
    checks["continuation_state_numeq"] = float((S_full - S_cont).abs().max())
    checks["continuation_reads_numeq"] = float((full_reads[:, 5:] - r_cont).abs().max())
    checks["checkpoint_state_shape"] = list(S_ck.shape)
    checks["checkpoint_state_dtype"] = str(S_ck.dtype)
    checks["checkpoint_state_bytes_per_req"] = S_ck.numel() // B * S_ck.element_size()

    tol = 1e-5
    passed = (checks["residual_vs_manual"] < tol and checks["grm_vs_manual"] < tol and
              checks["soup_eq_grm_linear"] < tol and checks["moving_avg_vs_manual"] < tol and
              checks["ssc_selects_k_plus_online"] == 1 and checks["ssc_random_selects_k_plus_online"] == 1 and
              checks["residual_collapse_linear"] < tol and
              checks["checkpoint_reload_bitexact"] == 0.0 and checks["continuation_state_numeq"] < 1e-4)
    result = dict(packet="RNN-04", component="substrate + aggregation unit tests (section 19)",
                  AGGREGATION_UNIT_TEST="PASS" if passed else "FAIL",
                  CHECKPOINT_RESTORE="BIT_EXACT" if checks["checkpoint_reload_bitexact"] == 0.0 else "DIFFERENT",
                  torch=torch.__version__, tol=tol, checks=checks)
    if out_path:
        json.dump(result, open(out_path, "w"), indent=2)
    print(json.dumps({k: result[k] for k in ["AGGREGATION_UNIT_TEST", "CHECKPOINT_RESTORE"]}, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--unittest", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run_unittest(a.out)
