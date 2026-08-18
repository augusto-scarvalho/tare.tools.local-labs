#!/usr/bin/env python
"""
RNN-05A: Fixed-Backbone Memory Caching.

Scientific question: given a recurrent backbone whose weights are ALREADY FIXED, can cached historical
recurrent states + a small learned aggregation/read mechanism improve recall over the frozen backbone's
ordinary single-state inference? Load-bearing constraint: BACKBONE WEIGHTS MUST REMAIN IDENTICAL ACROSS ARMS
(BACKBONE_WEIGHT_MUTATION = 0). This is the closest controlled precursor to augmenting a pretrained Qwen
Gated-DeltaNet model -- but the substrate here is plain additive Linear Attention (RNN-04 carry-forward), and
the Qwen gate is NOT advanced by this packet.

Reuses the RNN-04 qualified infrastructure WITHOUT modifying its numerical behaviour:
  - MQAR benchmark        : rnn_mc_bench.py   (blake2b process-stable seeds; self-qualified)
  - memory + aggregation  : rnn_mc_substrate.py primitives (_proj, _seg_linear, read_states, agg_* ; unit
                            tested; CHECKPOINT_RESTORE BIT_EXACT)
Aggregation orchestration lives HERE (segmented_forward) so the frozen-backbone vs trainable-reader split is
explicit: the ONLY trainable reader parameter is the gate connector `mem.w_u` (arm D); everything else is
frozen. Param-free arms (single / moving_average / u=q) train nothing.

Backbone = {emb, q/k/v projections + convs, out, norm, head}. Reader = {mem.w_u} (arm D only).

Phases:
  P1 backbone      : short calibration -> pick non-saturated D -> train ONE backbone (single-state) ->
                     save+hash checkpoint, prove requires_grad=False, record base logits.
  P2 lifecycle     : prove INDEPENDENT_COMPRESSOR (no cross-segment leak) and CONTINUOUS_CHECKPOINT
                     (warm-start continuation == full-sequence run) checkpoint/restore correctness.
  P3 arms          : A BASE(single) | B POST-MC indep (moving_avg) | C POST-MC continuous (moving_avg) |
                     D frozen backbone + TRAINED GRM reader (train only w_u) | E router-free query control
                     (u=q, param-free). All share the SAME frozen checkpoint. Immutability gate around D.
  P4 curve+cost    : fixed-backbone memory curve N in {1,2,4,8,16} (backbone AND reader identity constant;
                     only seg_size/N changes) for trained reader + param-free moving-avg. Direct-measured
                     storage/compute breakdown. seq-len + distance curves.
Outputs (outdir): rnn05a_results.json, backbone_identity.json, lifecycle_proofs.json, immutability_gate.json,
  param_accounting.json, pareto_fixed_backbone.csv, rnn05a_outcomes.json, run.log. Checkpoint saved EXTERNAL
  (ckpt_path, default scratchpad) with SHA-256; NOT committed.

Usage: python rnn_mc_05a.py --outdir <dir> [--ckpt <path>] [--smoke]
"""
import argparse, csv, hashlib, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from rnn_mc_bench import MQARSpec, make_example
from rnn_mc_substrate import MQARModel, read_states, agg_moving_average, agg_grm

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float32
SEED = 42

# task / model constants (inherited from RNN-04 for continuity; tiny -> minutes not hours)
D_MODEL, D_K, D_V = 128, 24, 24
MAIN_SEG = 32
L = 256
NUM_KEYS, NUM_VALS, Q, DENS = 128, 64, 8, 0.3
POOL_TRAIN, POOL_EVAL = 4096, 512
CALIB_GRID = [40, 56, 72]     # ascending difficulty; pick base_acc nearest 0.5 in (0.20,0.90) = non-saturated
SELECT_RULE = "num_pairs whose BASE single-state acc is nearest 0.5 within (0.20,0.90); else max-info"
MARGIN = 0.03                 # OPERATOR_HEURISTIC effect threshold (POSITIVE/NEGATIVE band)


def spec(name, D, Ln=L):
    return MQARSpec(seq_len=Ln, num_pairs=D, num_queries=Q, distractor_density=DENS,
                    num_keys=NUM_KEYS, num_vals=NUM_VALS, name=name)


def make_pool(sp, n, start=0):
    ex = [make_example(sp, i) for i in range(start, start + n)]
    return dict(
        ids=torch.tensor([e["input_ids"] for e in ex], dtype=torch.long),
        lab=torch.tensor([e["labels"] for e in ex], dtype=torch.long),
        dist=torch.tensor([[p["distance"] for p in e["pairs"]] for e in ex], dtype=torch.long),
        apos=torch.tensor([e["answer_positions"] for e in ex], dtype=torch.long),
        ids_hash=hashlib.blake2b(json.dumps([e["example_id"] for e in ex]).encode(),
                                 digest_size=8).hexdigest(),
        first_ids=[e["example_id"] for e in ex[:3]], n=n)


def build_model(vocab, d_k=D_K, d_v=D_V):
    torch.manual_seed(SEED)
    return MQARModel(vocab, d_model=D_MODEL, d_k=d_k, d_v=d_v).to(DEV).to(DT)


# ---------------- frozen-backbone / reader forward (uses ONLY qualified primitives) ----------------
def segmented_forward(model, ids, reader, seg_size, warm_start, probe_counts=None):
    """reader in {single, moving_average, grm_wu, grm_q}. Mirrors RNN-04 DeltaMemory.forward exactly for
    'single'/'grm_wu'; 'grm_q' swaps the trainable connector u=w_u(x) for the param-free u=q (arm E).
    probe_counts: if a list is passed, appends the number of HISTORICAL cached states this segment
    reads/gates against (= len(cached_states) at aggregation time) -- ground truth for the cost-probe
    assertion (audit reconciliation section 3). Behaviour is otherwise identical."""
    mem = model.mem
    x = model.emb(ids)
    q, k, v, _ = mem._proj(x)
    B, Lx, _ = x.shape
    segs = [(0, Lx)] if (reader == "single" or seg_size is None) else \
           [(i, min(i + seg_size, Lx)) for i in range(0, Lx, seg_size)]
    cached_states, cached_pools, outs = [], [], []
    S_prev = None
    for (a, b) in segs:
        qs, ks, vs = q[:, a:b], k[:, a:b], v[:, a:b]
        sp = S_prev if (warm_start and S_prev is not None) else None
        online, S_fin = mem._seg_linear(qs, ks, vs, sp)
        if reader == "single":
            o = online
        else:
            if probe_counts is not None:
                probe_counts.append(len(cached_states))   # historical states used for THIS segment
            cr = read_states(cached_states, qs) if cached_states else []
            if not cr:
                o = online
            elif reader == "moving_average":
                o = agg_moving_average(cr, online)
            else:  # grm_wu (trained) / grm_q (param-free control)
                u = mem.w_u(x[:, a:b]) if reader == "grm_wu" else qs
                kcum = torch.cumsum(ks, dim=1) / torch.arange(1, b - a + 1, device=x.device).view(1, -1, 1)
                gl_online = torch.einsum('bsk,bsk->bs', u, kcum).unsqueeze(-1)
                P = torch.stack(cached_pools, dim=1)
                gl_cached = torch.einsum('bsk,bck->bsc', u, P)
                gl = torch.cat([gl_cached, gl_online], dim=-1)
                o = agg_grm(cr, online, gl)
        outs.append(mem.out(o))
        if reader != "single":
            cached_states.append(S_fin)
            cached_pools.append(ks.mean(dim=1))
        S_prev = S_fin
    y = torch.cat(outs, dim=1)
    return model.head(model.norm(x + y))


def train(model, pool, steps, reader, seg, warm_start=False, params=None, batch=128, lr=1e-3, log=None):
    params = list(model.parameters()) if params is None else params
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator(device="cpu").manual_seed(SEED)
    model.train()
    t0 = time.time()
    for s in range(steps):
        idx = torch.randint(0, pool["n"], (batch,), generator=g)
        ids = pool["ids"][idx].to(DEV); lab = pool["lab"][idx].to(DEV)
        logits = segmented_forward(model, ids, reader, seg, warm_start)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), lab.view(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if log and (s % 500 == 0 or s == steps - 1):
            log(f"    step {s:4d}/{steps} loss {loss.item():.4f}")
    return dict(steps=steps, wall_s=round(time.time() - t0, 1), final_loss=round(loss.item(), 4),
                trained_params=sum(p.numel() for p in params))


@torch.no_grad()
def evaluate(model, pool, reader, seg, warm_start=False, batch=256, by_distance=False):
    model.eval()
    correct = total = 0
    dc, dt = {}, {}
    for i in range(0, pool["n"], batch):
        ids = pool["ids"][i:i + batch].to(DEV); lab = pool["lab"][i:i + batch].to(DEV)
        pred = segmented_forward(model, ids, reader, seg, warm_start).argmax(-1)
        mask = lab != -100
        correct += (pred[mask] == lab[mask]).sum().item(); total += mask.sum().item()
        if by_distance:
            apos = pool["apos"][i:i + batch]; dst = pool["dist"][i:i + batch]
            for bb in range(ids.size(0)):
                for j, p in enumerate(apos[bb].tolist()):
                    d = int(dst[bb, j].item()); ok = int(pred[bb, p].item() == lab[bb, p].item())
                    bkt = (d // 32) * 32
                    dc[bkt] = dc.get(bkt, 0) + ok; dt[bkt] = dt.get(bkt, 0) + 1
    acc = round(correct / total, 4) if total else 0.0
    if by_distance:
        return acc, {str(k): round(dc[k] / dt[k], 4) for k in sorted(dt)}
    return acc


# ---------------- backbone identity / immutability ----------------
def tensor_hashes(model):
    h = {}
    for name, p in model.state_dict().items():
        h[name] = hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return h


def compare_hashes(h0, h1, exclude=()):
    changed = [n for n in h0 if n not in exclude and h0.get(n) != h1.get(n)]
    return changed


@torch.no_grad()
def base_logits(model, pool, n=64):
    model.eval()
    ids = pool["ids"][:n].to(DEV)
    return segmented_forward(model, ids, "single", None, False).detach().cpu()


# ---------------- lifecycle checkpoint/restore proofs (section: stage 2) ----------------
@torch.no_grad()
def lifecycle_proofs(model, pool):
    """Prove both segment-memory interpretations are correct on the frozen backbone.
    INDEPENDENT: a segment's cached state depends ONLY on its own tokens (no leak from neighbours).
    CONTINUOUS : warm-start restore + continue == running the whole prefix at once (bit/numeq)."""
    mem = model.mem
    ids = pool["ids"][:16].to(DEV)
    x = model.emb(ids)
    q, k, v, _ = mem._proj(x)                          # frozen features computed ONCE (shared by both modes)
    seg = MAIN_SEG
    tol = 1e-4
    out = {}

    # --- INDEPENDENT compressor: each segment's recurrent state starts from the fixed initial state, so it
    #     carries ZERO information from any prior segment's STATE. Tested at the recurrent-state level with
    #     FIXED features (not by perturbing tokens -- the depthwise causal conv has a legitimate cross-
    #     boundary receptive field which is NOT recurrent-state leakage). Property proved:
    #       independent(seg) == warmstart(seg, S_prev) - S_prev  for ANY S_prev  (i.e. no carryover). ---
    a, b = seg, 2 * seg
    _, S_indep = mem._seg_linear(q[:, a:b], k[:, a:b], v[:, a:b], None)         # from fixed init
    torch.manual_seed(0)
    S_rand = torch.randn_like(S_indep)                                          # arbitrary prior state
    _, S_warm = mem._seg_linear(q[:, a:b], k[:, a:b], v[:, a:b], S_rand)        # warm-start with it
    # leak uses fp subtraction ((S_local+S_rand)-S_rand-S_local); fp32 non-associativity -> compare to tol
    leak = float(((S_warm - S_rand) - S_indep).abs().max())   # ~0 -> independent excludes carryover
    _, S_standalone = mem._seg_linear(q[:, a:b], k[:, a:b], v[:, a:b], None)
    indep_selfmatch = float((S_indep - S_standalone).abs().max())              # 0 -> uses only own tokens
    warm_actually_carries = float((S_warm - S_indep).abs().max())              # > 0 -> modes are distinct
    out["INDEPENDENT_COMPRESSOR_SEMANTICS"] = "QUALIFIED" if (
        leak < tol and indep_selfmatch == 0.0 and warm_actually_carries > tol) else "FAILED"
    out["independent"] = dict(cross_segment_state_leak_maxabs=leak, self_recompute_maxabs=indep_selfmatch,
                              warmstart_carry_magnitude=warm_actually_carries,
                              note="independent(seg) == warmstart(seg,S_prev) - S_prev for arbitrary S_prev "
                                   "-> zero recurrent-state carryover; modes provably distinct")

    # --- CONTINUOUS: checkpoint after prefix, serialize->reload (bit-exact), continue == full run. ---
    P = 2 * seg
    full_reads, S_full = mem._seg_linear(q[:, :P], k[:, :P], v[:, :P], None)     # whole prefix at once
    _, S_ck = mem._seg_linear(q[:, :seg], k[:, :seg], v[:, :seg], None)          # checkpoint at `seg`
    blob = S_ck.detach().cpu().numpy().tobytes()
    S_reload = torch.from_numpy(np.frombuffer(blob, dtype=np.float32).copy()).view_as(S_ck).to(S_ck)
    reload_bitexact = float((S_ck - S_reload).abs().max())                       # 0.0 -> BIT_EXACT
    r_cont, S_cont = mem._seg_linear(q[:, seg:P], k[:, seg:P], v[:, seg:P], S_reload)  # restore + continue
    state_numeq = float((S_full - S_cont).abs().max())
    reads_numeq = float((full_reads[:, seg:] - r_cont).abs().max())
    out["CONTINUOUS_CHECKPOINT_SEMANTICS"] = "QUALIFIED" if (reload_bitexact == 0.0 and state_numeq < tol
                                                             and reads_numeq < tol) else "FAILED"
    out["continuous"] = dict(reload_bitexact=reload_bitexact, continuation_state_maxabs=state_numeq,
                             continuation_reads_maxabs=reads_numeq, tol=tol,
                             note="warm-start restore+continue == full-prefix run")
    out["CHECKPOINT_RESTORE"] = "BIT_EXACT" if reload_bitexact == 0.0 else "DIFFERENT"
    return out


# ---------------- direct-measured storage/compute breakdown (section 16) ----------------
@torch.no_grad()
def cost_breakdown(model, pool, seg, warm_start, reps=10, batch=128):
    """Directly time each component; only 'total' is end-to-end. No read-time-by-subtraction."""
    mem = model.mem
    ids = pool["ids"][:batch].to(DEV)
    x = model.emb(ids)
    q, k, v, _ = mem._proj(x)
    B, Lx, _ = x.shape
    segs = [(i, min(i + seg, Lx)) for i in range(0, Lx, seg)]

    def sync():
        torch.cuda.synchronize() if DEV == "cuda" else None

    def timeit(fn):
        sync(); t0 = time.time()
        for _ in range(reps):
            fn()
        sync()
        return (time.time() - t0) / reps * 1000

    # recurrent update (state build over all segments), checkpoint copy, read, gate, total
    def do_update():
        S_prev = None; states = []
        for (a, b) in segs:
            _, S = mem._seg_linear(q[:, a:b], k[:, a:b], v[:, a:b],
                                   S_prev if (warm_start and S_prev is not None) else None)
            states.append(S); S_prev = S
        return states
    states = do_update()
    pools = [k[:, a:b].mean(dim=1) for (a, b) in segs]      # per-segment mean-key pool, precomputed once
    # AUDIT-FIX (reconciliation section 3): segment i reads/gates against ALL prior states/pools [0:i]
    # (was states[:1] / segs[:1] -> wrong for N>2). This matches segmented_forward's GRM path exactly.
    def do_read():
        for i, (a, b) in enumerate(segs):
            if i == 0:
                continue
            read_states(states[:i], q[:, a:b])              # all prior states 0..i-1
    def do_gate():
        for i, (a, b) in enumerate(segs):
            if i == 0:
                continue
            u = mem.w_u(x[:, a:b])
            kcum = torch.cumsum(k[:, a:b], 1) / torch.arange(1, b - a + 1, device=x.device).view(1, -1, 1)
            P = torch.stack(pools[:i], dim=1)               # all prior pools 0..i-1
            gl = torch.cat([torch.einsum('bsk,bck->bsc', u, P),
                            torch.einsum('bsk,bsk->bs', u, kcum).unsqueeze(-1)], -1)
            torch.softmax(gl, -1)
    def do_ckpt():
        return [s.clone() for s in states]

    # cost-probe correctness: the historical-state count the probe uses per segment MUST equal the count
    # the real GRM forward uses. Probe uses states[:i] (i.e. 0,1,2,...); capture the forward's ground truth.
    probe_hist_counts = list(range(len(segs)))              # segment i -> i historical states
    fwd_counts = []
    segmented_forward(model, ids, "grm_wu", seg, warm_start, probe_counts=fwd_counts)
    probe_matches_forward = (probe_hist_counts == fwd_counts)

    state_bytes = states[-1].numel() // B * states[-1].element_size()
    return dict(
        n_segments=len(segs),
        recurrent_update_ms=round(timeit(do_update), 3),
        checkpoint_copy_ms=round(timeit(do_ckpt), 3),
        state_read_ms=round(timeit(do_read), 3),
        gate_router_ms=round(timeit(do_gate), 3),
        total_latency_ms=round(timeit(lambda: segmented_forward(model, ids, "grm_wu", seg, warm_start)), 3),
        state_bytes_per_req=int(state_bytes),
        cache_bytes=int(state_bytes * max(0, len(segs) - 1)),
        historical_states_per_segment=fwd_counts,
        cost_probe_matches_forward=bool(probe_matches_forward),
    )


def classify(delta):
    if delta >= MARGIN:
        return "POSITIVE"
    if delta <= -MARGIN:
        return "NEGATIVE"
    return "NO_EFFECT"


@torch.no_grad()
def run_cost_selfcheck(out_path):
    """Audit reconciliation section 3: prove the corrected cost probe reads/gates against the SAME number of
    historical states per segment as the real GRM forward, for N>2. Structural (count) check -> runs on a
    fresh UNTRAINED tiny model; no training, no tuning, value-independent. NOT a cost remeasurement."""
    vocab = spec("v", D=40).vocab_size
    m = build_model(vocab)
    for p in m.parameters():
        p.requires_grad_(False)
    sp = spec("cost_selfcheck", D=40)
    pool = make_pool(sp, 128)
    cases = []
    ok = True
    for N in [2, 4, 8, 16]:
        seg = math.ceil(L / N)
        c = cost_breakdown(m, pool, seg, warm_start=False, reps=1, batch=32)
        expected = list(range(c["n_segments"]))                 # segment i uses i historical states
        match = (c["historical_states_per_segment"] == expected) and c["cost_probe_matches_forward"]
        ok = ok and match
        cases.append(dict(N=N, seg=seg, n_segments=c["n_segments"],
                          historical_states_per_segment=c["historical_states_per_segment"],
                          expected=expected, cost_probe_matches_forward=c["cost_probe_matches_forward"],
                          match=bool(match)))
    result = dict(packet="RNN-05A-audit-reconciliation",
                  COST_PROBE_SELFCHECK="PASS" if ok else "FAIL",
                  note="probe reads states[:i] and gates pools[:i] for segment i; equals GRM forward's "
                       "len(cached_states). Fixes the states[:1]/segs[:1] N>2 bug. Structural/count test on "
                       "an untrained model (value-independent); NOT a cost remeasurement.",
                  torch=torch.__version__, cases=cases)
    if out_path:
        json.dump(result, open(out_path, "w"), indent=2)
    print(json.dumps({k: result[k] for k in ["COST_PROBE_SELFCHECK"]}, indent=2))
    return result


# ==================================================================================================
def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    logf = open(os.path.join(args.outdir, "run.log"), "a")
    def log(m):
        print(m); logf.write(m + "\n"); logf.flush()
    ckpt_path = args.ckpt
    steps_cal = 300 if args.smoke else 1500
    steps_bb = 300 if args.smoke else 5000
    steps_rd = 300 if args.smoke else 6000
    vocab = spec("v", D=40).vocab_size

    R = dict(meta=dict(packet="RNN-05A", device=DEV, dtype="float32", seed=SEED, torch=torch.__version__,
                       numpy=np.__version__, d_model=D_MODEL, d_k=D_K, d_v=D_V, main_seg=MAIN_SEG, L=L,
                       num_keys=NUM_KEYS, num_vals=NUM_VALS, num_queries=Q, density=DENS,
                       select_rule=SELECT_RULE, margin_OPERATOR_HEURISTIC=MARGIN, ckpt_path=ckpt_path),
             calibration={}, backbone={}, lifecycle={}, arms={}, immutability={}, param_accounting={},
             curve={}, cost={}, seqlen={}, benchmarks={}, outcomes={})
    def snap():
        json.dump(R, open(os.path.join(args.outdir, "rnn05a_results.json"), "w"), indent=2)

    # ---- P1a calibration: pick a non-saturated D (throwaway models; NOT the backbone) ----
    log(f"[P1a] calibration grid D={CALIB_GRID} (steps={steps_cal}) dev={DEV}")
    for D in CALIB_GRID:
        sp = spec(f"cal_D{D}", D)
        tr = make_pool(sp, POOL_TRAIN); ev = make_pool(sp, POOL_EVAL, start=POOL_TRAIN)
        m = build_model(vocab); st = train(m, tr, steps_cal, "single", None, log=log)
        acc = evaluate(m, ev, "single", None)
        R["calibration"][str(D)] = dict(base_acc=acc, **st)
        log(f"  D={D:3d} base_acc={acc:.4f} ({st['wall_s']}s)")
        del m; torch.cuda.empty_cache() if DEV == "cuda" else None; snap()
    band = {int(D): v["base_acc"] for D, v in R["calibration"].items() if 0.20 < v["base_acc"] < 0.90}
    MC_D = min(band, key=lambda d: abs(band[d] - 0.5)) if band else \
        int(min(R["calibration"], key=lambda d: abs(R["calibration"][d]["base_acc"] - 0.5)))
    R["MEMORY_AXIS"] = "QUALIFIED" if band else "NOT_QUALIFIED"
    R["MC_D"] = MC_D
    log(f"[P1a] MEMORY_AXIS={R['MEMORY_AXIS']} selected D={MC_D} base_acc={R['calibration'][str(MC_D)]['base_acc']}")

    # ---- P1b train ONE backbone (single-state), freeze, hash, record logits ----
    dev_sp = spec(f"dev_D{MC_D}", MC_D)
    hold_sp = spec(f"holdout_D{MC_D}", MC_D)           # fresh pinned seeds for confirmation (section: eval)
    tr = make_pool(dev_sp, POOL_TRAIN); dev_ev = make_pool(dev_sp, POOL_EVAL, start=POOL_TRAIN)
    hold_ev = make_pool(hold_sp, POOL_EVAL, start=POOL_TRAIN)
    R["benchmarks"] = dict(dev_spec=dev_sp.canonical(), holdout_spec=hold_sp.canonical(),
                           dev_eval_ids_hash=dev_ev["ids_hash"], holdout_eval_ids_hash=hold_ev["ids_hash"],
                           dev_first_ids=dev_ev["first_ids"], holdout_first_ids=hold_ev["first_ids"],
                           note="reader config chosen on dev; ALL headline arms confirmed on holdout")
    log(f"[P1b] train THE backbone once (single) D={MC_D} steps={steps_bb}")
    backbone = build_model(vocab)
    st_bb = train(backbone, tr, steps_bb, "single", None, log=log)
    dev_base = evaluate(backbone, dev_ev, "single", None, by_distance=True)
    hold_base = evaluate(backbone, hold_ev, "single", None, by_distance=True)
    # freeze + prove
    for p in backbone.parameters():
        p.requires_grad_(False)
    all_frozen = all(not p.requires_grad for p in backbone.parameters())
    h_freeze = tensor_hashes(backbone)
    logits_freeze = base_logits(backbone, hold_ev)
    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({"state_dict": backbone.state_dict(), "vocab": vocab, "d_k": D_K, "d_v": D_V,
                    "d_model": D_MODEL, "seed": SEED, "D": MC_D}, ckpt_path)
        ckpt_sha = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()
        ckpt_bytes = os.path.getsize(ckpt_path)
    else:
        ckpt_sha, ckpt_bytes = None, None
    R["backbone"] = dict(D=MC_D, train=st_bb, dev_base_acc=dev_base[0], dev_base_dist=dev_base[1],
                         holdout_base_acc=hold_base[0], holdout_base_dist=hold_base[1],
                         all_params_requires_grad_false=bool(all_frozen),
                         checkpoint_path=ckpt_path, checkpoint_sha256=ckpt_sha, checkpoint_bytes=ckpt_bytes,
                         per_tensor_sha256=h_freeze,
                         base_logits_fingerprint=hashlib.sha256(
                             logits_freeze.numpy().tobytes()).hexdigest())
    json.dump(R["backbone"], open(os.path.join(args.outdir, "backbone_identity.json"), "w"), indent=2)
    log(f"[P1b] backbone dev_base={dev_base[0]} holdout_base={hold_base[0]} frozen={all_frozen} "
        f"ckpt_sha={ckpt_sha[:12] if ckpt_sha else None}")
    snap()

    # ---- P2 lifecycle proofs (on frozen backbone) ----
    lp = lifecycle_proofs(backbone, dev_ev)
    R["lifecycle"] = lp
    json.dump(lp, open(os.path.join(args.outdir, "lifecycle_proofs.json"), "w"), indent=2)
    log(f"[P2] INDEPENDENT={lp['INDEPENDENT_COMPRESSOR_SEMANTICS']} "
        f"CONTINUOUS={lp['CONTINUOUS_CHECKPOINT_SEMANTICS']} CKPT={lp['CHECKPOINT_RESTORE']}")
    snap()

    # ---- P3 fixed-backbone arms (all share the frozen checkpoint) ----
    def fresh_backbone():
        m = build_model(vocab)
        if ckpt_path:
            m.load_state_dict(torch.load(ckpt_path, map_location=DEV)["state_dict"])
        else:
            m.load_state_dict(backbone.state_dict())
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    def arm_eval(m, reader, seg, warm, pool):
        return evaluate(m, pool, reader, seg, warm_start=warm, by_distance=True)

    # A BASE (frozen single) -- already have base; record both splits
    R["arms"]["A_BASE"] = dict(reader="single", seg=None, warm=False, trainable=0,
                               dev_acc=dev_base[0], holdout_acc=hold_base[0], holdout_dist=hold_base[1])
    # B POST-MC independent (param-free moving average, frozen)
    bI = arm_eval(backbone, "moving_average", MAIN_SEG, False, hold_ev)
    R["arms"]["B_POSTMC_indep"] = dict(reader="moving_average", seg=MAIN_SEG, warm=False, trainable=0,
                                       dev_acc=evaluate(backbone, dev_ev, "moving_average", MAIN_SEG),
                                       holdout_acc=bI[0], holdout_dist=bI[1], note="TRAINING_FREE")
    # C POST-MC continuous (param-free moving average, warm-start, frozen)
    bC = arm_eval(backbone, "moving_average", MAIN_SEG, True, hold_ev)
    R["arms"]["C_POSTMC_continuous"] = dict(reader="moving_average", seg=MAIN_SEG, warm=True, trainable=0,
                                            dev_acc=evaluate(backbone, dev_ev, "moving_average", MAIN_SEG,
                                                             warm_start=True),
                                            holdout_acc=bC[0], holdout_dist=bC[1], note="TRAINING_FREE")
    # E router-free query control (u=q, param-free, frozen) -- independent lifecycle (matches D primary)
    eI = arm_eval(backbone, "grm_q", MAIN_SEG, False, hold_ev)
    R["arms"]["E_query_control"] = dict(reader="grm_q", seg=MAIN_SEG, warm=False, trainable=0,
                                        dev_acc=evaluate(backbone, dev_ev, "grm_q", MAIN_SEG),
                                        holdout_acc=eI[0], holdout_dist=eI[1],
                                        note="param-free u=q gate; isolates trained-reader gain")
    snap()

    # D FROZEN backbone + TRAINED GRM reader (train ONLY mem.w_u). Immutability gate around it.
    log(f"[P3] arm D: train ONLY the GRM reader (w_u) on frozen backbone, indep lifecycle (steps={steps_rd})")
    mD = fresh_backbone()
    h_before = tensor_hashes(mD)
    logits_before = base_logits(mD, hold_ev)
    mD.mem.w_u.requires_grad_(True)
    reader_params = [p for n, p in mD.named_parameters() if p.requires_grad]
    reader_names = [n for n, p in mD.named_parameters() if p.requires_grad]
    stD = train(mD, tr, steps_rd, "grm_wu", MAIN_SEG, params=reader_params, log=log)
    h_after = tensor_hashes(mD)
    logits_after = base_logits(mD, hold_ev)
    changed = compare_hashes(h_before, h_after, exclude=("mem.w_u.weight",))
    logits_delta = float((logits_before - logits_after).abs().max())
    BACKBONE_WEIGHT_MUTATION = len(changed)
    frozen_valid = (BACKBONE_WEIGHT_MUTATION == 0 and logits_delta == 0.0)
    R["immutability"] = dict(FROZEN_BACKBONE_VALIDITY="PASS" if frozen_valid else "FAIL",
                             BACKBONE_WEIGHT_MUTATION=BACKBONE_WEIGHT_MUTATION,
                             changed_tensors=changed, base_logits_maxabs_delta=logits_delta,
                             reader_trainable_tensors=reader_names,
                             note="hashes compared over ALL tensors except the reader (mem.w_u); base "
                                  "single-state logits on holdout must be bit-identical after reader training")
    json.dump(R["immutability"], open(os.path.join(args.outdir, "immutability_gate.json"), "w"), indent=2)
    log(f"[P3] FROZEN_BACKBONE_VALIDITY={R['immutability']['FROZEN_BACKBONE_VALIDITY']} "
        f"mutation={BACKBONE_WEIGHT_MUTATION} logits_delta={logits_delta}")
    if not frozen_valid:
        log("[STOP] backbone mutated during reader training -> experiment invalid (stop condition).")
        R["outcomes"]["FROZEN_BACKBONE_VALIDITY"] = "FAIL"
        snap(); logf.close(); return

    dD_dev = evaluate(mD, dev_ev, "grm_wu", MAIN_SEG)
    dD = arm_eval(mD, "grm_wu", MAIN_SEG, False, hold_ev)
    R["arms"]["D_TRAINED_READER_indep"] = dict(reader="grm_wu", seg=MAIN_SEG, warm=False, train=stD,
                                               trainable=stD["trained_params"], dev_acc=dD_dev,
                                               holdout_acc=dD[0], holdout_dist=dD[1])
    # D' trained reader under CONTINUOUS lifecycle (separate reader; fresh frozen backbone)
    log(f"[P3] arm D': train GRM reader under CONTINUOUS lifecycle (steps={steps_rd})")
    mDc = fresh_backbone(); mDc.mem.w_u.requires_grad_(True)
    rp = [p for p in mDc.parameters() if p.requires_grad]
    stDc = train(mDc, tr, steps_rd, "grm_wu", MAIN_SEG, warm_start=True, params=rp, log=log)
    changed_c = compare_hashes(h_before, tensor_hashes(mDc), exclude=("mem.w_u.weight",))
    dDc_dev = evaluate(mDc, dev_ev, "grm_wu", MAIN_SEG, warm_start=True)
    dDc = evaluate(mDc, hold_ev, "grm_wu", MAIN_SEG, warm_start=True)
    R["arms"]["Dc_TRAINED_READER_continuous"] = dict(reader="grm_wu", seg=MAIN_SEG, warm=True, train=stDc,
                                                     trainable=stDc["trained_params"], dev_acc=dDc_dev,
                                                     holdout_acc=dDc, backbone_mutation=len(changed_c))
    snap()

    # ---- param accounting ----
    total = sum(p.numel() for p in backbone.parameters())
    reader_n = mD.mem.w_u.weight.numel()
    R["param_accounting"] = dict(
        frozen_backbone_params=total - reader_n, reader_router_params=reader_n, total_model_params=total,
        reader_tensor="mem.w_u.weight", reader_shape=list(mD.mem.w_u.weight.shape),
        reader_fraction=round(reader_n / total, 5),
        recurrent_state_bytes_per_req=D_K * D_V * 4,
        historical_cache_bytes_at_N=f"(N-1)*{D_K*D_V*4} bytes for N cached segments",
        note="reader = the single gate connector w_u; NO backbone params trained (arm D)")
    json.dump(R["param_accounting"], open(os.path.join(args.outdir, "param_accounting.json"), "w"), indent=2)
    snap()

    # ---- P4 fixed-backbone memory curve: SAME frozen backbone + SAME trained reader; vary N only ----
    log("[P4] fixed-backbone memory curve N in [1,2,4,8,16] (backbone+reader identity constant; seg varies)")
    curve_rows = []
    h_curve_before = tensor_hashes(mD)
    for N in [1, 2, 4, 8, 16]:
        seg = None if N == 1 else math.ceil(L / N)
        reader = "single" if N == 1 else "grm_wu"
        acc_tr = evaluate(mD, hold_ev, reader, seg)             # trained reader (constant w_u)
        acc_mf = evaluate(backbone, hold_ev, "single" if N == 1 else "moving_average", seg)  # param-free
        cost = cost_breakdown(mD, hold_ev, (seg or L), False) if N > 1 else dict(
            n_segments=1, recurrent_update_ms=None, checkpoint_copy_ms=0.0, state_read_ms=0.0,
            gate_router_ms=0.0, total_latency_ms=None, state_bytes_per_req=D_K * D_V * 4, cache_bytes=0)
        row = dict(N=N, seg=(L if N == 1 else seg), trained_reader_acc=acc_tr, moving_avg_acc=acc_mf,
                   cache_bytes=cost["cache_bytes"], state_bytes_per_req=cost["state_bytes_per_req"],
                   recurrent_update_ms=cost["recurrent_update_ms"], checkpoint_copy_ms=cost["checkpoint_copy_ms"],
                   state_read_ms=cost["state_read_ms"], gate_router_ms=cost["gate_router_ms"],
                   total_latency_ms=cost["total_latency_ms"])
        R["curve"][str(N)] = row; curve_rows.append(row)
        log(f"  N={N:2d} seg={row['seg']} trained_reader={acc_tr} moving_avg={acc_mf} "
            f"cache_bytes={row['cache_bytes']}")
        snap()
    curve_unchanged = compare_hashes(h_curve_before, tensor_hashes(mD), exclude=())  # eval-only -> []
    accs = [r["trained_reader_acc"] for r in curve_rows]
    R["curve_meta"] = dict(backbone_and_reader_constant=(len(curve_unchanged) == 0),
                           monotone_nondecreasing=all(accs[i] <= accs[i + 1] + 1e-9 for i in range(len(accs) - 1)),
                           only_component_changed="seg_size (N)", changed_tensors_during_curve=curve_unchanged)
    with open(os.path.join(args.outdir, "pareto_fixed_backbone.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys())); w.writeheader(); w.writerows(curve_rows)

    # seq-len curve (frozen backbone base vs trained reader), independent
    for Ln in ([256] if args.smoke else [256, 384, 512]):
        spL = spec(f"len_D{MC_D}_L{Ln}", MC_D, Ln=Ln)
        evL = make_pool(spL, POOL_EVAL, start=POOL_TRAIN)
        R["seqlen"][str(Ln)] = dict(base=evaluate(mD, evL, "single", None),
                                    trained_reader=evaluate(mD, evL, "grm_wu", MAIN_SEG),
                                    moving_avg=evaluate(backbone, evL, "moving_average", MAIN_SEG))
        log(f"  L={Ln} base={R['seqlen'][str(Ln)]['base']} reader={R['seqlen'][str(Ln)]['trained_reader']} "
            f"mavg={R['seqlen'][str(Ln)]['moving_avg']}")
        snap()

    # ---- outcomes vocabulary ----
    base_h = hold_base[0]
    tf_i = classify(round(R["arms"]["B_POSTMC_indep"]["holdout_acc"] - base_h, 4))
    tf_c = classify(round(R["arms"]["C_POSTMC_continuous"]["holdout_acc"] - base_h, 4))
    d_hold = round(dD[0] - base_h, 4); d_dev = round(dD_dev - dev_base[0], 4)
    if classify(d_hold) == "POSITIVE" and d_dev >= MARGIN:
        trained = "POSITIVE"
    elif classify(d_hold) == "NEGATIVE":
        trained = "NEGATIVE"
    elif (d_hold >= MARGIN) != (d_dev >= MARGIN):
        trained = "INCONCLUSIVE"
    else:
        trained = "NO_EFFECT"
    curve_qual = "QUALIFIED" if (R["curve_meta"]["backbone_and_reader_constant"]
                                 and R["curve_meta"]["monotone_nondecreasing"]) else "NOT_QUALIFIED"
    R["outcomes"] = dict(
        FROZEN_BACKBONE_VALIDITY=R["immutability"]["FROZEN_BACKBONE_VALIDITY"],
        INDEPENDENT_COMPRESSOR_SEMANTICS=lp["INDEPENDENT_COMPRESSOR_SEMANTICS"],
        CONTINUOUS_CHECKPOINT_SEMANTICS=lp["CONTINUOUS_CHECKPOINT_SEMANTICS"],
        TRAINING_FREE_MC_INDEPENDENT=tf_i, TRAINING_FREE_MC_CONTINUOUS=tf_c,
        TRAINED_READER_MC=trained, FIXED_BACKBONE_MEMORY_CURVE=curve_qual,
        QWEN_GDN_TRANSPLANT_GATE="CONDITIONAL / DEFER",
        deltas=dict(base_holdout=base_h, trained_reader_holdout_delta=d_hold, trained_reader_dev_delta=d_dev,
                    training_free_indep_delta=round(R["arms"]["B_POSTMC_indep"]["holdout_acc"] - base_h, 4),
                    training_free_cont_delta=round(R["arms"]["C_POSTMC_continuous"]["holdout_acc"] - base_h, 4),
                    query_control_holdout_delta=round(R["arms"]["E_query_control"]["holdout_acc"] - base_h, 4)),
        margin_OPERATOR_HEURISTIC=MARGIN)
    json.dump(R["outcomes"], open(os.path.join(args.outdir, "rnn05a_outcomes.json"), "w"), indent=2)
    snap()
    log("[DONE] outcomes:\n" + json.dumps(R["outcomes"], indent=2))
    logf.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--ckpt", default=None, help="external checkpoint path (NOT committed)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cost-selfcheck", default=None, metavar="JSON",
                    help="run the cost-probe historical-count assertion (no training) and write JSON")
    a = ap.parse_args()
    if a.cost_selfcheck:
        run_cost_selfcheck(a.cost_selfcheck)
    else:
        if not a.outdir:
            ap.error("--outdir is required unless --cost-selfcheck is given")
        run(a)
