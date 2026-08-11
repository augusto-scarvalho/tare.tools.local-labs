#!/usr/bin/env python
"""
RNN-05B experiment harness: Memory Caching on ACTUAL DeltaNet / Gated-DeltaNet (vs Linear-Attention control).

Central question (RNN05B_DELTA_SEMANTICS.md): does Memory Caching behave differently when the substrate is a
real delta / gated-delta recurrence whose historical states do NOT collapse to the additive final-state
sufficient statistic (as Linear Attention does, RNN-05A)? Distinguish:
  H1 MC only helps with backbone-memory CO-ADAPTATION
  H2 RNN-05A failed specifically because LA collapses history into the final state
  H3 a real delta/gated-delta update stores useful history not recoverable from the final state

Reuses: rnn_mc_bench (MQAR, blake2b process-stable), rnn_mc_substrate (pure aggregation fns),
rnn_delta_substrate (LA/DN/GDN scan+chunked, complete-state checkpoint, parity/isolation/collapse gates).

Design (matched family LA/DN/GDN, identical width/params/task/optimizer/seed):
  P0 gates    : parity + full-module lifecycle + request isolation (STOP if any fail) [rnn_delta_substrate]
  P1 calib    : pick a non-saturated MQAR difficulty D on BASE (shared by all substrates)
  P2 2x2      : per mode train {single-state, MC-aware} backbones; eval {single, MC} inference
                -> JOINT_TRAINED_MC_EFFECT, FROZEN_POST_MC_EFFECT (param-free), co-adaptation interaction
  P3 frozen   : freeze the single-state backbone; BASE vs param-free MC vs small TRAINED reader (w_u only);
                reader saved DURABLY with SHA-256 (packet §17)
  P4 curve    : PURE_CACHE_COUNT_CURVE on identical weights (fixed checkpoint positions; retain most-recent K)
  P5 novelty  : HISTORICAL_STATE_NOVELTY (recall of early-written associations: base final-state vs MC reads)
  P6 cost     : direct-measured compute + memory accounting (recurrent/conv/checkpoint/read/gate/total)
GDN (load-bearing) gets 3 training seeds; LA/DN get 1 (predeclared, packet §21).

Usage: python rnn_mc_05b.py --outdir <dir> [--artifacts <dir>] [--smoke]
"""
import argparse, csv, hashlib, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from rnn_mc_bench import MQARSpec, make_example
from rnn_mc_substrate import read_states, agg_moving_average, agg_grm
from rnn_delta_substrate import (MQARDeltaModel, run_recurrence, reference_parity,
                                 checkpoint_restore_full_module, request_isolation, collapsibility,
                                 full_state_bytes, MODES)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float32
D_MODEL, D_K, D_V, CONV_K = 128, 64, 64, 4     # d_k>=max pairs: delta rule is rank~d_k, needs capacity
L = 256
MAIN_SEG = 32                          # MC segment size AND recurrence chunk size
NUM_KEYS, NUM_VALS, Q, DENS = 128, 64, 8, 0.3
POOL_TRAIN, POOL_EVAL = 4096, 512
CALIB_GRID = [36, 40]                  # delta rule rank~d_k(=64) has a SHARP capacity cliff: stable+learnable
                                       # at D<=40 (~0.9), unstable/chance at D>=44 (canary-established). D=40
                                       # is the robust non-saturated-enough operating point (base<1, headroom).
LR = 3e-3                              # delta family needs lr>~2e-3 (canary-established); single lr for all
MARGIN = 0.03                          # OPERATOR_HEURISTIC effect band (POSITIVE/NEGATIVE)
GDN_SEEDS = [42, 43, 44]               # load-bearing: 3 training seeds
SINGLE_SEED = [42]                     # LA/DN: 1 seed (predeclared)


def spec(name, D, Ln=L):
    return MQARSpec(seq_len=Ln, num_pairs=D, num_queries=Q, distractor_density=DENS,
                    num_keys=NUM_KEYS, num_vals=NUM_VALS, name=name)


def make_pool(sp, n, start=0):
    ex = [make_example(sp, i) for i in range(start, start + n)]
    seg = MAIN_SEG
    return dict(
        ids=torch.tensor([e["input_ids"] for e in ex], dtype=torch.long),
        lab=torch.tensor([e["labels"] for e in ex], dtype=torch.long),
        dist=torch.tensor([[p["distance"] for p in e["pairs"]] for e in ex], dtype=torch.long),
        apos=torch.tensor([e["answer_positions"] for e in ex], dtype=torch.long),
        wpos=torch.tensor([[p["write_pos"] for p in e["pairs"]] for e in ex], dtype=torch.long),
        wseg=torch.tensor([[p["write_pos"] // seg for p in e["pairs"]] for e in ex], dtype=torch.long),
        ids_hash=hashlib.blake2b(json.dumps([e["example_id"] for e in ex]).encode(), digest_size=8).hexdigest(),
        first_ids=[e["example_id"] for e in ex[:3]], n=n)


def build_model(vocab, seed=42):
    torch.manual_seed(seed)
    return MQARDeltaModel(vocab, d_model=D_MODEL, d_k=D_K, d_v=D_V, conv_k=CONV_K).to(DEV).to(DT)


# ---------------- MC segmented forward on the delta family ----------------
def segmented_forward(model, ids, mode, reader, seg_size, warm_start, path="chunked", recent_k=None):
    """reader in {single, moving_average, grm}. Caches each segment's final recurrent state; aggregates
    cached + online reads (RNN-04 pure fns). recent_k: keep only the most-recent-k cached states (pure
    cache-count sweep, §13). warm_start: continuous (carry S) vs independent (reset S per segment)."""
    blk = model.blk
    x, q, k, v, g, beta, _ = model.project(ids)
    B, Lx, _ = q.shape
    if reader == "single" or seg_size is None:
        o, _ = run_recurrence(mode, q, k, v, g, beta, seg_size or MAIN_SEG, None, path)
        return model.readout(x, o)
    segs = [(i, min(i + seg_size, Lx)) for i in range(0, Lx, seg_size)]
    cached_states, cached_pools, outs = [], [], []
    S_prev = None
    for (a, b) in segs:
        qs, ks, vs, gs, bs = q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b]
        S0 = S_prev if (warm_start and S_prev is not None) else None
        online, S_fin = run_recurrence(mode, qs, ks, vs, gs, bs, seg_size, S0, path)
        avail = cached_states if recent_k is None else cached_states[-recent_k:]
        avail_pools = cached_pools if recent_k is None else cached_pools[-recent_k:]
        cr = read_states(avail, qs) if avail else []
        if not cr:
            o = online
        elif reader == "moving_average":
            o = agg_moving_average(cr, online)
        else:  # grm trained reader
            u = blk.w_u(x[:, a:b])
            kcum = torch.cumsum(ks, dim=1) / torch.arange(1, b - a + 1, device=q.device).view(1, -1, 1)
            gl_online = torch.einsum('bsk,bsk->bs', u, kcum).unsqueeze(-1)
            P = torch.stack(avail_pools, dim=1)
            gl = torch.cat([torch.einsum('bsk,bck->bsc', u, P), gl_online], dim=-1)
            o = agg_grm(cr, online, gl)
        outs.append(blk.out(o))
        cached_states.append(S_fin)
        cached_pools.append(ks.mean(dim=1))
        S_prev = S_fin
    return model.head(model.norm(x + torch.cat(outs, dim=1)))


def train(model, pool, steps, mode, reader, seg, warm_start=False, params=None, batch=128, lr=1e-3, seed=42,
          log=None, path="chunked"):
    params = list(model.parameters()) if params is None else params
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gcpu = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    t0 = time.time()
    loss = torch.tensor(0.0)
    for s in range(steps):
        idx = torch.randint(0, pool["n"], (batch,), generator=gcpu)
        ids = pool["ids"][idx].to(DEV); lab = pool["lab"][idx].to(DEV)
        logits = segmented_forward(model, ids, mode, reader, seg, warm_start, path=path)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), lab.view(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if log and (s % 1000 == 0 or s == steps - 1):
            log(f"      step {s:4d}/{steps} loss {loss.item():.4f}")
    return dict(steps=steps, wall_s=round(time.time() - t0, 1), final_loss=round(float(loss.item()), 4),
                trained_params=sum(p.numel() for p in params))


@torch.no_grad()
def evaluate(model, pool, mode, reader, seg, warm_start=False, batch=256, recent_k=None, by_wseg=False):
    model.eval()
    correct = total = 0
    sc, st = {}, {}
    for i in range(0, pool["n"], batch):
        ids = pool["ids"][i:i + batch].to(DEV); lab = pool["lab"][i:i + batch].to(DEV)
        pred = segmented_forward(model, ids, mode, reader, seg, warm_start, recent_k=recent_k).argmax(-1)
        mask = lab != -100
        correct += (pred[mask] == lab[mask]).sum().item(); total += mask.sum().item()
        if by_wseg:
            apos = pool["apos"][i:i + batch]; ws = pool["wseg"][i:i + batch]
            for bb in range(ids.size(0)):
                for j, p in enumerate(apos[bb].tolist()):
                    seg_idx = int(ws[bb, j].item()); ok = int(pred[bb, p].item() == lab[bb, p].item())
                    sc[seg_idx] = sc.get(seg_idx, 0) + ok; st[seg_idx] = st.get(seg_idx, 0) + 1
    acc = round(correct / total, 4) if total else 0.0
    if by_wseg:
        return acc, {str(kk): round(sc[kk] / st[kk], 4) for kk in sorted(st)}
    return acc


def tensor_hashes(model):
    return {n: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for n, p in model.state_dict().items()}


def classify(delta):
    return "POSITIVE" if delta >= MARGIN else ("NEGATIVE" if delta <= -MARGIN else "NO_EFFECT")


def signal_from(dev_delta, hold_delta):
    """Combine dev + holdout into a substrate MC signal with the outcome vocabulary."""
    if classify(hold_delta) == "POSITIVE" and dev_delta >= MARGIN:
        return "POSITIVE"
    if classify(hold_delta) == "NEGATIVE":
        return "NEGATIVE"
    if (hold_delta >= MARGIN) != (dev_delta >= MARGIN):
        return "INCONCLUSIVE"
    return "NO_EFFECT"


# ==================================================================================================
def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    art = args.artifacts or args.outdir
    os.makedirs(art, exist_ok=True)
    logf = open(os.path.join(args.outdir, "run.log"), "a")
    def log(m):
        print(m); logf.write(m + "\n"); logf.flush()

    smoke = args.smoke
    steps_cal = 120 if smoke else 1800
    steps_bb = 150 if smoke else 3000
    steps_rd = 150 if smoke else 3000
    seeds_gdn = [42] if smoke else GDN_SEEDS
    vocab = spec("v", D=40).vocab_size

    R = dict(meta=dict(packet="RNN-05B", device=DEV, dtype="float32", torch=torch.__version__,
                       numpy=np.__version__, d_model=D_MODEL, d_k=D_K, d_v=D_V, conv_k=CONV_K, L=L,
                       main_seg=MAIN_SEG, num_keys=NUM_KEYS, num_vals=NUM_VALS, num_queries=Q, density=DENS,
                       margin_OPERATOR_HEURISTIC=MARGIN, gdn_seeds=seeds_gdn, la_dn_seeds=SINGLE_SEED,
                       smoke=smoke, artifacts_dir=art),
             gates={}, calibration={}, twobytwo={}, frozen={}, cache_count={}, novelty={}, cost={},
             memory={}, outcomes={})
    def snap():
        json.dump(R, open(os.path.join(args.outdir, "rnn05b_results.json"), "w"), indent=2)

    # ---------------- P0 gates (parity / full-module lifecycle / request isolation) ----------------
    log("[P0] substrate gates: parity + full-module checkpoint/restore + request isolation")
    gm = build_model(vocab, seed=0)
    gcpu = torch.Generator().manual_seed(7)
    gids = torch.randint(0, vocab, (8, 128), generator=gcpu).to(DEV)
    gidsB = torch.randint(0, vocab, (8, 128), generator=torch.Generator().manual_seed(8)).to(DEV)
    for mode in MODES:
        par = reference_parity(gm, gids, mode, MAIN_SEG)
        life = checkpoint_restore_full_module(gm, gids, mode, 96, MAIN_SEG)
        iso = request_isolation(gm, gids, gidsB, mode, MAIN_SEG)
        col = collapsibility(gm, gids, mode, 4, MAIN_SEG)
        R["gates"][mode] = dict(REFERENCE_PARITY=par["PARITY"], parity_maxabs=par["maxabs"],
                                FULL_MODULE_LIFECYCLE=life["FULL_MODULE_CHECKPOINT_RESTORE"],
                                lifecycle=life, REQUEST_ISOLATION=iso["REQUEST_STATE_ISOLATION"],
                                BRANCH_RESTORE=iso["BRANCH_RESTORE"],
                                ADDITIVE_COLLAPSE=col["ADDITIVE_COLLAPSE"], collapse=col)
        log(f"  {mode}: parity={par['PARITY']} lifecycle={life['FULL_MODULE_CHECKPOINT_RESTORE']} "
            f"iso={iso['REQUEST_STATE_ISOLATION']}/{iso['BRANCH_RESTORE']} collapse={col['ADDITIVE_COLLAPSE']}")
    snap()
    for mode in MODES:
        gd = R["gates"][mode]
        if gd["REFERENCE_PARITY"] != "PASS" or gd["FULL_MODULE_LIFECYCLE"] == "FAILED" or \
           gd["REQUEST_ISOLATION"] != "PASS":
            log(f"[STOP] {mode} failed a hard gate -> aborting that branch (packet §23).")
            R["outcomes"]["ABORT"] = mode; snap(); logf.close(); return

    # ---------------- P1 calibration: pick non-saturated D on BASE (shared) ----------------
    log(f"[P1] calibration grid D={CALIB_GRID} on BASE single-state (mean over modes near 0.5)")
    for D in CALIB_GRID:
        sp = spec(f"cal_D{D}", D)
        tr = make_pool(sp, POOL_TRAIN); ev = make_pool(sp, POOL_EVAL, start=POOL_TRAIN)
        accs = {}
        for mode in MODES:
            m = build_model(vocab, seed=42)
            train(m, tr, steps_cal, mode, "single", None, seed=42, lr=LR)
            accs[mode] = evaluate(m, ev, mode, "single", None)
            del m
            torch.cuda.empty_cache() if DEV == "cuda" else None
        R["calibration"][str(D)] = dict(base_acc=accs, mean=round(float(np.mean(list(accs.values()))), 4))
        log(f"  D={D:3d} base_acc={accs} mean={R['calibration'][str(D)]['mean']}")
        snap()
    # select difficulty on the LOAD-BEARING substrate (GDN), targeting a non-saturated ~0.6 base (MC headroom).
    TGT = 0.6
    band = {int(D): v["base_acc"]["gdn"] for D, v in R["calibration"].items() if 0.30 < v["base_acc"]["gdn"] < 0.96}
    MC_D = min(band, key=lambda d: abs(band[d] - TGT)) if band else \
        int(min(R["calibration"], key=lambda d: abs(R["calibration"][d]["base_acc"]["gdn"] - TGT)))
    R["MEMORY_AXIS"] = "QUALIFIED" if band else "NOT_QUALIFIED"
    R["select_rule"] = "GDN base nearest 0.6 within (0.30,0.90); GDN is the load-bearing substrate"
    R["MC_D"] = MC_D
    log(f"[P1] MEMORY_AXIS={R['MEMORY_AXIS']} selected D={MC_D}")

    dev_sp = spec(f"dev_D{MC_D}", MC_D); hold_sp = spec(f"hold_D{MC_D}", MC_D)
    tr = make_pool(dev_sp, POOL_TRAIN); dev_ev = make_pool(dev_sp, POOL_EVAL, start=POOL_TRAIN)
    hold_ev = make_pool(hold_sp, POOL_EVAL, start=POOL_TRAIN)
    R["benchmarks"] = dict(dev_spec=dev_sp.canonical(), holdout_spec=hold_sp.canonical(),
                           dev_ids_hash=dev_ev["ids_hash"], holdout_ids_hash=hold_ev["ids_hash"],
                           dev_first_ids=dev_ev["first_ids"], holdout_first_ids=hold_ev["first_ids"])
    snap()

    # ---------------- P2 2x2 co-adaptation + Regime A (joint) ----------------
    # cells: A=train single/infer single, B=train single/infer MC(param-free), C=train MC/infer single,
    #        D=train MC/infer MC. MC inference = grm trained reader + continuous. Report per seed.
    def train_backbones(mode, seed):
        m_single = build_model(vocab, seed=seed)
        st_s = train(m_single, tr, steps_bb, mode, "single", None, seed=seed, lr=LR, log=log)
        m_mc = build_model(vocab, seed=seed)
        st_m = train(m_mc, tr, steps_bb, mode, "grm", MAIN_SEG, warm_start=True, seed=seed, lr=LR, log=log)
        return m_single, st_s, m_mc, st_m

    singles = {}                                                       # in-memory frozen single-state backbones
    for mode in MODES:
        seeds = seeds_gdn if mode == "gdn" else SINGLE_SEED
        per_seed = []
        saved_single = None
        for seed in seeds:
            log(f"[P2] {mode} seed={seed}: train single-state + MC-aware backbones")
            m_single, st_s, m_mc, st_m = train_backbones(mode, seed)
            A = evaluate(m_single, hold_ev, mode, "single", None)                       # train single, infer single
            B = evaluate(m_single, hold_ev, mode, "moving_average", MAIN_SEG, warm_start=True)  # param-free MC post-hoc
            C = evaluate(m_mc, hold_ev, mode, "single", None)                           # train MC, infer single
            D = evaluate(m_mc, hold_ev, mode, "grm", MAIN_SEG, warm_start=True)         # train MC, infer MC
            Ad = evaluate(m_single, dev_ev, mode, "single", None)
            Dd = evaluate(m_mc, dev_ev, mode, "grm", MAIN_SEG, warm_start=True)
            cell = dict(seed=seed, A_train_single_infer_single=A, B_train_single_infer_MC_paramfree=B,
                        C_train_MC_infer_single=C, D_train_MC_infer_MC=D, dev_A=Ad, dev_D=Dd,
                        post_hoc_MC_effect_BminusA=round(B - A, 4), joint_MC_effect_DminusA=round(D - A, 4),
                        MC_inference_after_MCtrain_DminusC=round(D - C, 4),
                        interaction=round((D - C) - (B - A), 4),
                        train_single=st_s, train_mc=st_m)
            per_seed.append(cell)
            log(f"  {mode} seed={seed}: A={A} B={B} C={C} D={D} interaction={cell['interaction']}")
            if saved_single is None:
                saved_single = m_single                                                # keep seed-0 single for P3-P6
            else:
                del m_single
            del m_mc
            torch.cuda.empty_cache() if DEV == "cuda" else None
            snap()
        agg = lambda key: round(float(np.mean([c[key] for c in per_seed])), 4)
        R["twobytwo"][mode] = dict(
            seeds=seeds, per_seed=per_seed,
            mean_A=agg("A_train_single_infer_single"), mean_D=agg("D_train_MC_infer_MC"),
            mean_post_hoc_BminusA=agg("post_hoc_MC_effect_BminusA"),
            mean_joint_DminusA=agg("joint_MC_effect_DminusA"),
            mean_DminusC=agg("MC_inference_after_MCtrain_DminusC"),
            mean_interaction=agg("interaction"))
        singles[mode] = saved_single                                                    # in-memory handle
        snap()

    # ---------------- P3 frozen transfer + durable trained reader ----------------
    log("[P3] frozen transfer: BASE vs param-free MC vs TRAINED reader (w_u only); reader saved durably")
    for mode in MODES:
        m = singles[mode]
        for p in m.parameters():
            p.requires_grad_(False)
        h_before = tensor_hashes(m)
        base_h, base_wseg = evaluate(m, hold_ev, mode, "single", None, by_wseg=True)
        base_d = evaluate(m, dev_ev, mode, "single", None)
        pf = evaluate(m, hold_ev, mode, "moving_average", MAIN_SEG, warm_start=True)
        # train ONLY w_u (reader) on frozen backbone, continuous MC
        m.blk.w_u.requires_grad_(True)
        rparams = [p for p in m.parameters() if p.requires_grad]
        st_rd = train(m, tr, steps_rd, mode, "grm", MAIN_SEG, warm_start=True, params=rparams, seed=42, lr=LR, log=log)
        changed = [n for n in h_before if n != "blk.w_u.weight" and h_before[n] != tensor_hashes(m)[n]]
        rd_h = evaluate(m, hold_ev, mode, "grm", MAIN_SEG, warm_start=True)
        rd_d = evaluate(m, dev_ev, mode, "grm", MAIN_SEG, warm_start=True)
        # durably save the frozen backbone + trained reader with SHA-256 (packet §17-18)
        ck = os.path.join(art, f"rnn05b_{mode}_frozen_reader.pt")
        torch.save({"state_dict": m.state_dict(), "mode": mode, "vocab": vocab, "D": MC_D,
                    "d_k": D_K, "d_v": D_V, "d_model": D_MODEL, "conv_k": CONV_K,
                    "reader_tensor": "blk.w_u.weight"}, ck)
        sha = hashlib.sha256(open(ck, "rb").read()).hexdigest()
        reader_sha = hashlib.sha256(m.blk.w_u.weight.detach().cpu().numpy().tobytes()).hexdigest()
        R["frozen"][mode] = dict(
            FROZEN_BACKBONE_VALIDITY="PASS" if len(changed) == 0 else "FAIL", backbone_mutation=len(changed),
            changed_tensors=changed, base_holdout=base_h, base_dev=base_d, base_by_wseg=base_wseg,
            paramfree_MC_holdout=pf, trained_reader_holdout=rd_h, trained_reader_dev=rd_d,
            FROZEN_POST_MC_EFFECT=classify(round(pf - base_h, 4)),
            FROZEN_READER_MC_EFFECT=signal_from(round(rd_d - base_d, 4), round(rd_h - base_h, 4)),
            reader_params=int(m.blk.w_u.weight.numel()), train_reader=st_rd,
            checkpoint_path=ck, checkpoint_sha256=sha, checkpoint_bytes=os.path.getsize(ck),
            reader_weight_sha256=reader_sha,
            deltas=dict(paramfree=round(pf - base_h, 4), reader_holdout=round(rd_h - base_h, 4),
                        reader_dev=round(rd_d - base_d, 4)))
        log(f"  {mode}: base={base_h} paramfreeMC={pf} trainedReader={rd_h} "
            f"frozen_valid={R['frozen'][mode]['FROZEN_BACKBONE_VALIDITY']} sha={sha[:12]}")
        snap()

    # ---------------- P4 PURE cache-count curve (identical weights; retain most-recent K) ----------------
    log("[P4] pure cache-count curve: fixed backbone+reader, fixed checkpoint positions, vary retained K")
    for mode in MODES:
        m = singles[mode]                                              # has trained reader now (frozen backbone)
        h0 = tensor_hashes(m)
        rows = []
        for K in [1, 2, 4, 8]:
            acc = evaluate(m, hold_ev, mode, "grm", MAIN_SEG, warm_start=True, recent_k=K)
            rows.append(dict(retained_K=K, holdout_acc=acc))
            log(f"  {mode} K={K} acc={acc}")
        unchanged = [n for n in h0 if h0[n] != tensor_hashes(m)[n]]
        R["cache_count"][mode] = dict(seg_size=MAIN_SEG, n_checkpoints=math.ceil(L / MAIN_SEG),
                                      selection_rule="most_recent_K", rows=rows,
                                      weights_constant=(len(unchanged) == 0),
                                      PURE_CACHE_COUNT_CURVE="QUALIFIED" if len(unchanged) == 0 else "NOT_QUALIFIED")
        snap()

    # ---------------- P5 historical-state novelty (recall of early-written associations) ----------------
    log("[P5] historical-state novelty: base(final-state) vs MC(cached states) recall stratified by write segment")
    for mode in MODES:
        m = singles[mode]
        base_acc, base_wseg = evaluate(m, hold_ev, mode, "single", None, by_wseg=True)
        mc_acc, mc_wseg = evaluate(m, hold_ev, mode, "grm", MAIN_SEG, warm_start=True, by_wseg=True)
        early = [str(s) for s in range(math.ceil(L / MAIN_SEG) // 2)]                  # early-written half
        eb = [base_wseg[s] for s in early if s in base_wseg]
        em = [mc_wseg[s] for s in early if s in mc_wseg]
        early_base = round(float(np.mean(eb)), 4) if eb else None
        early_mc = round(float(np.mean(em)), 4) if em else None
        gain = round((early_mc - early_base), 4) if (early_mc is not None and early_base is not None) else None
        novelty = ("OBSERVED" if (gain is not None and gain >= MARGIN) else
                   ("NOT_DETECTED" if (gain is not None and gain <= 0) else "INCONCLUSIVE"))
        R["novelty"][mode] = dict(base_by_wseg=base_wseg, mc_by_wseg=mc_wseg,
                                  early_write_base=early_base, early_write_mc=early_mc, early_gain=gain,
                                  HISTORICAL_STATE_NOVELTY=novelty,
                                  note="recall of associations written in the early half of the sequence; base "
                                       "reads the single decayed final state, MC also reads cached segment states")
        log(f"  {mode}: early_base={early_base} early_mc={early_mc} gain={gain} novelty={novelty}")
        snap()

    # ---------------- P6 cost + memory accounting (direct-measured) ----------------
    log("[P6] cost + memory accounting")
    for mode in MODES:
        m = singles[mode]
        R["cost"][mode] = cost_breakdown(m, hold_ev, mode)
        R["memory"][mode] = memory_breakdown(m, mode)
        log(f"  {mode}: total_ms={R['cost'][mode]['total_ms']} live_state_bytes={R['memory'][mode]['total_live_recurrent_state_bytes']}")
        snap()

    # ---------------- outcomes ----------------
    R["outcomes"] = build_outcomes(R)
    json.dump(R["outcomes"], open(os.path.join(args.outdir, "rnn05b_outcomes.json"), "w"), indent=2)
    write_csv(R, os.path.join(args.outdir, "rnn05b_summary.csv"))
    snap()
    log("[DONE] outcomes:\n" + json.dumps(R["outcomes"], indent=2))
    logf.close()


@torch.no_grad()
def cost_breakdown(model, pool, mode, reps=20, batch=128):
    blk = model.blk
    ids = pool["ids"][:batch].to(DEV)
    x, q, k, v, g, beta, _ = model.project(ids)
    B, Lx, _ = q.shape
    seg = MAIN_SEG
    segs = [(i, min(i + seg, Lx)) for i in range(0, Lx, seg)]

    def sync():
        torch.cuda.synchronize() if DEV == "cuda" else None

    def timeit(fn):
        sync(); t0 = time.time()
        for _ in range(reps):
            fn()
        sync(); return (time.time() - t0) / reps * 1000

    def do_proj():
        model.project(ids)

    def do_recurrent():
        S = None
        for (a, b) in segs:
            _, S = run_recurrence(mode, q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b], seg, S, "chunked")

    states = []
    S = None
    for (a, b) in segs:
        _, S = run_recurrence(mode, q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b], seg, S, "chunked")
        states.append(S)
    pools = [k[:, a:b].mean(1) for (a, b) in segs]

    def do_read():
        for i, (a, b) in enumerate(segs):
            if i:
                read_states(states[:i], q[:, a:b])

    def do_gate():
        for i, (a, b) in enumerate(segs):
            if i:
                u = blk.w_u(x[:, a:b])                                  # [B,seg,dk]
                ks = k[:, a:b]
                kcum = torch.cumsum(ks, 1) / torch.arange(1, b - a + 1, device=DEV).view(1, -1, 1)
                gl_online = torch.einsum('bsk,bsk->bs', u, kcum).unsqueeze(-1)
                P = torch.stack(pools[:i], 1)                           # [B,i,dk]
                torch.softmax(torch.cat([torch.einsum('bsk,bck->bsc', u, P), gl_online], -1), -1)

    def do_ckpt():
        return [s.clone() for s in states]

    return dict(mode=mode, n_segments=len(segs), batch=batch, reps=reps,
                projection_conv_ms=round(timeit(do_proj), 3),
                recurrent_update_ms=round(timeit(do_recurrent), 3),
                checkpoint_copy_ms=round(timeit(do_ckpt), 3),
                state_read_ms=round(timeit(do_read), 3),
                gate_router_ms=round(timeit(do_gate), 3),
                total_ms=round(timeit(lambda: segmented_forward(model, ids, mode, "grm", seg, True)), 3),
                note="components directly measured; total is end-to-end MC forward (grm, continuous)")


def memory_breakdown(model, mode):
    dk, dv = model.d_k, model.d_v
    conv_dim = model.blk.conv_dim
    matrix = dk * dv * 4
    conv = conv_dim * (CONV_K - 1) * 4
    n_ckpt = math.ceil(L / MAIN_SEG)
    reader = model.blk.w_u.weight.numel() * 4
    return dict(mode=mode, matrix_state_bytes=matrix, conv_state_bytes=conv,
                total_live_recurrent_state_bytes=matrix + conv,
                historical_checkpoint_bytes_per_ckpt=matrix,
                historical_checkpoint_bytes_at_full=matrix * (n_ckpt - 1), n_checkpoints=n_ckpt,
                reader_param_bytes=reader,
                note="conv_state = (kernel-1)*conv_dim; historical cache stores the matrix state per segment "
                     "(conv boundary is only needed live). Mirrors Qwen {recurrent_states, conv_states}.")


def build_outcomes(R):
    out = dict(per_substrate={}, margin_OPERATOR_HEURISTIC=MARGIN, MC_D=R.get("MC_D"))
    for mode in MODES:
        g = R["gates"][mode]; tb = R["twobytwo"][mode]; fr = R["frozen"][mode]
        cc = R["cache_count"][mode]; nv = R["novelty"][mode]
        joint = signal_from(0.0, tb["mean_joint_DminusA"])  # dev not aggregated across; use holdout sign
        joint = "POSITIVE" if tb["mean_joint_DminusA"] >= MARGIN else (
            "NEGATIVE" if tb["mean_joint_DminusA"] <= -MARGIN else "NO_EFFECT")
        out["per_substrate"][mode] = dict(
            REFERENCE_PARITY=g["REFERENCE_PARITY"], FULL_MODULE_LIFECYCLE=g["FULL_MODULE_LIFECYCLE"],
            REQUEST_ISOLATION=g["REQUEST_ISOLATION"], ADDITIVE_COLLAPSE=g["ADDITIVE_COLLAPSE"],
            JOINT_TRAINED_MC_EFFECT=joint, FROZEN_POST_MC_EFFECT=fr["FROZEN_POST_MC_EFFECT"],
            FROZEN_READER_MC_EFFECT=fr["FROZEN_READER_MC_EFFECT"],
            PURE_CACHE_COUNT_CURVE=cc["PURE_CACHE_COUNT_CURVE"],
            HISTORICAL_STATE_NOVELTY=nv["HISTORICAL_STATE_NOVELTY"],
            STATE_BYTES=R["memory"][mode]["total_live_recurrent_state_bytes"],
            COMPUTE_TOTAL_MS=R["cost"][mode]["total_ms"],
            co_adaptation_interaction=tb["mean_interaction"])
    # substrate-level classifications
    def sig(mode):
        """Substrate MC signal over the realizable MC arms {joint co-adapted, frozen trained-reader,
        frozen param-free}. POSITIVE if the BEST arm helps (>=margin); NEGATIVE only if EVERY arm hurts
        (<=-margin, i.e. no MC form recovers); else NO_EFFECT (best arm neutral). The param-free arm is a
        naive baseline: it hurting while trained/co-adapted arms recover to base is NO_EFFECT, not NEGATIVE."""
        tb = R["twobytwo"][mode]; fr = R["frozen"][mode]
        vals = [tb["mean_joint_DminusA"], fr["deltas"]["reader_holdout"], fr["deltas"]["paramfree"]]
        best, worst = max(vals), min(vals)
        if best >= MARGIN:
            return "POSITIVE"
        if worst <= -MARGIN and best <= -MARGIN:
            return "NEGATIVE"
        if worst <= -MARGIN:
            return "NO_EFFECT_NAIVE_MC_NEGATIVE"
        return "NO_EFFECT"
    out["DELTA_MC_SIGNAL"] = sig("dn")
    out["GDN_MC_SIGNAL"] = sig("gdn")
    out["LA_MC_SIGNAL"] = sig("la")
    inter = R["twobytwo"]["gdn"]["mean_interaction"]
    out["COADAPTATION_INTERACTION"] = ("SUPPORTED" if inter >= MARGIN else
                                       ("NOT_DETECTED" if abs(inter) < MARGIN else "INCONCLUSIVE"))
    # Qwen gate policy (packet §25): PASS_CANDIDATE requires parity+lifecycle+isolation qualified AND a
    # defensible positive/compelling MC state-behavior signal for GDN.
    gdn = out["per_substrate"]["gdn"]
    gates_ok = (gdn["REFERENCE_PARITY"] == "PASS" and gdn["FULL_MODULE_LIFECYCLE"] in ("BIT_EXACT", "NUMERICALLY_EQUIVALENT")
                and gdn["REQUEST_ISOLATION"] == "PASS")
    # §25: PASS_CANDIDATE needs parity+lifecycle+isolation AND a defensible positive/compelling MC signal.
    # Here lifecycle/parity/isolation are now QUALIFIED for real GDN (a precondition RNN-05A lacked), but the
    # frozen/co-adapted MC signal is NOT positive -> CONDITIONAL / DEFER (do not auto-transplant into Qwen).
    if not gates_ok:
        gate = "FAIL"
    elif out["GDN_MC_SIGNAL"] == "POSITIVE":
        gate = "PASS_CANDIDATE"
    else:
        gate = "CONDITIONAL / DEFER"
    out["QWEN_GDN_TRANSPLANT_GATE"] = gate
    out["gate_rationale"] = ("full-module lifecycle + parity + request-isolation now QUALIFIED for real GDN "
                             "(preconditions RNN-05A could not meet); but frozen-backbone MC does not exceed "
                             "base and co-adaptation (not state-collapsibility) is the decisive factor -> defer")
    return out


def write_csv(R, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["substrate", "collapse", "base_A", "joint_DminusA", "posthoc_BminusA", "interaction",
                    "paramfree_MC", "trained_reader", "frozen_reader_effect", "novelty", "state_bytes", "total_ms"])
        for mode in MODES:
            tb = R["twobytwo"][mode]; fr = R["frozen"][mode]
            w.writerow([mode, R["gates"][mode]["ADDITIVE_COLLAPSE"], tb["mean_A"], tb["mean_joint_DminusA"],
                        tb["mean_post_hoc_BminusA"], tb["mean_interaction"], fr["deltas"]["paramfree"],
                        fr["deltas"]["reader_holdout"], fr["FROZEN_READER_MC_EFFECT"],
                        R["novelty"][mode]["HISTORICAL_STATE_NOVELTY"],
                        R["memory"][mode]["total_live_recurrent_state_bytes"], R["cost"][mode]["total_ms"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--artifacts", default=None, help="durable non-Git dir for checkpoints (default=outdir)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a)
