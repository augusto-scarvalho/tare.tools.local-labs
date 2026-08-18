#!/usr/bin/env python
"""
RNN-04 experiment: Memory Caching arms on toy MQAR (packet sections 11-18, 24).

Phases:
  C1 CALIBRATE (sections 5/10 discriminative regime): train BASE single-state across a predeclared
     difficulty grid; deterministic rule selects MC_TASK (base_acc nearest 0.5 in (0.2,0.9)); else
     MEMORY_AXIS=NOT_QUALIFIED.
  C2 MAIN comparison at MC_TASK: A single-state (BASE_RNN), B GRM (context gate), C equal-memory control
     (single state, ~same total bytes as B's cache), B0 residual (training-free on A; predicted collapse),
     Post moving-average (training-free on A, section 18). GATE on B vs A; if signal -> SSC (learned Top-k)
     + D random-selection control (eval on SSC weights).
  C3 SWEEP (sections 14/15): #cached states N in {1,2,4,8,16} via seg_size = L/N -> accuracy/latency/bytes.
     accuracy-vs-seq_len sweep for base vs GRM. accuracy-vs-distance from per-answer distance bins.
  C4 REPLICATE (section 24): re-evaluate the selected arms on a FRESH pinned spec (new seeds/examples).

Storage vs compute separated (section 16). float32. Isolated venv. No large LM training.
Usage: python rnn_mc_experiment.py --outdir <dir> [--smoke]
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from rnn_mc_bench import MQARSpec, make_example
from rnn_mc_substrate import MQARModel

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float32
SEED = 42

# predeclared difficulty grid + selection rule (section 5/10) -- fixed BEFORE any arm comparison
CALIB_GRID = [24, 40, 56, 72]            # num_pairs D (ascending difficulty), > single-state capacity ~d_k
CALIB_L, CALIB_Q, CALIB_DENS = 256, 8, 0.3
NUM_KEYS, NUM_VALS = 128, 64             # keys > max D so distractor pool stays disjoint
SELECT_RULE = "num_pairs whose BASE single-state acc is nearest 0.5 within (0.20,0.90); else NOT_QUALIFIED"

# model / training defaults (tiny; minutes not hours; section 8)
D_MODEL, D_K, D_V = 128, 24, 24
MAIN_SEG = 32                             # L/seg = 8 segments for the main comparison
POOL_TRAIN, POOL_EVAL = 4096, 512


def spec(name, L=CALIB_L, D=16, Q=CALIB_Q, dens=CALIB_DENS):
    return MQARSpec(seq_len=L, num_pairs=D, num_queries=Q, distractor_density=dens,
                    num_keys=NUM_KEYS, num_vals=NUM_VALS, name=name)


def make_pool(sp, n, start=0):
    ex = [make_example(sp, i) for i in range(start, start + n)]
    ids = torch.tensor([e["input_ids"] for e in ex], dtype=torch.long)
    lab = torch.tensor([e["labels"] for e in ex], dtype=torch.long)
    dist = torch.tensor([[p["distance"] for p in e["pairs"]] for e in ex], dtype=torch.long)
    apos = torch.tensor([e["answer_positions"] for e in ex], dtype=torch.long)
    return dict(ids=ids, lab=lab, dist=dist, apos=apos, n=n)


def build_model(vocab, d_k=D_K, d_v=D_V):
    torch.manual_seed(SEED)
    return MQARModel(vocab, d_model=D_MODEL, d_k=d_k, d_v=d_v).to(DEV).to(DT)


def train(model, pool, steps, agg, seg, batch=128, lr=1e-3, warm_start=False, ssc_k=2, log=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator(device="cpu").manual_seed(SEED)
    model.train()
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats() if DEV == "cuda" else None
    for s in range(steps):
        idx = torch.randint(0, pool["n"], (batch,), generator=g)
        ids = pool["ids"][idx].to(DEV); lab = pool["lab"][idx].to(DEV)
        logits = model(ids, agg=agg, seg_size=seg, warm_start=warm_start, ssc_k=ssc_k)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), lab.view(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if log and (s % 200 == 0 or s == steps - 1):
            log(f"    step {s:4d}/{steps} loss {loss.item():.4f}")
    wall = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 1e6) if DEV == "cuda" else 0.0
    return dict(steps=steps, wall_s=round(wall, 1), final_loss=round(loss.item(), 4),
                peak_vram_mb=round(peak, 1), trainable=sum(p.numel() for p in model.parameters()))


@torch.no_grad()
def evaluate(model, pool, agg, seg, batch=256, warm_start=False, ssc_k=2, ssc_random=False, by_distance=False):
    model.eval()
    gen = torch.Generator(device=DEV).manual_seed(0) if ssc_random else None
    correct = total = 0
    dist_correct, dist_total = {}, {}
    for i in range(0, pool["n"], batch):
        ids = pool["ids"][i:i + batch].to(DEV); lab = pool["lab"][i:i + batch].to(DEV)
        logits = model(ids, agg=agg, seg_size=seg, warm_start=warm_start, ssc_k=ssc_k,
                       ssc_random=ssc_random, gen=gen)
        pred = logits.argmax(-1)
        mask = lab != -100
        correct += (pred[mask] == lab[mask]).sum().item(); total += mask.sum().item()
        if by_distance:
            apos = pool["apos"][i:i + batch]; dst = pool["dist"][i:i + batch]
            for b in range(ids.size(0)):
                for j, p in enumerate(apos[b].tolist()):
                    d = int(dst[b, j].item()); ok = int(pred[b, p].item() == lab[b, p].item())
                    bkt = (d // 32) * 32
                    dist_correct[bkt] = dist_correct.get(bkt, 0) + ok
                    dist_total[bkt] = dist_total.get(bkt, 0) + 1
    acc = correct / total if total else 0.0
    if by_distance:
        curve = {str(k): round(dist_correct[k] / dist_total[k], 4) for k in sorted(dist_total)}
        return round(acc, 4), curve
    return round(acc, 4)


@torch.no_grad()
def measure_cost(model, pool, agg, seg, batch=128, ssc_k=2, reps=5):
    """Section 16: separate storage (state/cache bytes, peak vram) from compute (read/update/total time)."""
    model.eval()
    ids = pool["ids"][:batch].to(DEV)
    _, si = model(ids, agg=agg, seg_size=seg, ssc_k=ssc_k, return_state_info=True)
    # total inference time
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        model(ids, agg=agg, seg_size=seg, ssc_k=ssc_k)
    if DEV == "cuda":
        torch.cuda.synchronize()
    total_ms = (time.time() - t0) / reps * 1000
    # 'single' baseline time isolates the state-update/scan cost; delta = aggregation/read cost
    t0 = time.time()
    for _ in range(reps):
        model(ids, agg="single", seg_size=None)
    if DEV == "cuda":
        torch.cuda.synchronize()
    single_ms = (time.time() - t0) / reps * 1000
    return dict(n_cached=si["n_cached"], state_bytes_per_req=si["state_bytes_per_req"],
                total_cache_bytes=si["total_cache_bytes"], dk=si["dk"], dv=si["dv"],
                total_infer_ms=round(total_ms, 2), update_scan_ms=round(single_ms, 2),
                aggregation_read_ms=round(total_ms - single_ms, 2))


def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    logf = open(os.path.join(args.outdir, "run.log"), "a")
    def log(m):
        print(m); logf.write(m + "\n"); logf.flush()
    R = dict(meta=dict(device=DEV, dtype="float32", seed=SEED, d_model=D_MODEL, d_k=D_K, d_v=D_V,
                       main_seg=MAIN_SEG, torch=torch.__version__, numpy=np.__version__,
                       select_rule=SELECT_RULE, calib_grid=CALIB_GRID),
             calibration={}, arms={}, gate={}, sweep={}, cost={}, replication={})
    def snap():
        json.dump(R, open(os.path.join(args.outdir, "rnn04_results.json"), "w"), indent=2)

    vocab = spec("v").vocab_size
    steps_c = 300 if args.smoke else 2500
    steps_m = 300 if args.smoke else 5000
    steps_s = 300 if args.smoke else 3000

    # ---- C1 calibration ----
    log(f"[C1] calibration on grid D={CALIB_GRID} (steps={steps_c}) dev={DEV}")
    for D in CALIB_GRID:
        sp = spec(f"calib_D{D}", D=D)
        tr = make_pool(sp, POOL_TRAIN); ev = make_pool(sp, POOL_EVAL, start=POOL_TRAIN)
        m = build_model(vocab)
        st = train(m, tr, steps_c, "single", None, log=log)
        acc = evaluate(m, ev, "single", None)
        R["calibration"][str(D)] = dict(base_acc=acc, **st)
        log(f"  D={D:3d} base_acc={acc:.4f} ({st['wall_s']}s)")
        del m; torch.cuda.empty_cache() if DEV == "cuda" else None
        snap()
    band = {int(D): v["base_acc"] for D, v in R["calibration"].items() if 0.20 < v["base_acc"] < 0.90}
    if band:
        MC_D = min(band, key=lambda d: abs(band[d] - 0.5))
        R["MEMORY_AXIS"] = "QUALIFIED"
    else:
        MC_D = max(R["calibration"], key=lambda d: -abs(R["calibration"][d]["base_acc"] - 0.5))
        MC_D = int(MC_D); R["MEMORY_AXIS"] = "NOT_QUALIFIED"
    R["MC_TASK"] = dict(num_pairs=MC_D, seq_len=CALIB_L, num_queries=CALIB_Q, density=CALIB_DENS,
                        selected_base_acc=R["calibration"][str(MC_D)]["base_acc"])
    log(f"[C1] MEMORY_AXIS={R['MEMORY_AXIS']} MC_TASK D={MC_D} base_acc={R['MC_TASK']['selected_base_acc']}")
    snap()

    # ---- C2 main comparison ----
    sp = spec(f"main_D{MC_D}", D=MC_D)
    tr = make_pool(sp, POOL_TRAIN); ev = make_pool(sp, POOL_EVAL, start=POOL_TRAIN)
    N_main = math.ceil(CALIB_L / MAIN_SEG)
    log(f"[C2] main comparison D={MC_D} seg={MAIN_SEG} N={N_main} (steps={steps_m})")

    # A base single
    mA = build_model(vocab); stA = train(mA, tr, steps_m, "single", None, log=log)
    accA = evaluate(mA, ev, "single", None, by_distance=True)
    R["arms"]["A_base_single"] = dict(acc=accA[0], dist_curve=accA[1], train=stA, agg="single", seg=None)
    # B0 residual (training-free on A) -- predicted collapse
    R["arms"]["B0_residual_free"] = dict(acc=evaluate(mA, ev, "residual", MAIN_SEG),
                                         agg="residual", seg=MAIN_SEG, note="training-free on A weights")
    # Post moving-average (training-free on A, section 18)
    R["arms"]["Post_moving_avg_free"] = dict(acc=evaluate(mA, ev, "moving_average", MAIN_SEG),
                                             agg="moving_average", seg=MAIN_SEG,
                                             note="frozen A weights, param-free (POST_TRAINING_MC)")
    R["cost"]["A_single"] = measure_cost(mA, ev, "single", None)
    snap(); del mA; torch.cuda.empty_cache() if DEV == "cuda" else None

    # B GRM
    mB = build_model(vocab); stB = train(mB, tr, steps_m, "grm", MAIN_SEG, log=log)
    accB = evaluate(mB, ev, "grm", MAIN_SEG, by_distance=True)
    R["arms"]["B_grm"] = dict(acc=accB[0], dist_curve=accB[1], train=stB, agg="grm", seg=MAIN_SEG)
    R["cost"]["B_grm"] = measure_cost(mB, ev, "grm", MAIN_SEG)
    snap()

    # C equal-memory control: single state with ~N_main x more bytes (d' = d*sqrt(N))
    d_big = int(round(D_K * math.sqrt(N_main)))
    mC = build_model(vocab, d_k=d_big, d_v=d_big); stC = train(mC, tr, steps_m, "single", None, log=log)
    R["arms"]["C_equal_mem_single"] = dict(acc=evaluate(mC, ev, "single", None), train=stC, agg="single",
                                           seg=None, d_k=d_big, d_v=d_big,
                                           state_bytes=d_big * d_big * 4,
                                           note=f"single state matched to B cache bytes (~{N_main}x)")
    R["cost"]["C_equal_mem"] = measure_cost(mC, ev, "single", None)
    snap(); del mC; torch.cuda.empty_cache() if DEV == "cuda" else None

    # GATE (section 12): does B beat A?
    gate_delta = round(R["arms"]["B_grm"]["acc"] - R["arms"]["A_base_single"]["acc"], 4)
    R["gate"] = dict(B_minus_A=gate_delta, threshold_OPERATOR_HEURISTIC=0.03,
                     signal=bool(gate_delta >= 0.03))
    log(f"[C2] GATE B-A={gate_delta} signal={R['gate']['signal']}")
    snap()

    # SSC + random control only if B shows signal
    if R["gate"]["signal"]:
        log("[C2] gate PASSED -> SSC (learned Top-k) + random control (arm D)")
        mS = build_model(vocab); stS = train(mS, tr, steps_m, "ssc", MAIN_SEG, ssc_k=2, log=log)
        R["arms"]["SSC_learned_k2"] = dict(acc=evaluate(mS, ev, "ssc", MAIN_SEG, ssc_k=2),
                                           train=stS, agg="ssc", seg=MAIN_SEG, ssc_k=2)
        R["arms"]["D_ssc_random_k2"] = dict(acc=evaluate(mS, ev, "ssc", MAIN_SEG, ssc_k=2, ssc_random=True),
                                            agg="ssc", seg=MAIN_SEG, ssc_k=2, ssc_random=True,
                                            note="same weights as SSC, RANDOM selection (section 11D)")
        R["cost"]["SSC_k2"] = measure_cost(mS, ev, "ssc", MAIN_SEG, ssc_k=2)
        del mS; torch.cuda.empty_cache() if DEV == "cuda" else None
    else:
        R["arms"]["SSC_learned_k2"] = dict(skipped="gate did not pass (section 12 ordering)")
    snap()

    # ---- C3 memory sweep (accuracy/latency/bytes vs N) ----
    log(f"[C3] memory sweep N in [1,2,4,8,16] (steps={steps_s})")
    for N in [1, 2, 4, 8, 16]:
        seg = math.ceil(CALIB_L / N)
        agg = "single" if N == 1 else "grm"
        m = build_model(vocab); st = train(m, tr, steps_s, agg, (None if N == 1 else seg), log=log)
        acc = evaluate(m, ev, agg, (None if N == 1 else seg))
        cost = measure_cost(m, ev, agg, (None if N == 1 else seg))
        R["sweep"][str(N)] = dict(N=N, seg_size=(CALIB_L if N == 1 else seg), agg=agg, acc=acc,
                                  total_cache_bytes=cost["total_cache_bytes"],
                                  state_bytes_per_req=cost["state_bytes_per_req"],
                                  total_infer_ms=cost["total_infer_ms"],
                                  aggregation_read_ms=cost["aggregation_read_ms"])
        log(f"  N={N:2d} seg={seg} acc={acc:.4f} cache_bytes={cost['total_cache_bytes']} "
            f"infer_ms={cost['total_infer_ms']}")
        del m; torch.cuda.empty_cache() if DEV == "cuda" else None
        snap()

    # accuracy vs seq_len (base vs GRM) -- light
    R["sweep_seqlen"] = {}
    for L in ([256] if args.smoke else [256, 384, 512]):    # all >= body 2*num_pairs for MC_D
        spL = spec(f"len_D{MC_D}_L{L}", L=L, D=MC_D)
        trL = make_pool(spL, POOL_TRAIN); evL = make_pool(spL, POOL_EVAL, start=POOL_TRAIN)
        mAl = build_model(vocab); train(mAl, trL, steps_s, "single", None)
        mBl = build_model(vocab); train(mBl, trL, steps_s, "grm", MAIN_SEG)
        R["sweep_seqlen"][str(L)] = dict(base=evaluate(mAl, evL, "single", None),
                                         grm=evaluate(mBl, evL, "grm", MAIN_SEG),
                                         N=math.ceil(L / MAIN_SEG))
        log(f"  L={L} base={R['sweep_seqlen'][str(L)]['base']} grm={R['sweep_seqlen'][str(L)]['grm']}")
        del mAl, mBl; torch.cuda.empty_cache() if DEV == "cuda" else None
        snap()

    # ---- C4 replication on FRESH pinned spec (section 24) ----
    log("[C4] replication on fresh seeds (new spec name -> new examples)")
    spR = spec(f"replication_D{MC_D}", D=MC_D)
    trR = make_pool(spR, POOL_TRAIN); evR = make_pool(spR, POOL_EVAL, start=POOL_TRAIN)
    mAr = build_model(vocab); train(mAr, trR, steps_m, "single", None)
    mBr = build_model(vocab); train(mBr, trR, steps_m, "grm", MAIN_SEG)
    R["replication"] = dict(spec=spR.canonical(), base=evaluate(mAr, evR, "single", None),
                            grm=evaluate(mBr, evR, "grm", MAIN_SEG))
    R["replication"]["grm_minus_base"] = round(R["replication"]["grm"] - R["replication"]["base"], 4)
    log(f"[C4] replication base={R['replication']['base']} grm={R['replication']['grm']} "
        f"delta={R['replication']['grm_minus_base']}")
    snap()
    log("[DONE]")
    print(json.dumps(dict(MEMORY_AXIS=R["MEMORY_AXIS"], MC_TASK=R["MC_TASK"], gate=R["gate"],
                          arms={k: R["arms"][k].get("acc") for k in R["arms"]},
                          replication=R["replication"]), indent=2))
    logf.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    run(ap.parse_args())
