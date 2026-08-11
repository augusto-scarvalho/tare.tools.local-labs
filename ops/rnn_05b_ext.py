#!/usr/bin/env python
"""
RNN-05B-EXT: Historical-state recoverability OUTSIDE the ceiling (direct H3 test).

RNN-05B closed ceiling-limited (GDN base ~0.97): the frozen-backbone Memory-Caching signal was NO_EFFECT and
the pure-cache-count curve gave only a WEAK_DIRECTIONAL_SIGNAL for DN/GDN. That test could not distinguish "MC
adds nothing" from "the task left no room to add anything." This packet asks the narrow question:

  When a stable DN/GDN backbone actually LOSES an old association from its final recurrent state, can a
  HISTORICAL recurrent-state snapshot recover that association and produce a measurable improvement over the
  SAME frozen backbone?

Hypotheses:
  H3   : historical DN/GDN states contain recoverable task information no longer sufficiently represented in
         the final recurrent state.
  NULL : historical snapshots do not add recoverable information beyond the final state under a stable regime.
  LA is the mechanistic control (additive/collapsible final state -> should not benefit from redundant history).

Design discipline (see PRE_REGISTRATION.md, written BEFORE any outcome-bearing run):
  * Do NOT inherit RNN-05B's calibration band. New headroom rule: 0.40 <= GDN BASE <= 0.80 (operator-design
    criterion for TESTABILITY, not a scientific noise floor); per-seed 0.20..0.90; else H3_TESTABILITY=BLOCKED.
  * Single-source config object -> PRE_REGISTRATION.md == preregistration.json == executed constants
    (CALIBRATION_RULE_IDENTITY self-check).
  * Temporal-memory pressure, NOT capacity overload: pairs kept far below the RNN-05B capacity cliff; recall is
    made hard by long write->query distance + distractor interference over a long retention gap.
  * Reuse the RNN-05B-qualified LA/DN/GDN toy family UNCHANGED (rnn_delta_substrate). No recurrence edits, no
    deeper readers, no GDN-mechanism edits to manufacture forgetting.
  * Frozen-backbone load-bearing comparison: A base / B param-free MC / C trained reader (w_u only, backbone
    mutation must be 0). Reader saved durably + SHA-256.
  * Recovery/harm decomposition (BASE-wrong -> MC-correct = recovery); snapshot attribution (DESCRIPTIVE) +
    small ablation; fixed-position pure cache-count curve; LA falsification control; DN vs GDN reported apart;
    small secondary 2x2. Effects reported as raw paired deltas across 3 training seeds (no p-values from n=3).

Touches NOTHING shared/immutable: this is a new module that imports the qualified substrate/generator building
blocks read-only. No Qwen, no llama.cpp/serving/deploy, no FLA, no push.

GOVERNING INTERPRETATION: runs/rnn/RNN-05B-EXT/AUDIT_RECONCILIATION.md (+ rnn05bext_audit_reconciliation.json)
supersede interpretation/protocol-scope. Key: PROTOCOL_GATE_ORDERING=FAILED (P2 MC/reader work runs before the
P1c stability gate) -> post-block MC results are EXPLORATORY_NON_LOAD_BEARING; the interference cliff is
TRAIN_PER_CONDITION_STABILITY=FAILED (FIXED_BACKBONE_GRADED_FORGETTING=NOT_TESTED); snapshot causal signal
INCONCLUSIVE/NOT_QUALIFIED. Result stands: H3_TESTABILITY=BLOCKED_BY_UNSTABLE_BASE, Qwen gate DEFER.

Usage:
  python rnn_05b_ext.py --preregister <dir>          # write PRE_REGISTRATION.md + JSON + selftests (no GPU)
  python rnn_05b_ext.py --selftest <json>            # memory-bound generator self-qualification (no GPU)
  python rnn_05b_ext.py --run --outdir <dir> [--artifacts <dir>] [--smoke]
"""
import argparse, csv, hashlib, json, math, os, sys, time
from dataclasses import dataclass, asdict, field
import numpy as np
import torch
import torch.nn.functional as F

from rnn_mc_bench import PAD, BOS, QSEP, FILL, KEY_LO, _example_id
from rnn_mc_substrate import read_states, agg_moving_average, agg_grm, gate_softmax, combine
from rnn_delta_substrate import (MQARDeltaModel, run_recurrence, reference_parity,
                                 checkpoint_restore_full_module, request_isolation, MODES)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float32


# ==================================================================================================
# 1. Single-source config object (PRE_REGISTRATION == JSON == executed constants)
# ==================================================================================================
@dataclass(frozen=True)
class ExtConfig:
    # --- architecture: RNN-05B-qualified family, UNCHANGED (rnn_delta_substrate.MQARDeltaModel) ---
    d_model: int = 128
    d_k: int = 64
    d_v: int = 64
    conv_k: int = 4
    seg: int = 64                          # MC segment size AND recurrence chunk size
    # --- task: memory-bound MQAR (early writes, long gap, late queries) ---
    num_pairs: int = 12                    # FAR below RNN-05B capacity cliff (~40 @ d_k=64): no capacity overload
    num_queries: int = 8
    num_keys: int = 128
    num_vals: int = 64
    write_frac: float = 0.25               # writes spread across the EARLY quarter of the body
    # --- predeclared challenge grid (cheap-first: seq_len asc, then distractor asc) ---
    seq_lens: tuple = (512, 768, 1024)     # retention-gap axis
    distractor_tiers: tuple = (("low", 0.15), ("med", 0.35), ("high", 0.55))   # interference axis
    # --- calibration headroom rule (THE single source of truth for difficulty selection) ---
    band_lo: float = 0.40                  # required GDN BASE mean band (testability window; NOT a noise floor)
    band_hi: float = 0.80
    seed_lo: float = 0.20                  # per-seed stability band on the selected condition
    seed_hi: float = 0.90
    # --- training ---
    steps_bb: int = 2500                   # BASE backbone steps
    steps_rd: int = 1800                   # frozen reader steps
    lr: float = 3e-3
    batch: int = 96
    pool_train: int = 4096
    pool_eval: int = 512
    # --- seeds (predeclared) ---
    gdn_seeds: tuple = (42, 43, 44)        # load-bearing
    dn_seeds: tuple = (42, 43, 44)         # load-bearing
    la_seeds: tuple = (42,)                # mechanistic control (>=1)
    # --- analysis ---
    margin: float = 0.03                   # OPERATOR_HEURISTIC effect band (NOT a measured noise floor)
    cache_K: tuple = (1, 2, 4, 8)
    recovery_margin: float = 0.02          # recovery must exceed harm by this to count as "meaningful"
    # amendment hook: a pre-committed explicit cheap-first condition list REPLACING the seq_len x tier product
    # (used only when the original coarse grid is BLOCKED by granularity; see AMENDMENT_*.md). Everything else
    # -- the headroom rule, seeds, steps, generator, analyses -- is UNCHANGED.
    grid_override_json: str = ""

    # ---- selection rule as a string, DERIVED from the constants (recorded==executed by construction) ----
    def select_rule(self):
        return (f"iterate the predeclared grid cheap-first (seq_len asc {list(self.seq_lens)}, then distractor "
                f"asc {[t[0] for t in self.distractor_tiers]}); SELECT the first condition whose GDN seed-"
                f"{self.gdn_seeds[0]} BASE holdout accuracy is within ({self.band_lo},{self.band_hi}); QUALIFY "
                f"it iff all {len(self.gdn_seeds)} GDN seeds are within ({self.seed_lo},{self.seed_hi}) and "
                f"their mean is within ({self.band_lo},{self.band_hi}); otherwise H3_TESTABILITY=BLOCKED with "
                f"NO nearest-condition fallback")

    def grid(self):
        """Predeclared, ordered candidate conditions (cheap-first). An amendment may pre-commit an explicit
        list via grid_override_json (finer distractor granularity); the selection RULE is unchanged."""
        if self.grid_override_json:
            conds = json.loads(self.grid_override_json)
            for c in conds:
                c.setdefault("num_pairs", self.num_pairs)
                c.setdefault("num_queries", self.num_queries)
                c.setdefault("write_frac", self.write_frac)
                c.setdefault("name", f"mb_L{c['seq_len']}_{c['distractor_tier']}")
            return conds
        out = []
        for sl in self.seq_lens:
            for tier, dens in self.distractor_tiers:
                out.append(dict(seq_len=sl, distractor_tier=tier, distractor_density=dens,
                                num_pairs=self.num_pairs, num_queries=self.num_queries,
                                write_frac=self.write_frac,
                                name=f"mb_L{sl}_{tier}"))
        return out


def calibration_rule_selfcheck(cfg: ExtConfig):
    """CALIBRATION_RULE_IDENTITY: the executed band constants, the serialized preregistration bounds, and the
    numbers parsed back out of the human-readable select_rule string must all be identical (RNN-05B audit §1)."""
    import re
    s = cfg.select_rule()
    nums = re.findall(r"within \(([\d.]+),([\d.]+)\)", s)
    parsed_band = (float(nums[0][0]), float(nums[0][1]))     # first "within (...)" = difficulty band
    parsed_seed = (float(nums[1][0]), float(nums[1][1]))     # second = per-seed band
    executed_band = (cfg.band_lo, cfg.band_hi)
    executed_seed = (cfg.seed_lo, cfg.seed_hi)
    ok = (parsed_band == executed_band and parsed_seed == executed_seed)
    return dict(CALIBRATION_RULE_IDENTITY="PASS" if ok else "FAIL",
                executed_band=list(executed_band), recorded_band=list(parsed_band),
                executed_seed_band=list(executed_seed), recorded_seed_band=list(parsed_seed),
                note="select_rule string is f-string-derived from the constants -> recorded==executed holds")


# ==================================================================================================
# 2. Memory-bound MQAR generator (self-contained; reuses qualified token layout + blake2b seeding)
#    early writes (write_frac of body) -> long retention gap w/ distractor interference -> late queries.
#    No answer leak (values live only at body write slots; query region = QSEP + query keys).
# ==================================================================================================
def _mb_canonical(cond, num_keys, num_vals):
    return json.dumps(dict(sorted(cond.items())) | dict(num_keys=num_keys, num_vals=num_vals),
                      sort_keys=True, separators=(",", ":"))


def _mb_seed(cond, num_keys, num_vals, idx):
    h = hashlib.blake2b((_mb_canonical(cond, num_keys, num_vals) + f"|idx={idx}").encode(), digest_size=8).digest()
    return int.from_bytes(h, "big")       # process-stable (NOT builtin hash())


def make_mb_example(cond, num_keys, num_vals, idx):
    seq_len, num_pairs, num_queries = cond["seq_len"], cond["num_pairs"], cond["num_queries"]
    write_frac, dens = cond["write_frac"], cond["distractor_density"]
    if num_queries > num_pairs:
        raise ValueError("num_queries > num_pairs")
    if num_pairs > num_keys:
        raise ValueError("num_pairs > num_keys")
    rng = np.random.default_rng(_mb_seed(cond, num_keys, num_vals, idx))
    val_lo = KEY_LO + num_keys

    key_perm = rng.permutation(num_keys)
    written_keys = (key_perm[:num_pairs] + KEY_LO).tolist()
    values = (rng.integers(0, num_vals, size=num_pairs) + val_lo).tolist()
    free_keys = (key_perm[num_pairs:] + KEY_LO).tolist()           # disjoint hard-distractor pool
    kv = dict(zip(written_keys, values))

    q_idx = rng.permutation(num_pairs)[:num_queries]               # random query order (no copy shortcut)
    queried_keys = [written_keys[i] for i in q_idx]
    query = [QSEP] + queried_keys

    body_len = seq_len - 1 - len(query)
    if body_len < 2 * num_pairs:
        raise ValueError(f"seq_len {seq_len} too short (body {body_len} < 2*num_pairs {2*num_pairs})")

    # write region = early write_frac of the body; one pair per equal chunk within it (spread over early segs)
    region = min(body_len, max(2 * num_pairs, int(round(write_frac * body_len))))
    chunk = max(2, region // num_pairs)
    body = [FILL] * body_len
    write_pos = {}
    for i, k in enumerate(written_keys):
        base = i * chunk
        if base + 1 >= body_len:
            raise ValueError("write region overflow")
        body[base], body[base + 1] = k, values[i]
        write_pos[k] = base
    occupied = set()
    for base in write_pos.values():
        occupied.update((base, base + 1))
    # distractors spread over ALL free slots (early gaps AND the long retention tail -> interference over the gap)
    free_slots = [p for p in range(body_len) if p not in occupied]
    n_hard = int(round(dens * len(free_slots)))
    if free_keys and n_hard:
        hard_pos = rng.choice(free_slots, size=min(n_hard, len(free_slots)), replace=False)
        for j, p in enumerate(hard_pos):
            body[int(p)] = free_keys[j % len(free_keys)]           # standalone key, no value follows

    input_ids = [BOS] + body + query
    assert len(input_ids) == seq_len
    labels = [-100] * seq_len
    q_start = 1 + body_len + 1
    answer_positions, pairs = [], []
    for j, k in enumerate(queried_keys):
        p = q_start + j
        labels[p] = kv[k]
        answer_positions.append(p)
        wp = write_pos[k] + 1                                      # absolute (account for BOS)
        pairs.append(dict(key=int(k), value=int(kv[k]), write_pos=int(wp), query_pos=int(p),
                          distance=int(p - wp)))
    return dict(input_ids=input_ids, labels=labels, answer_positions=answer_positions, pairs=pairs,
                example_id=_example_id(input_ids, labels), idx=idx, val_lo=val_lo, num_vals=num_vals)


def mb_vocab(cfg: ExtConfig):
    return KEY_LO + cfg.num_keys + cfg.num_vals


def mb_selftest(cfg: ExtConfig, out_path=None):
    """Self-qualify the memory-bound generator: length-exact, kv survives, NO answer leak, no lexical shortcut,
    process-stable seeding (fresh subprocess), and (design check) large write->query distance."""
    cond = cfg.grid()[0]
    ex = [make_mb_example(cond, cfg.num_keys, cfg.num_vals, i) for i in range(64)]
    checks = {}
    checks["length_exact"] = all(len(e["input_ids"]) == cond["seq_len"] for e in ex)
    surv = True
    for e in ex:
        for pr in e["pairs"]:
            wp = pr["write_pos"]
            surv &= (e["input_ids"][wp] == pr["key"] and e["input_ids"][wp + 1] == pr["value"]
                     and e["input_ids"][pr["query_pos"]] == pr["key"])
    checks["kv_survives"] = bool(surv)
    leak = False
    for e in ex:
        qregion = set(e["input_ids"][min(e["answer_positions"]):])
        for p in e["answer_positions"]:
            leak |= (e["labels"][p] in qregion)
    checks["no_answer_leak"] = (not leak)
    kmap = {}
    for e in ex:
        for pr in e["pairs"]:
            kmap.setdefault(pr["key"], set()).add(pr["value"])
    multi = sum(1 for v in kmap.values() if len(v) > 1)
    checks["no_lexical_shortcut"] = (multi >= max(1, len(kmap) // 2))
    dists = [pr["distance"] for e in ex for pr in e["pairs"]]
    checks["mean_write_query_distance"] = float(np.mean(dists))
    checks["min_write_query_distance"] = int(np.min(dists))
    # long-gap design check: median distance should be a large fraction of seq_len
    checks["memory_bound_design"] = bool(np.median(dists) > 0.5 * cond["seq_len"])
    # process stability: fresh subprocess must reproduce example ids
    code = ("import json,sys;sys.path.insert(0,r'%s');from rnn_05b_ext import make_mb_example;"
            "c=json.loads(sys.argv[1]);"
            "print(json.dumps([make_mb_example(c,%d,%d,i)['example_id'] for i in range(16)]))"
            % (os.path.dirname(os.path.abspath(__file__)), cfg.num_keys, cfg.num_vals))
    import subprocess
    a = subprocess.run([sys.executable, "-c", code, json.dumps(cond)], capture_output=True, text=True)
    inproc = json.dumps([make_mb_example(cond, cfg.num_keys, cfg.num_vals, i)["example_id"] for i in range(16)])
    checks["process_stable"] = (a.returncode == 0 and json.loads(a.stdout.strip()) == json.loads(inproc))
    passed = (checks["length_exact"] and checks["kv_survives"] and checks["no_answer_leak"]
              and checks["no_lexical_shortcut"] and checks["memory_bound_design"] and checks["process_stable"])
    res = dict(packet="RNN-05B-EXT", component="memory-bound MQAR generator self-qualification",
               MB_GENERATOR_SELFTEST="PASS" if passed else "FAIL",
               condition=cond, checks=checks, numpy=np.__version__, python=sys.version.split()[0])
    if out_path:
        json.dump(res, open(out_path, "w"), indent=2)
    print(json.dumps({k: res[k] for k in ["MB_GENERATOR_SELFTEST"]} | {"checks": checks}, indent=2))
    return res


# ==================================================================================================
# 3. Model / data / MC forward (flexible: single | moving_average | grm; recent_k or explicit snapshot mask;
#    optional last-segment gate capture for attribution)
# ==================================================================================================
def build_model(cfg, vocab, seed=42):
    torch.manual_seed(seed)
    return MQARDeltaModel(vocab, d_model=cfg.d_model, d_k=cfg.d_k, d_v=cfg.d_v, conv_k=cfg.conv_k).to(DEV).to(DT)


def make_pool(cfg, cond, n, start=0):
    ex = [make_mb_example(cond, cfg.num_keys, cfg.num_vals, i) for i in range(start, start + n)]
    seg = cfg.seg
    return dict(
        ids=torch.tensor([e["input_ids"] for e in ex], dtype=torch.long),
        lab=torch.tensor([e["labels"] for e in ex], dtype=torch.long),
        apos=torch.tensor([e["answer_positions"] for e in ex], dtype=torch.long),
        wpos=torch.tensor([[p["write_pos"] for p in e["pairs"]] for e in ex], dtype=torch.long),
        dist=torch.tensor([[p["distance"] for p in e["pairs"]] for e in ex], dtype=torch.long),
        wseg=torch.tensor([[p["write_pos"] // seg for p in e["pairs"]] for e in ex], dtype=torch.long),
        ids_hash=hashlib.blake2b(json.dumps([e["example_id"] for e in ex]).encode(), digest_size=8).hexdigest(),
        first_ids=[e["example_id"] for e in ex[:3]], n=n)


def mc_forward(cfg, model, ids, mode, reader, warm_start=True, path="chunked",
               recent_k=None, snap_mask=None, return_gates=False):
    """reader in {single, moving_average, grm}. Caches each segment's final recurrent state; aggregates cached +
    online reads. recent_k: keep only most-recent-k cached states. snap_mask: explicit set/list of cached-
    snapshot indices to KEEP (overrides recent_k) -> ablation. return_gates: capture last-segment grm softmax."""
    seg_size = cfg.seg
    blk = model.blk
    x, q, k, v, g, beta, _ = model.project(ids)
    B, Lx, _ = q.shape
    if reader == "single":
        o, _ = run_recurrence(mode, q, k, v, g, beta, seg_size, None, path)
        return (model.readout(x, o), None) if return_gates else model.readout(x, o)
    segs = [(i, min(i + seg_size, Lx)) for i in range(0, Lx, seg_size)]
    cached_states, cached_pools, outs = [], [], []
    S_prev = None
    gate_info = None
    for si, (a, b) in enumerate(segs):
        qs, ks, vs, gs, bs = q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b]
        S0 = S_prev if (warm_start and S_prev is not None) else None
        online, S_fin = run_recurrence(mode, qs, ks, vs, gs, bs, seg_size, S0, path)
        # choose which cached snapshots are visible to this segment
        if snap_mask is not None:
            idxs = [j for j in range(len(cached_states)) if j in snap_mask]
        elif recent_k is not None:
            idxs = list(range(len(cached_states)))[-recent_k:]
        else:
            idxs = list(range(len(cached_states)))
        avail = [cached_states[j] for j in idxs]
        avail_pools = [cached_pools[j] for j in idxs]
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
            if return_gates and si == len(segs) - 1:
                gate_info = dict(seg_start=a, gates=gate_softmax(gl).detach(), snap_idxs=list(idxs))
            o = agg_grm(cr, online, gl)
        outs.append(blk.out(o))
        cached_states.append(S_fin)
        cached_pools.append(ks.mean(dim=1))
        S_prev = S_fin
    logits = model.head(model.norm(x + torch.cat(outs, dim=1)))
    return (logits, gate_info) if return_gates else logits


def train(cfg, model, pool, steps, mode, reader, warm_start=True, params=None, seed=42, log=None, path="chunked"):
    params = list(model.parameters()) if params is None else params
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gcpu = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    t0 = time.time()
    loss = torch.tensor(0.0)
    for s in range(steps):
        idx = torch.randint(0, pool["n"], (cfg.batch,), generator=gcpu)
        ids = pool["ids"][idx].to(DEV); lab = pool["lab"][idx].to(DEV)
        logits = mc_forward(cfg, model, ids, mode, reader, warm_start=warm_start, path=path)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), lab.view(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if log and (s % 500 == 0 or s == steps - 1):
            log(f"      step {s:4d}/{steps} loss {loss.item():.4f}")
    return dict(steps=steps, wall_s=round(time.time() - t0, 1), final_loss=round(float(loss.item()), 4),
                trained_params=sum(p.numel() for p in params))


@torch.no_grad()
def evaluate(cfg, model, pool, mode, reader, warm_start=True, batch=256, recent_k=None):
    """Plain accuracy over answer positions."""
    model.eval()
    correct = total = 0
    for i in range(0, pool["n"], batch):
        ids = pool["ids"][i:i + batch].to(DEV); lab = pool["lab"][i:i + batch].to(DEV)
        pred = mc_forward(cfg, model, ids, mode, reader, warm_start=warm_start, recent_k=recent_k).argmax(-1)
        mask = lab != -100
        correct += (pred[mask] == lab[mask]).sum().item(); total += mask.sum().item()
    return round(correct / total, 4) if total else 0.0


@torch.no_grad()
def evaluate_paired(cfg, model, pool, mode, reader_mc="grm", recent_k=None, snap_mask=None, batch=256,
                    capture_attr=False):
    """Per-target BASE(single) vs MC correctness + recovery/harm decomposition. Optionally capture last-segment
    grm gate attribution for BASE-wrong/MC-correct targets."""
    model.eval()
    nseg = math.ceil(pool["ids"].shape[1] / cfg.seg)
    base_ok, mc_ok, dists, wsegs = [], [], [], []
    attr = []            # per recovered target: {argmax_snap_seg, write_seg, gate_on_proximal, n_snap}
    for i in range(0, pool["n"], batch):
        ids = pool["ids"][i:i + batch].to(DEV); lab = pool["lab"][i:i + batch]
        apos = pool["apos"][i:i + batch]; wseg = pool["wseg"][i:i + batch]; dist = pool["dist"][i:i + batch]
        base_pred = mc_forward(cfg, model, ids, mode, "single").argmax(-1).cpu()
        _mc = mc_forward(cfg, model, ids, mode, reader_mc, recent_k=recent_k, snap_mask=snap_mask,
                         return_gates=capture_attr)
        mc_logits, ginfo = _mc if capture_attr else (_mc, None)
        mc_pred = mc_logits.argmax(-1).cpu()
        for bb in range(ids.size(0)):
            for j, p in enumerate(apos[bb].tolist()):
                tgt = lab[bb, p].item()
                bok = int(base_pred[bb, p].item() == tgt)
                mok = int(mc_pred[bb, p].item() == tgt)
                base_ok.append(bok); mc_ok.append(mok)
                dists.append(int(dist[bb, j].item())); wsegs.append(int(wseg[bb, j].item()))
                if capture_attr and ginfo is not None and bok == 0 and mok == 1:
                    pos_in_seg = p - ginfo["seg_start"]
                    if 0 <= pos_in_seg < ginfo["gates"].shape[1]:
                        gvec = ginfo["gates"][bb, pos_in_seg]         # [n_snap+1] (snaps..., online)
                        snap_idxs = ginfo["snap_idxs"]
                        if snap_idxs:
                            snap_g = gvec[:len(snap_idxs)]
                            am = int(snap_g.argmax().item())
                            argmax_seg = snap_idxs[am]                # snapshot index == segment index
                            ws = int(wseg[bb, j].item())
                            # proximal snapshot = first snapshot AFTER the write segment
                            prox = min((s for s in snap_idxs if s >= ws), default=snap_idxs[0])
                            prox_col = snap_idxs.index(prox)
                            attr.append(dict(argmax_snap_seg=argmax_seg, write_seg=ws,
                                             gate_on_proximal=float(gvec[prox_col].item()),
                                             gate_online=float(gvec[-1].item()), n_snap=len(snap_idxs)))
    base_ok = np.array(base_ok); mc_ok = np.array(mc_ok)
    n = len(base_ok)
    bw = base_ok == 0; bc = base_ok == 1
    recovery = float(mc_ok[bw].mean()) if bw.any() else None      # BASE-wrong -> MC-correct
    harm = float((1 - mc_ok[bc]).mean()) if bc.any() else None    # BASE-correct -> MC-wrong
    res = dict(n_targets=int(n), base_acc=round(float(base_ok.mean()), 4), mc_acc=round(float(mc_ok.mean()), 4),
               net_delta=round(float(mc_ok.mean() - base_ok.mean()), 4),
               n_base_wrong=int(bw.sum()), n_base_correct=int(bc.sum()),
               RECOVERY_RATE=(round(recovery, 4) if recovery is not None else None),
               HARM_RATE=(round(harm, 4) if harm is not None else None),
               cell_Bok_MCok=int(((base_ok == 1) & (mc_ok == 1)).sum()),
               cell_Bok_MCwrong=int(((base_ok == 1) & (mc_ok == 0)).sum()),
               cell_Bwrong_MCok=int(((base_ok == 0) & (mc_ok == 1)).sum()),
               cell_Bwrong_MCwrong=int(((base_ok == 0) & (mc_ok == 0)).sum()))
    if capture_attr:
        if attr:
            prox_hit = np.mean([a["argmax_snap_seg"] >= a["write_seg"] and
                                a["argmax_snap_seg"] <= a["write_seg"] + 1 for a in attr])
            res["attribution"] = dict(
                n_recovered_examined=len(attr),
                frac_argmax_temporally_plausible=round(float(prox_hit), 4),
                mean_gate_on_proximal=round(float(np.mean([a["gate_on_proximal"] for a in attr])), 4),
                mean_gate_online=round(float(np.mean([a["gate_online"] for a in attr])), 4),
                SNAPSHOT_ATTRIBUTION="DESCRIPTIVE",
                note="argmax snapshot vs target write segment; high gate != causal (see ablation)")
        else:
            res["attribution"] = dict(n_recovered_examined=0, SNAPSHOT_ATTRIBUTION="DESCRIPTIVE",
                                      note="no BASE-wrong/MC-correct targets to attribute")
    return res


def tensor_hashes(model):
    return {n: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for n, p in model.state_dict().items()}


# ==================================================================================================
# 4. Pre-registration writer
# ==================================================================================================
def write_preregistration(cfg: ExtConfig, outdir):
    os.makedirs(outdir, exist_ok=True)
    pre = dict(packet="RNN-05B-EXT", title="Historical-state recoverability outside the ceiling (H3)",
               config=asdict(cfg), select_rule=cfg.select_rule(), grid=cfg.grid(),
               calibration_rule_selfcheck=calibration_rule_selfcheck(cfg))
    json.dump(pre, open(os.path.join(outdir, "preregistration.json"), "w"), indent=2)
    scheck = calibration_rule_selfcheck(cfg)
    g = cfg.grid()
    md = f"""# RNN-05B-EXT — PRE-REGISTRATION (written before any outcome-bearing run)

Direct **H3** test: can a HISTORICAL DN/GDN recurrent-state snapshot recover an association the stable frozen
backbone has LOST from its final state, beyond what the final state alone yields? LA is the mechanistic control.
This file and `preregistration.json` are generated from ONE frozen config object (`ExtConfig`); the executed
constants, the JSON, and the human-readable selection rule are identity-checked
(`CALIBRATION_RULE_IDENTITY = {scheck['CALIBRATION_RULE_IDENTITY']}`).

## Hypotheses
- **H3**: historical DN/GDN states contain recoverable task information no longer sufficiently represented in
  the final recurrent state.
- **NULL**: historical snapshots add nothing recoverable beyond the final state under a stable regime.
- **LA control**: additive/collapsible final state -> should not benefit from redundant historical snapshots.

## Architecture (RNN-05B-qualified family, UNCHANGED)
`MQARDeltaModel` d_model={cfg.d_model}, d_k={cfg.d_k}, d_v={cfg.d_v}, conv_k={cfg.conv_k}; MC/chunk segment
seg={cfg.seg}. No recurrence-equation edits, no deeper readers, no GDN-mechanism edits. Reader = the existing
`w_u` grm connector only.

## Memory-bound challenge (temporal pressure, NOT capacity overload)
- num_pairs = **{cfg.num_pairs}** (far below the RNN-05B capacity cliff ~40 at d_k=64 — no capacity overload).
- num_queries = {cfg.num_queries}; num_keys = {cfg.num_keys}; num_vals = {cfg.num_vals}.
- Writes spread across the EARLY {cfg.write_frac:.0%} of the body; a long retention gap with distractor
  interference follows; queries are at the very end. Recall is hard because old associations decay / are
  interfered with, NOT because the recurrence is globally untrainable.

## Predeclared candidate grid (cheap-first; frozen numeric generator params)
seq_len axis = {list(cfg.seq_lens)} (retention gap); distractor axis =
{[(t, d) for t, d in cfg.distractor_tiers]} (interference). {len(g)} conditions, order:
{", ".join(c['name'] for c in g)}.

## Headroom / calibration rule (single source of truth)
> {cfg.select_rule()}

Required GDN BASE band **[{cfg.band_lo}, {cfg.band_hi}]** (operator-design TESTABILITY window, not a scientific
noise floor); per-seed stability band [{cfg.seed_lo}, {cfg.seed_hi}]. Calibration observes **BASE only** — no MC
or reader result may influence difficulty selection. If no condition qualifies: `H3_TESTABILITY = BLOCKED`
(`BLOCKED_BY_CEILING` if every condition > {cfg.band_hi}; `BLOCKED_BY_UNSTABLE_BASE` if the base collapses),
and STOP with **no nearest-condition fallback**.

## Training identity (predeclared seeds)
BASE backbones trained single-state at steps={cfg.steps_bb}, lr={cfg.lr}, batch={cfg.batch},
pool_train={cfg.pool_train}. Load-bearing: GDN seeds {list(cfg.gdn_seeds)}, DN seeds {list(cfg.dn_seeds)};
control: LA seeds {list(cfg.la_seeds)}. Frozen reader trained {cfg.steps_rd} steps (w_u only; backbone mutation
must be 0). Disjoint CALIBRATION / DEVELOPMENT / FINAL-HOLDOUT example ranges; pinned id hashes.

## Load-bearing comparison (frozen backbone)
A = BASE single final state · B = param-free historical snapshots (moving average) · C = trained `w_u` reader.
Primary quantity = **RECOVERY_RATE** (fraction of BASE-wrong target queries MC flips to correct) with
**HARM_RATE** (BASE-correct flipped to wrong) reported alongside (a net gain can hide large recovery+harm).
Then: snapshot attribution (DESCRIPTIVE), a small snapshot ablation, a fixed-position pure cache-count curve
(K={list(cfg.cache_K)}), the LA falsification control, DN vs GDN reported separately, and a small secondary 2x2.

## Effects
Raw paired deltas across the {len(cfg.gdn_seeds)} training seeds; direction agreement; MARGIN={cfg.margin}
retained only as an OPERATOR_HEURISTIC (not a measured noise floor). No p-values manufactured from n=3.

## Decision
A strong positive H3 requires: `H3_TESTABILITY=QUALIFIED`; GDN positive net holdout delta; direction positive
across all {len(cfg.gdn_seeds)} seeds; recovery exceeds harm by a meaningful margin; ablation supports the
snapshot mechanism; and LA does NOT show the same pattern. If BASE has headroom and MC still adds nothing ->
`H3 = NOT_DETECTED_IN_QUALIFIED_REGIME` (a much stronger negative than RNN-05B). Qwen: no weights used; gate
= `PASS_CANDIDATE` only on a defensible positive mechanism (authorizes DESIGN of a separate Qwen packet only),
else `DEFER`.

## Guardrails
No Qwen weights · no llama.cpp/serving/deploy · no TPTT · no RNN-05C · no FLA install · not pushed. Budget
target < 1 GPU-hr (hard 2). RNN-05B raw evidence immutable.
"""
    open(os.path.join(outdir, "PRE_REGISTRATION.md"), "w", encoding="utf-8").write(md)
    return pre, scheck


# ==================================================================================================
# 5. main run (calibration -> frozen H3 -> analyses -> outcomes)
# ==================================================================================================
def classify(delta, margin):
    return "POSITIVE" if delta >= margin else ("NEGATIVE" if delta <= -margin else "NO_EFFECT")


def run(cfg: ExtConfig, outdir, artifacts=None, smoke=False):
    os.makedirs(outdir, exist_ok=True)
    art = artifacts or outdir
    os.makedirs(art, exist_ok=True)
    logf = open(os.path.join(outdir, "run.log"), "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    if smoke:
        # pipeline smoke: widen the band so a condition is SELECTED and every downstream phase runs (accuracies
        # are meaningless at these step counts — this only validates code paths, not science).
        cfg = ExtConfig(steps_bb=200, steps_rd=150, batch=16, pool_train=256, pool_eval=128,
                        seq_lens=(256,), distractor_tiers=(("low", 0.15),),
                        band_lo=0.0, band_hi=1.0, seed_lo=0.0, seed_hi=1.0,
                        gdn_seeds=(42, 43), dn_seeds=(42,), la_seeds=(42,))
    vocab = mb_vocab(cfg)
    t_start = time.time()

    R = dict(meta=dict(packet="RNN-05B-EXT", device=DEV, torch=torch.__version__, numpy=np.__version__,
                       smoke=smoke, config=asdict(cfg), artifacts_dir=art),
             preregistration=dict(select_rule=cfg.select_rule(),
                                  calibration_rule_selfcheck=calibration_rule_selfcheck(cfg)),
             substrate_sanity={}, calibration=dict(scan=[]), selected=None, frozen={}, recovery={},
             ablation={}, cache_count={}, twobytwo={}, cost={}, memory={}, outcomes={})

    def snap():
        json.dump(R, open(os.path.join(outdir, "rnn05bext_results.json"), "w"), indent=2)

    scheck = calibration_rule_selfcheck(cfg)
    log(f"[CFG] CALIBRATION_RULE_IDENTITY={scheck['CALIBRATION_RULE_IDENTITY']} band={scheck['executed_band']} "
        f"seed_band={scheck['executed_seed_band']}")
    if scheck["CALIBRATION_RULE_IDENTITY"] != "PASS":
        R["outcomes"] = dict(H3_TESTABILITY="ABORTED_CALIBRATION_IDENTITY"); snap(); logf.close(); return

    # ---- substrate sanity (already qualified in RNN-05B; cheap reconfirm on THIS family instance) ----
    log("[P0] substrate sanity: reference parity + full-module checkpoint/restore + request isolation")
    gm = build_model(cfg, vocab, seed=0)
    gcpu = torch.Generator().manual_seed(7)
    gids = torch.randint(0, vocab, (6, 160), generator=gcpu).to(DEV)
    gidsB = torch.randint(0, vocab, (6, 160), generator=torch.Generator().manual_seed(8)).to(DEV)
    for mode in MODES:
        par = reference_parity(gm, gids, mode, cfg.seg)
        life = checkpoint_restore_full_module(gm, gids, mode, 96, cfg.seg)
        iso = request_isolation(gm, gids, gidsB, mode, cfg.seg)
        R["substrate_sanity"][mode] = dict(REFERENCE_PARITY=par["PARITY"], parity_maxabs=par["maxabs"],
                                           FULL_MODULE_LIFECYCLE=life["FULL_MODULE_CHECKPOINT_RESTORE"],
                                           REQUEST_ISOLATION=iso["REQUEST_STATE_ISOLATION"])
        log(f"  {mode}: parity={par['PARITY']} lifecycle={life['FULL_MODULE_CHECKPOINT_RESTORE']} "
            f"iso={iso['REQUEST_STATE_ISOLATION']}")
    del gm
    torch.cuda.empty_cache() if DEV == "cuda" else None
    snap()

    # ---- P1 calibration: BASE-only, cheap-first scan; select first GDN seed-0 base in band ----
    log(f"[P1] calibration (BASE-only) — rule: {cfg.select_rule()}")
    grid = cfg.grid()
    selected = None
    gdn_seed0_backbone = None
    for cond in grid:
        pools = _pools_for(cfg, cond)
        m = build_model(cfg, vocab, seed=cfg.gdn_seeds[0])
        train(cfg, m, pools["train"], cfg.steps_bb, "gdn", "single", seed=cfg.gdn_seeds[0])
        base = evaluate(cfg, m, pools["hold"], "gdn", "single")
        R["calibration"]["scan"].append(dict(condition=cond, gdn_seed0_base=base,
                                              in_band=bool(cfg.band_lo <= base <= cfg.band_hi)))
        log(f"  {cond['name']}: GDN seed{cfg.gdn_seeds[0]} BASE={base} "
            f"in_band={cfg.band_lo <= base <= cfg.band_hi}")
        snap()
        if cfg.band_lo <= base <= cfg.band_hi:
            selected = cond
            gdn_seed0_backbone = m       # reuse as the seed-0 GDN frozen backbone
            break
        del m
        torch.cuda.empty_cache() if DEV == "cuda" else None

    if selected is None:
        bases = [s["gdn_seed0_base"] for s in R["calibration"]["scan"]]
        reason = ("BLOCKED_BY_CEILING" if all(b > cfg.band_hi for b in bases) else
                  "BLOCKED_BY_UNSTABLE_BASE" if all(b < cfg.band_lo for b in bases) else "BLOCKED")
        log(f"[STOP] no condition satisfied the headroom rule -> H3_TESTABILITY={reason} (no fallback)")
        R["outcomes"] = dict(H3_TESTABILITY=reason, scan_bases=bases,
                             QWEN_GDN_TRANSPLANT_GATE="DEFER",
                             GDN_HISTORICAL_RECOVERY="NOT_APPLICABLE_BLOCKED")
        snap(); json.dump(R["outcomes"], open(os.path.join(outdir, "rnn05bext_outcomes.json"), "w"), indent=2)
        logf.close(); return

    log(f"[P1] SELECTED condition: {selected['name']} {selected}")
    R["selected"] = selected
    pools = _pools_for(cfg, selected)
    R["benchmarks"] = dict(condition=selected,
                           train_ids_hash=pools["train"]["ids_hash"], dev_ids_hash=pools["dev"]["ids_hash"],
                           holdout_ids_hash=pools["hold"]["ids_hash"],
                           holdout_first_ids=pools["hold"]["first_ids"])
    snap()

    # ---- P2 frozen H3: train BASE backbones, freeze, A/B/C + recovery/harm ----
    # AUDIT NOTE (RNN-05B-EXT audit reconciliation §1, AUDIT_RECONCILIATION.md): this P2 block runs the
    # outcome-bearing MC/reader work BEFORE the 3-seed stability gate below (P1c, ~line 707) that sets
    # H3_TESTABILITY -> PROTOCOL_GATE_ORDERING=FAILED; the recovery/harm numbers are therefore
    # POST_STABILITY_GATE_MC_RESULTS=EXPLORATORY_NON_LOAD_BEARING. Behavior is left UNCHANGED (historical
    # runner preserved). EXT2 HARD REQUIREMENT: complete + persist all preregistered-seed BASE qualification
    # BEFORE computing any MC/reader outcome (gate strictly before outcome).
    seeds_by_mode = dict(gdn=list(cfg.gdn_seeds), dn=list(cfg.dn_seeds), la=list(cfg.la_seeds))
    for mode in MODES:
        R["frozen"][mode] = []
        for si, seed in enumerate(seeds_by_mode[mode]):
            if mode == "gdn" and seed == cfg.gdn_seeds[0] and gdn_seed0_backbone is not None:
                m = gdn_seed0_backbone       # reuse calibration model (same seed, same condition, same steps)
                bb_stat = dict(reused_from_calibration=True)
                log(f"[P2] {mode} seed={seed}: reuse calibration BASE backbone")
            else:
                m = build_model(cfg, vocab, seed=seed)
                bb_stat = train(cfg, m, pools["train"], cfg.steps_bb, mode, "single", seed=seed, log=log)
                log(f"[P2] {mode} seed={seed}: trained BASE backbone loss={bb_stat['final_loss']}")
            # freeze backbone
            for p in m.parameters():
                p.requires_grad_(False)
            h_before = tensor_hashes(m)
            base_h = evaluate(cfg, m, pools["hold"], mode, "single")
            base_d = evaluate(cfg, m, pools["dev"], mode, "single")
            pf_h = evaluate(cfg, m, pools["hold"], mode, "moving_average")     # B: param-free MC
            # C: train ONLY w_u reader on frozen backbone
            m.blk.w_u.requires_grad_(True)
            rparams = [p for p in m.parameters() if p.requires_grad]
            rd_stat = train(cfg, m, pools["train"], cfg.steps_rd, mode, "grm", params=rparams, seed=seed, log=log)
            changed = [n for n in h_before if n != "blk.w_u.weight" and h_before[n] != tensor_hashes(m)[n]]
            rd_h = evaluate(cfg, m, pools["hold"], mode, "grm")
            rd_d = evaluate(cfg, m, pools["dev"], mode, "grm")
            # recovery/harm on holdout (trained reader vs base), with attribution capture
            rec = evaluate_paired(cfg, m, pools["hold"], mode, "grm", capture_attr=True)
            # durable save (backbone + trained reader) + SHA-256
            ck = os.path.join(art, f"rnn05bext_{mode}_seed{seed}_frozen_reader.pt")
            torch.save({"state_dict": m.state_dict(), "mode": mode, "seed": seed, "vocab": vocab,
                        "condition": selected, "config": asdict(cfg), "reader_tensor": "blk.w_u.weight"}, ck)
            sha = hashlib.sha256(open(ck, "rb").read()).hexdigest()
            reader_sha = hashlib.sha256(m.blk.w_u.weight.detach().cpu().numpy().tobytes()).hexdigest()
            cell = dict(seed=seed, base_holdout=base_h, base_dev=base_d, paramfree_MC_holdout=pf_h,
                        reader_holdout=rd_h, reader_dev=rd_d,
                        BACKBONE_WEIGHT_MUTATION=len(changed), changed_tensors=changed,
                        FROZEN_BACKBONE_VALIDITY="PASS" if len(changed) == 0 else "FAIL",
                        delta_paramfree=round(pf_h - base_h, 4), delta_reader=round(rd_h - base_h, 4),
                        recovery=rec, checkpoint_path=ck, checkpoint_sha256=sha,
                        reader_weight_sha256=reader_sha, backbone_stat=bb_stat, reader_stat=rd_stat)
            R["frozen"][mode].append(cell)
            log(f"  {mode} seed={seed}: base={base_h} paramfreeMC={pf_h} reader={rd_h} "
                f"RECOVERY={rec['RECOVERY_RATE']} HARM={rec['HARM_RATE']} "
                f"mutation={len(changed)} sha={sha[:12]}")
            snap()
            if not (mode == "gdn" and seed == cfg.gdn_seeds[0]):
                # keep gdn seed0 model reference for cache-count/cost reuse; free others
                if not (mode in ("gdn", "dn", "la") and seed == seeds_by_mode[mode][0]):
                    del m
                    torch.cuda.empty_cache() if DEV == "cuda" else None
        # keep the first-seed frozen model of each mode in memory for P4/P6 (reload if needed)

    # seed-band qualification on GDN (headroom rule confirmation)
    gdn_bases = [c["base_holdout"] for c in R["frozen"]["gdn"]]
    mean_gdn = float(np.mean(gdn_bases))
    band_ok = (cfg.band_lo <= mean_gdn <= cfg.band_hi) and all(cfg.seed_lo <= b <= cfg.seed_hi for b in gdn_bases)
    R["calibration"]["confirm"] = dict(gdn_seed_bases=gdn_bases, mean=round(mean_gdn, 4),
                                       seed_band_ok=band_ok,
                                       H3_TESTABILITY="QUALIFIED" if band_ok else "BLOCKED_BY_UNSTABLE_BASE")
    log(f"[P1c] GDN seed bases={gdn_bases} mean={mean_gdn:.4f} band_ok={band_ok}")
    snap()

    # ---- P3 snapshot ablation (small, global; early-write design -> proximal snapshot = early index) ----
    # AUDIT NOTE (§3): this is a GLOBAL early-snapshot ablation (proximal_idx=0, irrelevant_idx=n_snap-1), and
    # the random control drew random_idx=0 (== proximal) in the historical run -> RANDOM_ABLATION_CONTROL=
    # INVALID_DUPLICATE_OF_EARLY, TARGET_PROXIMAL_SNAPSHOT_CAUSALITY=NOT_QUALIFIED, HISTORICAL_SNAPSHOT_CAUSAL_
    # SIGNAL=INCONCLUSIVE/NOT_QUALIFIED. EXT2: deterministic random control excluding proximal+irrelevant, and
    # a per-target/write-region-aware proximal ablation.
    log("[P3] snapshot ablation on GDN/DN (recovery vs dropping proximal/irrelevant/random snapshot)")
    for mode in ("gdn", "dn"):
        m = _reload_frozen(cfg, R["frozen"][mode][0]["checkpoint_path"], vocab)
        nseg = math.ceil(pools["hold"]["ids"].shape[1] / cfg.seg)
        n_snap = nseg - 1                                   # snapshots available to the last segment
        full = set(range(n_snap))
        proximal = 0                                        # earliest snapshot (writes live in the early segs)
        irrelevant = n_snap - 1                             # a late snapshot far from the writes
        rng = np.random.default_rng(123)
        rnd = int(rng.integers(0, n_snap))
        variants = dict(
            full=evaluate_paired(cfg, m, pools["hold"], mode, "grm", snap_mask=full),
            drop_proximal=evaluate_paired(cfg, m, pools["hold"], mode, "grm", snap_mask=full - {proximal}),
            drop_irrelevant=evaluate_paired(cfg, m, pools["hold"], mode, "grm", snap_mask=full - {irrelevant}),
            drop_random=evaluate_paired(cfg, m, pools["hold"], mode, "grm", snap_mask=full - {rnd}))
        rec_full = variants["full"]["RECOVERY_RATE"] or 0.0
        rec_prox = variants["drop_proximal"]["RECOVERY_RATE"] or 0.0
        rec_irr = variants["drop_irrelevant"]["RECOVERY_RATE"] or 0.0
        drop_prox = rec_full - rec_prox
        drop_irr = rec_full - rec_irr
        signal = ("SUPPORTED" if (drop_prox >= cfg.margin and drop_prox > drop_irr + cfg.margin) else
                  ("NOT_DETECTED" if drop_prox <= 0 else "INCONCLUSIVE"))
        R["ablation"][mode] = dict(n_snap=n_snap, proximal_idx=proximal, irrelevant_idx=irrelevant,
                                   random_idx=rnd, recovery_full=rec_full, recovery_drop_proximal=rec_prox,
                                   recovery_drop_irrelevant=rec_irr,
                                   recovery_drop_random=variants["drop_random"]["RECOVERY_RATE"],
                                   drop_from_proximal=round(drop_prox, 4), drop_from_irrelevant=round(drop_irr, 4),
                                   HISTORICAL_SNAPSHOT_CAUSAL_SIGNAL=signal)
        log(f"  {mode}: rec_full={rec_full} drop_proximal->{rec_prox} drop_irrelevant->{rec_irr} signal={signal}")
        del m
        torch.cuda.empty_cache() if DEV == "cuda" else None
        snap()

    # ---- P4 pure cache-count curve (fixed weights/positions; vary retained K) ----
    log("[P4] pure cache-count curve (fixed backbone+reader; vary retained-K)")
    for mode in MODES:
        m = _reload_frozen(cfg, R["frozen"][mode][0]["checkpoint_path"], vocab)
        h0 = tensor_hashes(m)
        rows = []
        for K in cfg.cache_K:
            rec = evaluate_paired(cfg, m, pools["hold"], mode, "grm", recent_k=K)
            rows.append(dict(retained_K=K, holdout_acc=rec["mc_acc"], recovery=rec["RECOVERY_RATE"],
                             harm=rec["HARM_RATE"], cache_bytes=cfg.d_k * cfg.d_v * 4 * K))
            log(f"  {mode} K={K} acc={rec['mc_acc']} recovery={rec['RECOVERY_RATE']} harm={rec['HARM_RATE']}")
        unchanged = [n for n in h0 if h0[n] != tensor_hashes(m)[n]]
        R["cache_count"][mode] = dict(seg=cfg.seg, n_checkpoints=math.ceil(pools["hold"]["ids"].shape[1] / cfg.seg),
                                      selection_rule="most_recent_K", rows=rows,
                                      PURE_CACHE_COUNT_CURVE="QUALIFIED" if not unchanged else "NOT_QUALIFIED")
        del m
        torch.cuda.empty_cache() if DEV == "cuda" else None
        snap()

    # ---- P5 secondary 2x2 (small; gated on budget) ----
    elapsed_min = (time.time() - t_start) / 60
    if elapsed_min < 90 and R["calibration"]["confirm"]["H3_TESTABILITY"] == "QUALIFIED":
        log(f"[P5] secondary 2x2 (elapsed {elapsed_min:.1f} min < 90) — GDN {cfg.gdn_seeds}, DN/LA 1 seed")
        for mode in MODES:
            seeds = list(cfg.gdn_seeds) if mode == "gdn" else [seeds_by_mode[mode][0]]
            per = []
            for seed in seeds:
                m_s = _reload_frozen(cfg, _find_ck(R, mode, seed), vocab, reader=False)  # train-single backbone
                A = evaluate(cfg, m_s, pools["hold"], mode, "single")
                B = evaluate(cfg, m_s, pools["hold"], mode, "moving_average")
                m_mc = build_model(cfg, vocab, seed=seed)
                train(cfg, m_mc, pools["train"], cfg.steps_bb, mode, "grm", seed=seed)
                C = evaluate(cfg, m_mc, pools["hold"], mode, "single")
                D = evaluate(cfg, m_mc, pools["hold"], mode, "grm")
                per.append(dict(seed=seed, A=A, B=B, C=C, D=D, interaction=round((D - C) - (B - A), 4),
                                joint_DminusA=round(D - A, 4)))
                log(f"  2x2 {mode} seed={seed}: A={A} B={B} C={C} D={D} interaction={per[-1]['interaction']}")
                del m_s, m_mc
                torch.cuda.empty_cache() if DEV == "cuda" else None
                snap()
            R["twobytwo"][mode] = dict(seeds=seeds, per_seed=per,
                                       mean_interaction=round(float(np.mean([p["interaction"] for p in per])), 4),
                                       mean_joint=round(float(np.mean([p["joint_DminusA"] for p in per])), 4))
    else:
        log(f"[P5] secondary 2x2 SKIPPED (elapsed {elapsed_min:.1f} min / testability "
            f"{R['calibration']['confirm']['H3_TESTABILITY']})")
        R["twobytwo"] = dict(skipped=True, reason=f"elapsed={elapsed_min:.1f}min")

    # ---- P6 cost + memory ----
    log("[P6] cost + memory accounting")
    for mode in MODES:
        m = _reload_frozen(cfg, R["frozen"][mode][0]["checkpoint_path"], vocab)
        R["cost"][mode] = cost_breakdown(cfg, m, pools["hold"], mode)
        R["memory"][mode] = memory_breakdown(cfg, m, pools["hold"]["ids"].shape[1])
        log(f"  {mode}: total_ms={R['cost'][mode]['total_ms']} "
            f"state_bytes={R['memory'][mode]['total_live_recurrent_state_bytes']}")
        del m
        torch.cuda.empty_cache() if DEV == "cuda" else None
        snap()

    # ---- outcomes ----
    R["outcomes"] = build_outcomes(cfg, R)
    R["meta"]["wall_min"] = round((time.time() - t_start) / 60, 1)
    json.dump(R["outcomes"], open(os.path.join(outdir, "rnn05bext_outcomes.json"), "w"), indent=2)
    write_csv(cfg, R, os.path.join(outdir, "rnn05bext_summary.csv"))
    snap()
    log("[DONE] outcomes:\n" + json.dumps(R["outcomes"], indent=2))
    logf.close()


def _pools_for(cfg, cond):
    train_p = make_pool(cfg, cond, cfg.pool_train, start=0)
    dev_p = make_pool(cfg, cond, cfg.pool_eval, start=cfg.pool_train)                 # DEVELOPMENT (disjoint)
    hold_p = make_pool(cfg, cond, cfg.pool_eval, start=cfg.pool_train + cfg.pool_eval)  # FINAL HOLDOUT (disjoint)
    return dict(train=train_p, dev=dev_p, hold=hold_p)


def _reload_frozen(cfg, path, vocab, reader=True):
    ck = torch.load(path, map_location=DEV)
    m = build_model(cfg, vocab, seed=0)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def _find_ck(R, mode, seed):
    for c in R["frozen"][mode]:
        if c["seed"] == seed:
            return c["checkpoint_path"]
    return R["frozen"][mode][0]["checkpoint_path"]


@torch.no_grad()
def cost_breakdown(cfg, model, pool, mode, reps=20, batch=64):
    blk = model.blk
    ids = pool["ids"][:batch].to(DEV)
    x, q, k, v, g, beta, _ = model.project(ids)
    B, Lx, _ = q.shape
    seg = cfg.seg
    segs = [(i, min(i + seg, Lx)) for i in range(0, Lx, seg)]

    def sync():
        torch.cuda.synchronize() if DEV == "cuda" else None

    def timeit(fn):
        sync(); t0 = time.time()
        for _ in range(reps):
            fn()
        sync(); return (time.time() - t0) / reps * 1000

    def do_recurrent():
        S = None
        for (a, b) in segs:
            _, S = run_recurrence(mode, q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b], seg, S, "chunked")

    states = []
    S = None
    for (a, b) in segs:
        _, S = run_recurrence(mode, q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b], seg, S, "chunked")
        states.append(S)

    def do_read():
        for i, (a, b) in enumerate(segs):
            if i:
                read_states(states[:i], q[:, a:b])

    return dict(mode=mode, n_segments=len(segs), batch=batch, reps=reps,
                projection_conv_ms=round(timeit(lambda: model.project(ids)), 3),
                recurrent_update_ms=round(timeit(do_recurrent), 3),
                state_read_ms=round(timeit(do_read), 3),
                total_ms=round(timeit(lambda: mc_forward(cfg, model, ids, mode, "grm")), 3))


def memory_breakdown(cfg, model, L):
    matrix = cfg.d_k * cfg.d_v * 4
    conv = model.blk.conv_dim * (cfg.conv_k - 1) * 4
    n_ckpt = math.ceil(L / cfg.seg)
    return dict(mode="all", matrix_state_bytes=matrix, conv_state_bytes=conv,
                total_live_recurrent_state_bytes=matrix + conv,
                historical_checkpoint_bytes_per_ckpt=matrix,
                historical_checkpoint_bytes_at_full=matrix * (n_ckpt - 1), n_checkpoints=n_ckpt,
                reader_param_bytes=model.blk.w_u.weight.numel() * 4)


def _recovery_label(cfg, net, recovery, harm, all_pos):
    """Per-substrate H*_HISTORICAL_RECOVERY from net delta + recovery/harm + seed-direction agreement."""
    meaningful = (recovery is not None and harm is not None and recovery > harm + cfg.recovery_margin)
    if net >= cfg.margin and meaningful and all_pos:
        return "POSITIVE"
    if net <= -cfg.margin:
        return "NEGATIVE"
    if net > 0 and meaningful:
        return "WEAK_DIRECTIONAL"
    if abs(net) < cfg.margin:
        return "NO_EFFECT"
    return "INCONCLUSIVE"


def build_outcomes(cfg, R):
    out = dict(margin_OPERATOR_HEURISTIC=cfg.margin, selected_condition=R["selected"],
               H3_TESTABILITY=R["calibration"]["confirm"]["H3_TESTABILITY"], per_substrate={})
    for mode in MODES:
        cells = R["frozen"][mode]
        bases = [c["base_holdout"] for c in cells]
        readers = [c["reader_holdout"] for c in cells]
        nets = [c["delta_reader"] for c in cells]
        recs = [c["recovery"]["RECOVERY_RATE"] for c in cells if c["recovery"]["RECOVERY_RATE"] is not None]
        harms = [c["recovery"]["HARM_RATE"] for c in cells if c["recovery"]["HARM_RATE"] is not None]
        net = float(np.mean(nets))
        recovery = float(np.mean(recs)) if recs else None
        harm = float(np.mean(harms)) if harms else None
        all_pos = all(n > 0 for n in nets)
        label = _recovery_label(cfg, net, recovery, harm, all_pos)
        out["per_substrate"][mode] = dict(
            seeds=[c["seed"] for c in cells], base_holdout=bases, reader_holdout=readers,
            mean_base=round(float(np.mean(bases)), 4), mean_reader=round(float(np.mean(readers)), 4),
            mean_net_delta=round(net, 4), per_seed_net=nets, direction_all_positive=all_pos,
            paramfree_delta=round(float(np.mean([c["delta_paramfree"] for c in cells])), 4),
            RECOVERY_RATE=(round(recovery, 4) if recovery is not None else None),
            HARM_RATE=(round(harm, 4) if harm is not None else None),
            FROZEN_BACKBONE_VALIDITY=("PASS" if all(c["FROZEN_BACKBONE_VALIDITY"] == "PASS" for c in cells)
                                      else "FAIL"),
            HISTORICAL_RECOVERY=label)
    out["GDN_HISTORICAL_RECOVERY"] = out["per_substrate"]["gdn"]["HISTORICAL_RECOVERY"]
    out["DN_HISTORICAL_RECOVERY"] = out["per_substrate"]["dn"]["HISTORICAL_RECOVERY"]
    out["LA_HISTORICAL_RECOVERY"] = out["per_substrate"]["la"]["HISTORICAL_RECOVERY"]
    out["GDN_RECOVERY_RATE"] = out["per_substrate"]["gdn"]["RECOVERY_RATE"]
    out["GDN_HARM_RATE"] = out["per_substrate"]["gdn"]["HARM_RATE"]
    out["DN_RECOVERY_RATE"] = out["per_substrate"]["dn"]["RECOVERY_RATE"]
    out["DN_HARM_RATE"] = out["per_substrate"]["dn"]["HARM_RATE"]
    out["HISTORICAL_SNAPSHOT_CAUSAL_SIGNAL"] = R.get("ablation", {}).get("gdn", {}).get(
        "HISTORICAL_SNAPSHOT_CAUSAL_SIGNAL", "NOT_TESTED")
    # LA falsification: does the additive control show the same positive pattern?
    la_pos = out["LA_HISTORICAL_RECOVERY"] in ("POSITIVE", "WEAK_DIRECTIONAL")
    out["LA_FALSIFICATION"] = ("CONTROL_HELD" if not la_pos else "CONTROL_ALSO_POSITIVE_WEAKENS_SPECIFICITY")
    # decision policy (§20) -> Qwen gate (§22)
    gdn = out["per_substrate"]["gdn"]
    strong = (out["H3_TESTABILITY"] == "QUALIFIED" and gdn["HISTORICAL_RECOVERY"] == "POSITIVE"
              and gdn["direction_all_positive"]
              and gdn["RECOVERY_RATE"] is not None and gdn["HARM_RATE"] is not None
              and gdn["RECOVERY_RATE"] > gdn["HARM_RATE"] + cfg.recovery_margin
              and out["HISTORICAL_SNAPSHOT_CAUSAL_SIGNAL"] == "SUPPORTED"
              and out["LA_FALSIFICATION"] == "CONTROL_HELD")
    if out["H3_TESTABILITY"] != "QUALIFIED":
        gate = "DEFER"
        h3 = out["H3_TESTABILITY"]
    elif strong:
        gate = "PASS_CANDIDATE"
        h3 = "POSITIVE"
    elif gdn["HISTORICAL_RECOVERY"] in ("NO_EFFECT", "NEGATIVE"):
        gate = "DEFER"
        h3 = "NOT_DETECTED_IN_QUALIFIED_REGIME"
    else:
        gate = "DEFER"
        h3 = "WEAK_DIRECTIONAL"
    out["H3_RESULT"] = h3
    out["QWEN_GDN_TRANSPLANT_GATE"] = gate
    out["gate_note"] = ("PASS_CANDIDATE authorizes only DESIGN of a separate real-Qwen qualification packet; it "
                        "does not authorize automatic transplantation or training." if gate == "PASS_CANDIDATE"
                        else "H3 not a defensible positive mechanism in the qualified regime -> defer Qwen work.")
    return out


def write_csv(cfg, R, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["substrate", "seed", "base_holdout", "paramfree_MC", "reader_holdout", "net_delta",
                    "recovery_rate", "harm_rate", "backbone_mutation"])
        for mode in MODES:
            for c in R["frozen"][mode]:
                w.writerow([mode, c["seed"], c["base_holdout"], c["paramfree_MC_holdout"], c["reader_holdout"],
                            c["delta_reader"], c["recovery"]["RECOVERY_RATE"], c["recovery"]["HARM_RATE"],
                            c["BACKBONE_WEIGHT_MUTATION"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preregister", default=None, metavar="DIR", help="write PRE_REGISTRATION.md + JSON (no GPU)")
    ap.add_argument("--selftest", default=None, metavar="JSON", help="memory-bound generator selftest (no GPU)")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--grid-json", default=None, metavar="FILE",
                    help="pre-committed amendment: explicit cheap-first condition list replacing the grid")
    a = ap.parse_args()
    cfg = ExtConfig(grid_override_json=open(a.grid_json).read()) if a.grid_json else ExtConfig()
    if a.preregister:
        pre, sc = write_preregistration(cfg, a.preregister)
        print(json.dumps(dict(wrote=a.preregister, CALIBRATION_RULE_IDENTITY=sc["CALIBRATION_RULE_IDENTITY"],
                              n_conditions=len(cfg.grid())), indent=2))
    elif a.selftest is not None:
        mb_selftest(cfg, a.selftest)
    elif a.run:
        if not a.outdir:
            ap.error("--run requires --outdir")
        run(cfg, a.outdir, a.artifacts, a.smoke)
    else:
        ap.error("one of --preregister / --selftest / --run is required")
