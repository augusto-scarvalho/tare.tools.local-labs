#!/usr/bin/env python
"""
RNN-05B-EXT2: Fixed-Backbone Retention Dose-Response (FINAL synthetic H3 test).

Scientific question (packet §1):
  For ONE already-trained, stable and FROZEN GDN/DN representation, does progressively increasing
  inference-time retention pressure produce GRADED loss of old associations, and can HISTORICAL recurrent-state
  snapshots recover associations the FINAL recurrent state no longer retrieves?

This must ISOLATE inference-time retention/forgetting from TRAINING INSTABILITY. Unlike RNN-05B-EXT (which
trained a NEW backbone per challenge condition -> a seed cliff = TRAIN_PER_CONDITION_STABILITY FAILED), EXT2
trains each backbone ONCE under one stable recipe, freezes+hashes the exact weights, and varies pressure ONLY
at inference. The SAME weights face every stress point.

Design (see PRE_REGISTRATION.md, written from ONE frozen Ext2Config BEFORE any outcome-bearing run):
  * Reuse the RNN-05B-qualified LA/DN/GDN toy family UNCHANGED (rnn_delta_substrate). No recurrence edits.
    The RNN-05B backbones were NOT saved to disk AND were trained on a different (capacity) MQAR distribution,
    so exact-artifact reuse is INVALID+IMPOSSIBLE -> §2 fallback: train each preregistered seed ONCE, save,
    hash, freeze; identical weights for every stress point. (BACKBONE_REUSE=RETRAIN_ONCE, justified in packet.)
  * NESTED MONOTONIC stress axis: distractor density in the post-write retention gap, at FIXED seq_len. Each
    higher dose = the SAME base example with MORE gap slots converted to distractor keys (superset). Writes,
    queries, target values and all positions are IDENTICAL across doses -> per-target dose-response + fixed
    snapshot positions (§8 identity holds by construction). Pair count fixed FAR below the capacity cliff.
  * ONE stable recipe per seed: BASE trained single-state on a MIXTURE over the whole dose ladder (domain
    randomization) so one representation is competent across the range; then freeze. Pressure varies at
    inference only.
  * HARD control-flow invariant (§6): frozen backbones -> ALL preregistered BASE stress points for ALL seeds ->
    persist BASE_QUALIFICATION.json -> verify hashes + challengeGridSha256 -> graded-region gate -> ONLY THEN MC.
    The MC entrypoint LOADS+VERIFIES the qualification artifact; absent/mismatched/unqualified => STOP.
  * challengeGridSha256 (§4): one canonical digest recorded IDENTICALLY in PRE_REGISTRATION, machine config,
    BASE qualification, run metadata, final results; self-checked; mismatch = STOP.
  * Graded-region gate (§7): qualify H3 only if the SAME frozen backbones show an OVERLAPPING graded retention
    region (not one cell in a band). Else FIXED_BACKBONE_GRADED_REGION=BLOCKED -> H3_TESTABILITY=
    BLOCKED_FIXED_BACKBONE, STOP, no MC, no EXT3.
  * Primary paired experiment (§10): A base final state / B param-free snapshot aggregation / C small trained
    w_u reader (backbone mutation MUST be 0), identical examples, reader saved durably+SHA-256. LA is the
    mechanistic control.
  * Metrics (§11): retention curve acc vs dose; AURC_RETENTION, D50, D80_D20_WIDTH, DELTA_AURC, DELTA_D50
    (preregistered LINEAR interpolation; not changed after seeing MC). Recovery/harm with denominators (§12).
  * Target-aware ablation (§13): per-target proximal snapshot; FULL / DROP_TARGET_PROXIMAL / DROP_IRRELEVANT /
    DROP_RANDOM (deterministic, EXCLUDES proximal+irrelevant, asserted in code) / SHAM.
  * Path-activation counters (§9); snapshot identity (§8); hierarchical stats (§15); SESOI (§16); efficiency
    Pareto (§18) with prewarm/warm methodology (§19); eager scan = correctness reference (§20).

Touches NOTHING shared/immutable: imports the qualified substrate/generator building blocks read-only. No Qwen,
no llama.cpp/serving/deploy, no TPTT, no RNN-05C, no FLA, no new kernels, no push.

Usage:
  python rnn_05b_ext2.py --preregister <dir>     # PRE_REGISTRATION.md + machine config + selftests (no GPU)
  python rnn_05b_ext2.py --selftest <json>       # nested-generator self-qualification (no GPU)
  python rnn_05b_ext2.py --run --outdir <dir> --artifacts <dir> [--smoke]
"""
import argparse, csv, hashlib, json, math, os, subprocess, sys, time
from dataclasses import dataclass, asdict, field
import numpy as np
import torch
import torch.nn.functional as F

from rnn_mc_bench import PAD, BOS, QSEP, FILL, KEY_LO, _example_id
from rnn_mc_substrate import read_states, agg_moving_average, agg_grm, gate_softmax
from rnn_delta_substrate import (MQARDeltaModel, run_recurrence, reference_parity,
                                 checkpoint_restore_full_module, request_isolation, MODES)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float32


def _git_head():
    try:
        return subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ==================================================================================================
# 1. Single-source config object (PRE_REGISTRATION == machine config == executed constants)
# ==================================================================================================
@dataclass(frozen=True)
class Ext2Config:
    # --- architecture: RNN-05B-qualified family, UNCHANGED (rnn_delta_substrate.MQARDeltaModel) ---
    d_model: int = 128
    d_k: int = 64
    d_v: int = 64
    conv_k: int = 4
    seg: int = 64                          # MC segment size AND recurrence chunk size
    # --- task: memory-bound MQAR, FIXED seq_len (snapshot positions constant across doses) ---
    seq_len: int = 512
    num_pairs: int = 12                    # FAR below RNN-05B capacity cliff (~40 @ d_k=64): NOT capacity overload
    num_queries: int = 8
    num_keys: int = 128
    num_vals: int = 64
    write_frac: float = 0.25               # writes spread across the EARLY quarter of the body
    # --- NESTED MONOTONIC stress axis: distractor density in the POST-WRITE retention gap ---
    dose_ladder: tuple = (0.0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64)
    stress_axis: str = "postwrite_gap_distractor_density_nested"
    # --- graded-region gate thresholds (§7): a COMMON overlapping graded region, not one cell ---
    grade_hi: float = 0.75                 # backbone must be competent at some LOW dose (max BASE >= hi)
    grade_lo: float = 0.45                 # pressure must genuinely degrade it (min BASE <= lo)
    mid_lo: float = 0.40                   # mid transition band
    mid_hi: float = 0.80
    min_mid_doses: int = 2                 # >=2 doses in the mid band per seed (resolved transition, not a cliff)
    # --- training: ONE stable recipe (mixture over the dose ladder) ---
    steps_bb: int = 2500                   # BASE backbone steps
    steps_rd: int = 1800                   # frozen reader steps
    lr: float = 3e-3
    batch: int = 96
    pool_train: int = 4096
    pool_eval: int = 512
    # --- seeds (predeclared; ALL count; no seed screening) ---
    gdn_seeds: tuple = (42, 43, 44)        # load-bearing (Qwen target substrate)
    dn_seeds: tuple = (42, 43, 44)         # load-bearing (reported alongside)
    la_seeds: tuple = (42,)                # mechanistic falsification control
    # --- analysis ---
    margin: float = 0.03                   # OPERATOR_HEURISTIC direction band (NOT the SESOI, NOT a noise floor)
    sesoi_delta_aurc: float = 0.05         # PRIMARY SESOI on DELTA_AURC (justified in PRE_REGISTRATION)
    recovery_margin: float = 0.02          # recovery must exceed harm by this to be "meaningful"
    cache_K: tuple = (1, 2, 4, 8)
    boot_iters: int = 2000                 # hierarchical bootstrap resamples (CPU)
    # amendment hook (unused unless a pre-committed grid override is supplied)
    dose_override_json: str = ""

    def doses(self):
        if self.dose_override_json:
            return tuple(json.loads(self.dose_override_json))
        return tuple(self.dose_ladder)

    # ---- the challenge grid, as a CANONICAL machine object (§4) ----
    def challenge_grid(self):
        """Everything that defines the stress challenge. The SHA-256 of its canonical JSON is challengeGridSha256
        and must appear identically in PRE_REGISTRATION, machine config, BASE qualification, run meta, results."""
        return dict(
            packet="RNN-05B-EXT2",
            architecture=dict(d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, conv_k=self.conv_k, seg=self.seg),
            task=dict(seq_len=self.seq_len, num_pairs=self.num_pairs, num_queries=self.num_queries,
                      num_keys=self.num_keys, num_vals=self.num_vals, write_frac=self.write_frac),
            stress_axis=self.stress_axis,
            dose_ladder=list(self.doses()),
            interpolation="linear_on_ladder",
            example_ranges=dict(train=[0, self.pool_train],
                                dev=[self.pool_train, self.pool_train + self.pool_eval],
                                holdout=[self.pool_train + self.pool_eval, self.pool_train + 2 * self.pool_eval]),
            seeds=dict(gdn=list(self.gdn_seeds), dn=list(self.dn_seeds), la=list(self.la_seeds)),
            training=dict(recipe="single_state_mixture_over_dose_ladder", steps_bb=self.steps_bb, lr=self.lr,
                          batch=self.batch, pool_train=self.pool_train),
            reader=dict(recipe="frozen_backbone_w_u_grm", steps_rd=self.steps_rd),
        )

    def challenge_grid_sha256(self):
        canon = json.dumps(self.challenge_grid(), sort_keys=True, separators=(",", ":"))
        return _sha256_bytes(canon.encode())

    def config_sha256(self):
        return _sha256_bytes(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode())

    def select_rule(self):
        return (f"Train each seed ONCE (single-state, mixture over dose ladder {list(self.doses())}); FREEZE + "
                f"SHA-256; run ALL doses for ALL seeds on the FROZEN weights as BASE qualification; QUALIFY the "
                f"FIXED_BACKBONE_GRADED_REGION iff every GDN seed has max BASE>={self.grade_hi}, min BASE<="
                f"{self.grade_lo}, and >={self.min_mid_doses} doses with BASE in ({self.mid_lo},{self.mid_hi}), "
                f"AND the mid-band doses OVERLAP across all GDN seeds; else H3_TESTABILITY=BLOCKED_FIXED_BACKBONE "
                f"and STOP with NO MC and NO EXT3.")


def grid_selfcheck(cfg: Ext2Config):
    """challengeGridSha256 must be reproducible from a fresh subprocess importing this module (process-stable).
    The subprocess reconstructs the EXACT config (passed as JSON) so this holds for any config, incl. smoke."""
    digest = cfg.challenge_grid_sha256()
    code = ("import json,sys;sys.path.insert(0,r'%s');from rnn_05b_ext2 import Ext2Config;"
            "c=Ext2Config(**json.loads(sys.argv[1]));print(c.challenge_grid_sha256())"
            % os.path.dirname(os.path.abspath(__file__)))
    a = subprocess.run([sys.executable, "-c", code, json.dumps(asdict(cfg))], capture_output=True, text=True)
    proc = a.stdout.strip()
    ok = (a.returncode == 0 and proc == digest)
    return dict(CHALLENGE_GRID_SHA256=digest, subprocess_digest=proc,
                GRID_IDENTITY_SELFCHECK="PASS" if ok else "FAIL")


# ==================================================================================================
# 2. Nested monotonic memory-bound MQAR generator
#    early writes (write_frac of body) -> long retention gap -> late queries.
#    Distractors fill the POST-WRITE gap in a fixed ascending order; dose ladder is NESTED (superset).
#    Writes/queries/targets/positions are IDENTICAL across doses.
# ==================================================================================================
def _base_canonical(cfg: Ext2Config):
    return json.dumps(dict(seq_len=cfg.seq_len, num_pairs=cfg.num_pairs, num_queries=cfg.num_queries,
                           num_keys=cfg.num_keys, num_vals=cfg.num_vals, write_frac=cfg.write_frac,
                           axis=cfg.stress_axis), sort_keys=True, separators=(",", ":"))


def _ex_seed(cfg, idx):
    h = hashlib.blake2b((_base_canonical(cfg) + f"|idx={idx}").encode(), digest_size=8).digest()
    return int.from_bytes(h, "big")        # process-stable (NOT builtin hash())


def gap_positions(cfg: Ext2Config):
    """The ordered POST-WRITE body positions eligible for distractors (ABSOLUTE indices incl. BOS).
    Deterministic + IDENTICAL for every example (write layout is fixed) -> the nested dose ladder shares one
    position list, so snapshot positions are constant across doses and examples."""
    body_len = cfg.seq_len - 1 - (1 + cfg.num_queries)          # minus BOS, minus (QSEP + queried keys)
    region = min(body_len, max(2 * cfg.num_pairs, int(round(cfg.write_frac * body_len))))
    chunk = max(2, region // cfg.num_pairs)
    occupied = set()
    for i in range(cfg.num_pairs):
        base = i * chunk
        occupied.update((base, base + 1))
    write_region_end = (cfg.num_pairs - 1) * chunk + 2          # first body slot after the last write pair
    gap = [p for p in range(write_region_end, body_len) if p not in occupied]
    # ABSOLUTE positions (account for BOS at index 0)
    return [1 + p for p in gap], body_len, write_region_end, chunk


def n_distractors_at(cfg: Ext2Config, dose):
    gp, _, _, _ = gap_positions(cfg)
    return int(round(dose * len(gp)))


def make_base_example(cfg: Ext2Config, idx):
    """Dose-INVARIANT scaffold: token ids with writes + FILL gap (NO distractors yet), labels, positions, and the
    per-example distractor KEY to place at each gap position (values differ per example; positions are shared)."""
    rng = np.random.default_rng(_ex_seed(cfg, idx))
    val_lo = KEY_LO + cfg.num_keys
    key_perm = rng.permutation(cfg.num_keys)
    written_keys = (key_perm[:cfg.num_pairs] + KEY_LO).tolist()
    values = (rng.integers(0, cfg.num_vals, size=cfg.num_pairs) + val_lo).tolist()
    free_keys = (key_perm[cfg.num_pairs:] + KEY_LO).tolist()    # disjoint hard-distractor pool
    kv = dict(zip(written_keys, values))

    q_idx = rng.permutation(cfg.num_pairs)[:cfg.num_queries]    # random query order (no copy shortcut)
    queried_keys = [written_keys[i] for i in q_idx]
    query = [QSEP] + queried_keys

    gp_abs, body_len, wre, chunk = gap_positions(cfg)
    body = [FILL] * body_len
    write_pos = {}
    for i, k in enumerate(written_keys):
        base = i * chunk
        body[base], body[base + 1] = k, values[i]
        write_pos[k] = base

    input_ids = [BOS] + body + query
    assert len(input_ids) == cfg.seq_len, (len(input_ids), cfg.seq_len)
    labels = [-100] * cfg.seq_len
    q_start = 1 + body_len + 1
    answer_positions, pairs = [], []
    for j, k in enumerate(queried_keys):
        p = q_start + j
        labels[p] = kv[k]
        answer_positions.append(p)
        wp = write_pos[k] + 1                                   # absolute (account for BOS)
        pairs.append(dict(key=int(k), value=int(kv[k]), write_pos=int(wp), query_pos=int(p),
                          distance=int(p - wp), write_seg=int(wp // cfg.seg)))
    # per-example distractor key at each shared gap position (cyclic over the disjoint free pool)
    dist_keys = [int(free_keys[j % len(free_keys)]) for j in range(len(gp_abs))]
    return dict(input_ids=input_ids, labels=labels, answer_positions=answer_positions, pairs=pairs,
                gap_abs=gp_abs, dist_keys=dist_keys, idx=idx,
                example_id=_example_id(input_ids, labels), val_lo=val_lo)


def materialize(cfg: Ext2Config, base_ex, dose):
    """Apply a dose to a base scaffold: set the first n_distractors gap positions to that example's distractor
    keys (NESTED: a higher dose is a superset). Returns a fresh input_ids list; labels/positions unchanged."""
    ids = list(base_ex["input_ids"])
    n = n_distractors_at(cfg, dose)
    for j in range(n):
        ids[base_ex["gap_abs"][j]] = base_ex["dist_keys"][j]
    return ids


# ---------------- tensor banks (fast, memory-light; distractors scattered on GPU) ----------------
def build_bank(cfg: Ext2Config, start, n):
    """Base ids + labels + shared gap positions + per-example distractor keys, as tensors. Dose is applied by
    scattering dist_keys[:, :n_dose] into gap_pos[:n_dose]."""
    exs = [make_base_example(cfg, i) for i in range(start, start + n)]
    gp_abs, _, _, _ = gap_positions(cfg)
    bank = dict(
        base_ids=torch.tensor([e["input_ids"] for e in exs], dtype=torch.long),
        labels=torch.tensor([e["labels"] for e in exs], dtype=torch.long),
        apos=torch.tensor([e["answer_positions"] for e in exs], dtype=torch.long),
        wseg=torch.tensor([[p["write_seg"] for p in e["pairs"]] for e in exs], dtype=torch.long),
        dist=torch.tensor([[p["distance"] for p in e["pairs"]] for e in exs], dtype=torch.long),
        dist_keys=torch.tensor([e["dist_keys"] for e in exs], dtype=torch.long),
        gap_abs=torch.tensor(gp_abs, dtype=torch.long),
        n=n, start=start,
        ids_hash=hashlib.blake2b(json.dumps([e["example_id"] for e in exs]).encode(), digest_size=8).hexdigest(),
        first_ids=[e["example_id"] for e in exs[:3]])
    return bank


def apply_dose_batch(cfg, bank, rows, dose, device=DEV):
    """Materialize a batch at a fixed dose. rows: LongTensor of row indices into the bank."""
    ids = bank["base_ids"][rows].clone()
    n = n_distractors_at(cfg, dose)
    if n > 0:
        pos = bank["gap_abs"][:n]                               # [n]
        ids[:, pos] = bank["dist_keys"][rows][:, :n]
    return ids.to(device)


def mb_vocab(cfg: Ext2Config):
    return KEY_LO + cfg.num_keys + cfg.num_vals


# ==================================================================================================
# 3. MC forward with snapshots + path-activation counters (§9)  [reuses the qualified aggregation fns]
# ==================================================================================================
def build_model(cfg, vocab, seed=42):
    torch.manual_seed(seed)
    return MQARDeltaModel(vocab, d_model=cfg.d_model, d_k=cfg.d_k, d_v=cfg.d_v, conv_k=cfg.conv_k).to(DEV).to(DT)


def mc_forward(cfg, model, ids, mode, reader, warm_start=True, path="chunked",
               recent_k=None, snap_mask=None, per_target_mask=None, return_gates=False, counters=None):
    """reader in {single, moving_average, grm}. Caches each segment's final recurrent state; aggregates cached +
    online reads. recent_k: keep most-recent-k. snap_mask: explicit kept indices (global ablation).
    per_target_mask: dict seg_of_last->None (unused here; per-target ablation handled by caller via snap_mask on
    grouped rows). counters: optional dict to accumulate path-activation counts."""
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
        if snap_mask is not None:
            idxs = [j for j in range(len(cached_states)) if j in snap_mask]
        elif recent_k is not None:
            idxs = list(range(len(cached_states)))[-recent_k:]
        else:
            idxs = list(range(len(cached_states)))
        avail = [cached_states[j] for j in idxs]
        avail_pools = [cached_pools[j] for j in idxs]
        cr = read_states(avail, qs) if avail else []
        if counters is not None and si == len(segs) - 1:
            counters["snapshotCandidates"] += len(cached_states) * B
            counters["snapshotReads"] += len(idxs) * B
            counters["readerCalls"] += B
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


# ==================================================================================================
# 4. Training (ONE stable recipe: single-state, MIXTURE over the dose ladder) + frozen reader
# ==================================================================================================
def train_backbone(cfg, model, bank, steps, mode, seed, log=None, reader="single", params=None):
    params = list(model.parameters()) if params is None else params
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gcpu = torch.Generator(device="cpu").manual_seed(seed)
    doses = cfg.doses()
    model.train()
    t0 = time.time()
    loss = torch.tensor(0.0)
    for s in range(steps):
        rows = torch.randint(0, bank["n"], (cfg.batch,), generator=gcpu)
        di = torch.randint(0, len(doses), (1,), generator=gcpu).item()   # domain-randomized dose per step
        ids = apply_dose_batch(cfg, bank, rows, doses[di])
        lab = bank["labels"][rows].to(DEV)
        logits = mc_forward(cfg, model, ids, mode, reader, True)   # warm_start=True (carry S across segments)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), lab.view(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if log and (s % 500 == 0 or s == steps - 1):
            log(f"      {mode} seed{seed} {reader} step {s:4d}/{steps} loss {loss.item():.4f}")
    return dict(steps=steps, wall_s=round(time.time() - t0, 1), final_loss=round(float(loss.item()), 4),
                trained_params=sum(p.numel() for p in params))


def tensor_hashes(model):
    return {n: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for n, p in model.state_dict().items()}


def backbone_sha256(model):
    """One digest over the FULL frozen backbone state_dict (order-stable)."""
    h = hashlib.sha256()
    for n, p in sorted(model.state_dict().items()):
        h.update(n.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ==================================================================================================
# 5. Evaluation: dose-swept BASE / MC + recovery-harm + counters + attribution
# ==================================================================================================
@torch.no_grad()
def eval_acc(cfg, model, bank, mode, reader, dose, batch=256, recent_k=None, snap_mask=None):
    model.eval()
    correct = total = 0
    for i in range(0, bank["n"], batch):
        rows = torch.arange(i, min(i + batch, bank["n"]))
        ids = apply_dose_batch(cfg, bank, rows, dose)
        lab = bank["labels"][rows].to(DEV)
        pred = mc_forward(cfg, model, ids, mode, reader, recent_k=recent_k, snap_mask=snap_mask).argmax(-1)
        mask = lab != -100
        correct += (pred[mask] == lab[mask]).sum().item(); total += mask.sum().item()
    return round(correct / total, 4) if total else 0.0


@torch.no_grad()
def eval_paired(cfg, model, bank, mode, dose, reader_mc="grm", recent_k=None, snap_mask=None, batch=256,
                capture=False, counters=None):
    """Per-target BASE(single) vs MC + recovery/harm with denominators. Optional attribution + path counters."""
    model.eval()
    base_ok, mc_ok, dists, wsegs = [], [], [], []
    attr, prox_reads, hist_nonzero, sel_hist = [], 0, 0, {}
    for i in range(0, bank["n"], batch):
        rows = torch.arange(i, min(i + batch, bank["n"]))
        ids = apply_dose_batch(cfg, bank, rows, dose)
        lab = bank["labels"][rows]; apos = bank["apos"][rows]; wseg = bank["wseg"][rows]; dist = bank["dist"][rows]
        base_pred = mc_forward(cfg, model, ids, mode, "single").argmax(-1).cpu()
        _mc = mc_forward(cfg, model, ids, mode, reader_mc, recent_k=recent_k, snap_mask=snap_mask,
                         return_gates=capture, counters=counters)
        mc_logits, ginfo = _mc if capture else (_mc, None)
        mc_pred = mc_logits.argmax(-1).cpu()
        for bb in range(ids.size(0)):
            for j, p in enumerate(apos[bb].tolist()):
                tgt = lab[bb, p].item()
                bok = int(base_pred[bb, p].item() == tgt); mok = int(mc_pred[bb, p].item() == tgt)
                base_ok.append(bok); mc_ok.append(mok)
                dists.append(int(dist[bb, j].item())); wsegs.append(int(wseg[bb, j].item()))
                if capture and ginfo is not None:
                    pos_in_seg = p - ginfo["seg_start"]
                    if 0 <= pos_in_seg < ginfo["gates"].shape[1] and ginfo["snap_idxs"]:
                        gvec = ginfo["gates"][bb, pos_in_seg]
                        snap_idxs = ginfo["snap_idxs"]
                        snap_g = gvec[:len(snap_idxs)]
                        am = int(snap_g.argmax().item()); argmax_seg = snap_idxs[am]
                        ws = int(wseg[bb, j].item())
                        prox = min((s for s in snap_idxs if s >= ws), default=snap_idxs[0])
                        hist_w = float(snap_g.sum().item())
                        if hist_w > 0.5:
                            hist_nonzero += 1
                        if argmax_seg == prox:
                            prox_reads += 1
                        sel_hist[argmax_seg] = sel_hist.get(argmax_seg, 0) + 1
                        if bok == 0 and mok == 1:
                            attr.append(dict(argmax_snap_seg=argmax_seg, write_seg=ws,
                                             gate_on_proximal=float(gvec[snap_idxs.index(prox)].item()),
                                             gate_online=float(gvec[-1].item())))
    base_ok = np.array(base_ok); mc_ok = np.array(mc_ok)
    n = len(base_ok); bw = base_ok == 0; bc = base_ok == 1
    n_recovered = int(mc_ok[bw].sum()) if bw.any() else 0
    n_harmed = int((1 - mc_ok[bc]).sum()) if bc.any() else 0
    res = dict(dose=dose, n_targets=int(n), base_acc=round(float(base_ok.mean()), 4),
               mc_acc=round(float(mc_ok.mean()), 4), net_delta=round(float(mc_ok.mean() - base_ok.mean()), 4),
               n_base_wrong=int(bw.sum()), n_base_correct=int(bc.sum()),
               n_recovered=n_recovered, n_harmed=n_harmed,
               RECOVERY_RATE=(round(n_recovered / int(bw.sum()), 4) if bw.any() else None),
               HARM_RATE=(round(n_harmed / int(bc.sum()), 4) if bc.any() else None),
               NET_RECOVERY_COUNT=n_recovered - n_harmed,
               NET_RECOVERY_RATE=round((n_recovered - n_harmed) / n, 4) if n else None)
    if capture:
        res["path_counters"] = dict(nonzeroHistoricalWeightQueries=hist_nonzero,
                                    targetProximalSnapshotReads=prox_reads,
                                    selectedSnapshotHistogram={str(kk): vv for kk, vv in sorted(sel_hist.items())})
        if attr:
            prox_plaus = float(np.mean([a["argmax_snap_seg"] >= a["write_seg"]
                                        and a["argmax_snap_seg"] <= a["write_seg"] + 1 for a in attr]))
            res["attribution"] = dict(n_recovered_examined=len(attr),
                                      frac_argmax_temporally_plausible=round(prox_plaus, 4),
                                      mean_gate_on_proximal=round(float(np.mean([a["gate_on_proximal"] for a in attr])), 4),
                                      mean_gate_online=round(float(np.mean([a["gate_online"] for a in attr])), 4),
                                      SNAPSHOT_ATTRIBUTION="DESCRIPTIVE")
        else:
            res["attribution"] = dict(n_recovered_examined=0, SNAPSHOT_ATTRIBUTION="DESCRIPTIVE")
    return res


# ==================================================================================================
# 6. Retention-curve metrics (§11): AURC, D50, D80_D20_WIDTH  (preregistered LINEAR interpolation)
# ==================================================================================================
def _interp_crossing(doses, accs, level):
    """First dose (ascending) where a DECREASING acc curve crosses `level`, by linear interpolation.
    Returns None if it never crosses (stays above or below)."""
    for i in range(1, len(doses)):
        a0, a1 = accs[i - 1], accs[i]
        if (a0 - level) * (a1 - level) <= 0 and a0 != a1:
            frac = (a0 - level) / (a0 - a1)
            return doses[i - 1] + frac * (doses[i] - doses[i - 1])
    return None


def _trapz(y, x):
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")   # numpy>=2 renamed trapz->trapezoid
    return float(fn(y, x))


def retention_metrics(doses, accs):
    doses = list(doses); accs = list(accs)
    lo, hi = doses[0], doses[-1]
    span = (hi - lo) or 1.0
    aurc = float(_trapz(accs, doses) / span)                   # normalized area under retention curve
    d50 = _interp_crossing(doses, accs, 0.5)
    d80 = _interp_crossing(doses, accs, 0.8)
    d20 = _interp_crossing(doses, accs, 0.2)
    width = (abs(d20 - d80) if (d20 is not None and d80 is not None) else None)
    return dict(AURC_RETENTION=round(aurc, 4),
                D50=(round(d50, 4) if d50 is not None else None),
                D80=(round(d80, 4) if d80 is not None else None),
                D20=(round(d20, 4) if d20 is not None else None),
                D80_D20_WIDTH=(round(width, 4) if width is not None else None))


# ==================================================================================================
# 7. Efficiency / performance (§18-19): prewarm -> warm steady-state; bytes + times
# ==================================================================================================
@torch.no_grad()
def efficiency(cfg, model, bank, mode, dose, reps=30, batch=64):
    ids = apply_dose_batch(cfg, bank, torch.arange(batch), dose)
    x, q, k, v, g, beta, _ = model.project(ids)
    B, Lx, _ = q.shape
    seg = cfg.seg
    segs = [(i, min(i + seg, Lx)) for i in range(0, Lx, seg)]

    def sync():
        torch.cuda.synchronize() if DEV == "cuda" else None

    def timeit(fn, warmup=5):
        for _ in range(warmup):
            fn()
        sync(); t0 = time.time()
        for _ in range(reps):
            fn()
        sync(); return (time.time() - t0) / reps * 1000

    states = []
    S = None
    for (a, b) in segs:
        _, S = run_recurrence(mode, q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b], seg, S, "chunked")
        states.append(S)

    def do_recurrent():
        S = None
        for (a, b) in segs:
            _, S = run_recurrence(mode, q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b], seg, S, "chunked")

    def do_read():
        for i, (a, b) in enumerate(segs):
            if i:
                read_states(states[:i], q[:, a:b])

    matrix = cfg.d_k * cfg.d_v * 4
    conv = model.blk.conv_dim * (cfg.conv_k - 1) * 4
    n_ckpt = len(segs)
    peak = None
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats(); mc_forward(cfg, model, ids, mode, "grm"); sync()
        peak = int(torch.cuda.max_memory_allocated())
    return dict(mode=mode, dose=dose, n_segments=n_ckpt, batch=batch, reps=reps,
                bytes=dict(live_recurrent_matrix=matrix, live_conv_state=conv,
                           historical_snapshots_at_full=matrix * (n_ckpt - 1),
                           reader_params=model.blk.w_u.weight.numel() * 4,
                           peak_vram=peak),
                warm_ms=dict(projection_conv=round(timeit(lambda: model.project(ids)), 3),
                             recurrent_update=round(timeit(do_recurrent), 3),
                             snapshot_read=round(timeit(do_read), 3),
                             base_total=round(timeit(lambda: mc_forward(cfg, model, ids, mode, "single")), 3),
                             mc_total=round(timeit(lambda: mc_forward(cfg, model, ids, mode, "grm")), 3)))


# ==================================================================================================
# 8. Snapshot identity (§8): record identity fields + assert encoded position == represented position
# ==================================================================================================
@torch.no_grad()
def snapshot_identity(cfg, model, bank, mode, dose, git_head):
    """For ONE representative holdout example, capture each per-segment snapshot's identity. Distinguishes a
    HISTORICAL_RECURRENT_STATE_SNAPSHOT (S only, what MC reads) from a FULL_RESTORABLE_SEQUENCE_CHECKPOINT
    (S + conv_state). Asserts the encoded sequencePosition equals the position the state actually represents."""
    ids = apply_dose_batch(cfg, bank, torch.arange(1), dose)   # single example
    seg = cfg.seg
    x, q, k, v, g, beta, _ = model.project(ids)
    B, Lx, _ = q.shape
    segs = [(i, min(i + seg, Lx)) for i in range(0, Lx, seg)]
    wsha = backbone_sha256(model)
    snaps = []
    S_prev = None
    conv_state = None
    pos_ok = True
    for si, (a, b) in enumerate(segs):
        # recurrence snapshot (S) over this segment carrying S_prev
        _, S_fin = run_recurrence(mode, q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b], seg, S_prev, "scan")
        # full-restorable conv boundary from the FULL prefix [0:b] (kernel-1 window ending at b-1)
        _, _, _, _, _, _, cstate = model.project(ids[:, :b], None)
        represented_pos = b - 1                                 # the state summarizes tokens [0, b) -> last idx b-1
        encoded_pos = a + (b - a) - 1                           # by construction the segment's last absolute idx
        pos_ok &= (encoded_pos == represented_pos)
        snaps.append(dict(exampleId=bank["first_ids"][0], segmentIndex=si,
                          sequencePosition=int(represented_pos), recurrenceBoundaryId=f"seg{si}_end@{b}",
                          stateSha256=_sha256_bytes(S_fin.detach().cpu().numpy().tobytes()),
                          convStateSha256=_sha256_bytes(cstate.detach().cpu().numpy().tobytes()),
                          modelWeightsSha256=wsha, recurrenceSemanticsId=f"{mode}:qwen3next_torch_recurrent_port",
                          dtype="float32", backend=DEV, kernel_source_revision=git_head[:12],
                          snapshot_kind="HISTORICAL_RECURRENT_STATE_SNAPSHOT (S); FULL checkpoint adds convStateSha256"))
        S_prev = S_fin
    return dict(mode=mode, dose=dose, n_snapshots=len(snaps),
                SNAPSHOT_POSITION_IDENTITY="PASS" if pos_ok else "FAIL", snapshots=snaps)


# ==================================================================================================
# 9. Target-aware ablation (§13): per-target proximal / irrelevant / random(excl) / sham
# ==================================================================================================
@torch.no_grad()
def target_aware_ablation(cfg, model, bank, mode, dose, batch=256):
    """Group targets by write_seg; for each group evaluate FULL and drop-that-group's-proximal snapshot. Also a
    global DROP_IRRELEVANT (late snapshot) and DROP_RANDOM (deterministic, EXCLUDING proximal set + irrelevant),
    and a SHAM (drop-none but same read-count via recent_k=all -> no target relation). Reports effects overall
    and restricted to BASE_WRONG->MC_CORRECT recovered targets."""
    model.eval()
    nseg = math.ceil(cfg.seq_len / cfg.seg)
    n_snap = nseg - 1                                          # snapshots visible to the last (query) segment
    full = set(range(n_snap))
    irrelevant = n_snap - 1                                    # a late snapshot far from the early writes
    # proximal set = the write segments actually used by targets (writes live in the early quarter)
    prox_candidates = sorted(set(int(w) for w in bank["wseg"].reshape(-1).tolist()) & full)
    rng = np.random.default_rng(20260811)
    excl = set(prox_candidates) | {irrelevant}
    random_pool = [s for s in range(n_snap) if s not in excl]
    assert random_pool, "no independent random snapshot available (grid too small)"
    random_idx = int(rng.choice(random_pool))
    # HARD asserts (audit §3): random control must NOT coincide with proximal set or irrelevant
    assert random_idx not in prox_candidates and random_idx != irrelevant, \
        f"random control {random_idx} collides with proximal/irrelevant"

    def paired_over_rows(snap_mask):
        """Global-mask recovery: fraction of BASE-wrong targets that MC (with this snap_mask) gets right."""
        base_ok, mc_ok = [], []
        for i in range(0, bank["n"], batch):
            rows = torch.arange(i, min(i + batch, bank["n"]))
            ids = apply_dose_batch(cfg, bank, rows, dose)
            lab = bank["labels"][rows]; apos = bank["apos"][rows]
            base_pred = mc_forward(cfg, model, ids, mode, "single").argmax(-1).cpu()
            mc_pred = mc_forward(cfg, model, ids, mode, "grm", snap_mask=snap_mask).argmax(-1).cpu()
            for bb in range(ids.size(0)):
                for j, p in enumerate(apos[bb].tolist()):
                    tgt = lab[bb, p].item()
                    base_ok.append(int(base_pred[bb, p].item() == tgt))
                    mc_ok.append(int(mc_pred[bb, p].item() == tgt))
        base_ok = np.array(base_ok); mc_ok = np.array(mc_ok)
        bw = base_ok == 0
        return dict(recovery=(float(mc_ok[bw].mean()) if bw.any() else None),
                    n_base_wrong=int(bw.sum()), mc_acc=round(float(mc_ok.mean()), 4) if len(mc_ok) else None)

    # per-target proximal drop, done at write-seg GROUP granularity (correct + tractable)
    groups = {}
    for i in range(0, bank["n"], batch):
        rows = torch.arange(i, min(i + batch, bank["n"]))
        ids = apply_dose_batch(cfg, bank, rows, dose)
        lab = bank["labels"][rows]; apos = bank["apos"][rows]; wseg = bank["wseg"][rows]
        base_pred = mc_forward(cfg, model, ids, mode, "single").argmax(-1).cpu()
        # cache per-group mc predictions
        gcache = {}
        for bb in range(ids.size(0)):
            for j, p in enumerate(apos[bb].tolist()):
                ws = int(wseg[bb, j].item()); prox = min((s for s in full if s >= ws), default=min(full))
                if prox not in gcache:
                    gcache[prox] = mc_forward(cfg, model, ids, mode, "grm", snap_mask=full - {prox}).argmax(-1).cpu()
                tgt = lab[bb, p].item(); bok = int(base_pred[bb, p].item() == tgt)
                mok = int(gcache[prox][bb, p].item() == tgt)
                groups.setdefault("dtp", {"base": [], "mc": []})
                groups["dtp"]["base"].append(bok); groups["dtp"]["mc"].append(mok)
    dtp_base = np.array(groups["dtp"]["base"]); dtp_mc = np.array(groups["dtp"]["mc"])
    dtp_bw = dtp_base == 0
    rec_dtp = float(dtp_mc[dtp_bw].mean()) if dtp_bw.any() else None

    full_v = paired_over_rows(full)
    irr_v = paired_over_rows(full - {irrelevant})
    rnd_v = paired_over_rows(full - {random_idx})
    sham_v = paired_over_rows(full)                            # same read-count, no target relation (== full)
    rec_full = full_v["recovery"] or 0.0

    def drop(x):
        return round(rec_full - (x or 0.0), 4)

    signal = ("SUPPORTED" if (rec_dtp is not None and (rec_full - rec_dtp) >= cfg.margin
                              and (rec_full - rec_dtp) > (rec_full - (irr_v["recovery"] or 0.0)) + cfg.margin
                              and (rec_full - rec_dtp) > (rec_full - (rnd_v["recovery"] or 0.0)) + cfg.margin)
              else ("NOT_DETECTED" if (rec_dtp is None or rec_full - rec_dtp <= 0) else "INCONCLUSIVE"))
    return dict(mode=mode, dose=dose, n_snap=n_snap, proximal_candidates=prox_candidates,
                irrelevant_idx=irrelevant, random_idx=random_idx,
                RANDOM_EXCLUDES_PROXIMAL_AND_IRRELEVANT=True,
                recovery_full=round(rec_full, 4), recovery_drop_target_proximal=(round(rec_dtp, 4) if rec_dtp is not None else None),
                recovery_drop_irrelevant=round(irr_v["recovery"] or 0.0, 4),
                recovery_drop_random=round(rnd_v["recovery"] or 0.0, 4),
                drop_from_target_proximal=drop(rec_dtp), drop_from_irrelevant=drop(irr_v["recovery"]),
                drop_from_random=drop(rnd_v["recovery"]),
                HISTORICAL_SNAPSHOT_CAUSAL_SIGNAL=signal)


# ==================================================================================================
# 10. Hierarchical bootstrap (§15): backbone(seed) -> sequence -> target
# ==================================================================================================
def hierarchical_bootstrap_delta_aurc(per_seed_curves, doses, iters, seed=12345):
    """per_seed_curves: {seed: {'base': [acc per dose], 'mc': [acc per dose]}} (curve-level scalars per backbone).
    With only a few backbones this is LOW-POWER: report direction/stability/heterogeneity, not population p-values.
    Resamples backbones with replacement; CI is over training randomness (cluster = backbone)."""
    rng = np.random.default_rng(seed)
    seeds = list(per_seed_curves.keys())
    per_seed_delta = {}
    for s in seeds:
        c = per_seed_curves[s]
        aur_b = retention_metrics(doses, c["base"])["AURC_RETENTION"]
        aur_m = retention_metrics(doses, c["mc"])["AURC_RETENTION"]
        per_seed_delta[s] = round(aur_m - aur_b, 4)
    deltas = np.array([per_seed_delta[s] for s in seeds])
    boots = []
    for _ in range(iters):
        pick = rng.choice(len(seeds), size=len(seeds), replace=True)
        boots.append(float(deltas[pick].mean()))
    boots = np.array(boots)
    return dict(per_seed_DELTA_AURC=per_seed_delta,
                mean_DELTA_AURC=round(float(deltas.mean()), 4),
                cluster_bootstrap_CI95=[round(float(np.percentile(boots, 2.5)), 4),
                                        round(float(np.percentile(boots, 97.5)), 4)],
                direction_all_positive=bool(all(deltas > 0)),
                heterogeneity_range=[round(float(deltas.min()), 4), round(float(deltas.max()), 4)],
                n_backbones=len(seeds),
                POWER_CAVEAT="n_backbones small -> establishes direction/stability/heterogeneity, NOT population inference")


# ==================================================================================================
# 11. Pre-registration writer
# ==================================================================================================
def write_preregistration(cfg: Ext2Config, outdir):
    os.makedirs(outdir, exist_ok=True)
    gc = grid_selfcheck(cfg)
    machine = dict(packet="RNN-05B-EXT2", config=asdict(cfg), challenge_grid=cfg.challenge_grid(),
                   challengeGridSha256=cfg.challenge_grid_sha256(), configSha256=cfg.config_sha256(),
                   grid_selfcheck=gc, select_rule=cfg.select_rule(), sourceGitHead=_git_head())
    json.dump(machine, open(os.path.join(outdir, "machine_config.json"), "w"), indent=2)
    doses = list(cfg.doses())
    md = f"""# RNN-05B-EXT2 — PRE-REGISTRATION (written BEFORE any outcome-bearing run)

**FINAL planned synthetic H3 test.** Question (§1): for ONE already-trained, stable, FROZEN GDN/DN
representation, does progressively increasing *inference-time* retention pressure produce **graded** loss of old
associations, and can HISTORICAL recurrent-state snapshots recover associations the FINAL recurrent state no
longer retrieves? This ISOLATES inference-time forgetting from TRAINING INSTABILITY (the RNN-05B-EXT confound).

This file, `machine_config.json`, and the executed constants derive from ONE frozen `Ext2Config`.
`challengeGridSha256 = {cfg.challenge_grid_sha256()}` (self-check
`{gc['GRID_IDENTITY_SELFCHECK']}`) must appear IDENTICALLY here, in the machine config, in
`BASE_QUALIFICATION.json`, in run metadata, and in the final results. Mismatch = STOP.

## Backbone reuse decision (§2)
The RNN-05B DN/GDN backbones were **NOT saved to disk** and were trained on a **different (capacity) MQAR**
distribution, so exact-artifact reuse is INVALID and IMPOSSIBLE. Per the §2 fallback: **train each preregistered
seed ONCE** under one stable recipe, **save + SHA-256 + freeze**, and use the **identical** weights for EVERY
stress point. `BACKBONE_REUSE = RETRAIN_ONCE_THEN_FREEZE`.

## Architecture (RNN-05B-qualified family, UNCHANGED)
`MQARDeltaModel` d_model={cfg.d_model}, d_k={cfg.d_k}, d_v={cfg.d_v}, conv_k={cfg.conv_k}; MC/chunk segment
seg={cfg.seg}. No recurrence-equation edits, no deeper readers, no GDN-mechanism edits, no new kernels. Reader =
the existing `w_u` grm connector only. Eager sequential scan remains the correctness reference (§20).

## Task — memory-bound MQAR at FIXED seq_len (temporal pressure, NOT capacity overload)
- seq_len = **{cfg.seq_len}** (FIXED -> snapshot/segment positions constant across doses; §8 identity holds).
- num_pairs = **{cfg.num_pairs}** (FAR below the RNN-05B capacity cliff ~40 @ d_k=64), num_queries = {cfg.num_queries},
  num_keys = {cfg.num_keys}, num_vals = {cfg.num_vals}. Writes in the EARLY {cfg.write_frac:.0%} of the body; queries at the end.

## Stress axis (§5) — NESTED MONOTONIC, inference-only
`{cfg.stress_axis}`: distractor keys fill the POST-WRITE retention gap in a FIXED ascending order; the dose
ladder is **nested** (a higher dose = the SAME base example with MORE gap slots converted to distractor keys, a
superset). Writes, queries, target values and ALL positions are IDENTICAL across doses. Pair count is NOT
increased. Dose ladder = {doses}.

## ONE stable recipe (§2)
Each seed trained ONCE, single-state, on a **MIXTURE over the dose ladder** (domain-randomized dose per step) so
one representation is competent across the range; steps={cfg.steps_bb}, lr={cfg.lr}, batch={cfg.batch},
pool_train={cfg.pool_train}. Then FREEZE + SHA-256. The SAME frozen weights face every stress point. Seeds
(ALL count; **no seed screening** — RNN-05B-EXT audit §7): GDN {list(cfg.gdn_seeds)} (load-bearing, Qwen target),
DN {list(cfg.dn_seeds)} (load-bearing), LA {list(cfg.la_seeds)} (mechanistic control). Disjoint TRAIN /
DEV / FINAL-HOLDOUT example ranges; pinned id hashes.

## Control-flow invariant (§6) — BASE qualification BEFORE any MC
Frozen backbones -> ALL preregistered doses for ALL seeds -> persist `BASE_QUALIFICATION.json`
(challengeGridSha256, backboneSha256[], sourceGitHead, configSha256, exampleSetSha256, stressAxis, per-seed
retention curves, qualified common region, verdict) -> verify hashes + grid digest -> graded-region gate ->
**ONLY THEN** MC/reader. The MC entrypoint LOADS+VERIFIES the artifact; absent/mismatched/unqualified => STOP.

## Graded-region gate (§7)
> {cfg.select_rule()}

`FIXED_BACKBONE_GRADED_REGION = QUALIFIED | BLOCKED`. No graded region => `H3_TESTABILITY =
BLOCKED_FIXED_BACKBONE`, STOP, no MC, **no EXT3**. (Qualifying on one cell in an arbitrary band is explicitly
insufficient; a COMMON overlapping graded region across the frozen GDN seeds is required.)

## Primary paired experiment (§10)
For every qualified frozen backbone: **A** BASE final recurrent state only · **B** parameter-free historical
snapshot aggregation (moving average) · **C** the same small trained `w_u` reader (backbone frozen; tensor
hashes before/after must prove BACKBONE_WEIGHT_MUTATION = 0). Identical examples. Reader saved durably + SHA-256.
LA remains the mechanistic control.

## Retention-curve metrics (§11) — preregistered LINEAR interpolation
Per seed/method: accuracy vs dose; AURC_RETENTION (normalized trapezoid), D50, D80/D20 transition width,
DELTA_AURC = AURC_MC − AURC_BASE, DELTA_D50. The interpolation/curve procedure is LINEAR on the ladder and is
NOT changed after seeing MC. Raw per-dose scores remain authoritative.

## Recovery / harm (§12) — expose denominators
Per seed & dose: n_base_wrong, n_recovered, RECOVERY_RATE; n_base_correct, n_harmed, HARM_RATE;
NET_RECOVERY_COUNT = n_recovered − n_harmed; NET_RECOVERY_RATE. Denominators always exposed; per-seed rates are
NOT averaged and reported as pooled query rates.

## Target-aware ablation (§13)
Per target, the proximal snapshot = the first snapshot at/after its WRITE segment. Compare FULL /
DROP_TARGET_PROXIMAL / DROP_IRRELEVANT (late) / DROP_RANDOM (deterministic, EXCLUDING the proximal set and the
irrelevant index — asserted in code) / SHAM. Report aggregate effects AND effects restricted to
BASE_WRONG->MC_CORRECT. Gate argmax is DESCRIPTIVE only; causal support requires this ablation (§14).

## Statistics (§15) & SESOI (§16)
Hierarchy: training seed / frozen backbone -> sequence -> target. Cluster-aware (backbone-level) bootstrap;
with {len(cfg.gdn_seeds)} backbones this establishes **direction / stability / heterogeneity**, NOT population
inference. PRIMARY SESOI on DELTA_AURC = **{cfg.sesoi_delta_aurc}**. *Justification*: MC's cost is storing
(n_ckpt−1) historical matrix snapshots (each {cfg.d_k*cfg.d_v*4} B) + per-segment read/gate latency — at
seq_len {cfg.seq_len}, seg {cfg.seg} that is a ~{math.ceil(cfg.seq_len/cfg.seg)-1}x live-state-memory multiplier;
DELTA_AURC {cfg.sesoi_delta_aurc} (5 pts of average retention area) is the smallest average lift that plausibly
justifies that storage+latency for a post-hoc memory mechanism. The old **3% margin is retained ONLY as an
`OPERATOR_HEURISTIC`** for direction labels, NOT as scientific authority. Decision: CI clearly above +SESOI ->
meaningful positive; CI spanning +SESOI and trivial -> INCONCLUSIVE/DIRECTIONAL; CI fully inside ±SESOI ->
PRACTICALLY_EQUIVALENT; negative -> REGRESSION. p>0.05 is NOT equivalence.

## Efficiency / performance (§18-19)
Report live matrix/conv bytes, historical-snapshot bytes, reader bytes, peak VRAM; recurrent-update / snapshot-
read / reader / total latency with prewarm -> warm steady-state (compile/cold separated). Derive
RECOVERY_PER_MiB, DELTA_AURC_PER_KiB, DELTA_AURC_PER_ADDED_ms. Efficiency is NOT inferred from bytes alone.

## Decision policy (§21) — FINAL synthetic H3
- **Case A** no graded fixed-backbone region -> H3_TESTABILITY=BLOCKED_FIXED_BACKBONE, QWEN_GDN_TRANSPLANT_GATE=
  DEFER, SYNTHETIC_DENSE_MC=PARK, STOP, NO EXT3.
- **Case B** qualified region, DELTA_AURC practically equivalent to 0 or negative -> H3=
  NOT_DETECTED_IN_QUALIFIED_REGIME, gate=DEFER, DENSE_POST_HOC_MEMORY_CACHING=PARK, STOP, NO EXT3.
- **Case C** defensible positive (graded region + positive DELTA_AURC/DELTA_D50 + directionally consistent GDN +
  recovery >> harm + target-proximal ablation supports mechanism + random/irrelevant do NOT reproduce it + LA
  does NOT show it + path counters prove historical-state use) -> H3=POSITIVE_CANDIDATE, gate=PASS_CANDIDATE
  (authorizes only DESIGN of a separate Qwen qualification packet; no automatic Qwen run).

## Guardrails
No Qwen weights · no llama.cpp/serving/deploy · no TPTT · no RNN-05C · no StateX/DART/Sparse-Delta/GDN-2/FG2-GDN/
ReplaySSM · no FLA · no new kernels · not pushed. Budget target < 1 GPU-hr (hard 2). RNN-05B/EXT evidence
immutable. External research pointers recorded for a possible RNN-06 packet only.
"""
    open(os.path.join(outdir, "PRE_REGISTRATION.md"), "w", encoding="utf-8").write(md)
    return machine


# ==================================================================================================
# 12. Nested-generator self-qualification
# ==================================================================================================
def generator_selftest(cfg: Ext2Config, out_path=None):
    exs = [make_base_example(cfg, i) for i in range(64)]
    doses = cfg.doses()
    checks = {}
    checks["length_exact_all_doses"] = all(len(materialize(cfg, e, d)) == cfg.seq_len for e in exs for d in doses)
    # kv survives at dose 0 AND writes/queries never overwritten at max dose (nested distractors avoid them)
    surv = True
    maxd = doses[-1]
    for e in exs:
        ids0 = materialize(cfg, e, 0.0); idsM = materialize(cfg, e, maxd)
        for pr in e["pairs"]:
            wp = pr["write_pos"]
            surv &= (ids0[wp] == pr["key"] and ids0[wp + 1] == pr["value"] and ids0[pr["query_pos"]] == pr["key"])
            surv &= (idsM[wp] == pr["key"] and idsM[wp + 1] == pr["value"] and idsM[pr["query_pos"]] == pr["key"])
    checks["writes_queries_survive_all_doses"] = bool(surv)
    # NESTED: higher dose is a strict superset of distractor positions changed vs base
    nested = True
    for e in exs:
        prev = set()
        base = e["input_ids"]
        for d in doses:
            ids = materialize(cfg, e, d)
            changed = {p for p in range(cfg.seq_len) if ids[p] != base[p]}
            nested &= prev.issubset(changed)
            prev = changed
    checks["nested_monotonic"] = bool(nested)
    # no answer leak: target value never appears in the query region
    leak = False
    for e in exs:
        qregion = set(e["input_ids"][min(e["answer_positions"]):])
        for p in e["answer_positions"]:
            leak |= (e["labels"][p] in qregion)
    checks["no_answer_leak"] = (not leak)
    # multi-value keys across examples (no lexical shortcut)
    kmap = {}
    for e in exs:
        for pr in e["pairs"]:
            kmap.setdefault(pr["key"], set()).add(pr["value"])
    checks["no_lexical_shortcut"] = sum(1 for v in kmap.values() if len(v) > 1) >= max(1, len(kmap) // 2)
    dists = [pr["distance"] for e in exs for pr in e["pairs"]]
    checks["mean_write_query_distance"] = float(np.mean(dists))
    checks["memory_bound_design"] = bool(np.median(dists) > 0.5 * cfg.seq_len)
    # process stability of example ids
    code = ("import json,sys;sys.path.insert(0,r'%s');from rnn_05b_ext2 import Ext2Config,make_base_example;"
            "c=Ext2Config();print(json.dumps([make_base_example(c,i)['example_id'] for i in range(16)]))"
            % os.path.dirname(os.path.abspath(__file__)))
    a = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    inproc = [make_base_example(cfg, i)["example_id"] for i in range(16)]
    checks["process_stable"] = (a.returncode == 0 and json.loads(a.stdout.strip()) == inproc)
    # dose response sanity: n_distractors strictly increases with dose (except possibly 0->0)
    ns = [n_distractors_at(cfg, d) for d in doses]
    checks["n_distractors_per_dose"] = ns
    checks["dose_monotonic"] = all(ns[i] <= ns[i + 1] for i in range(len(ns) - 1)) and ns[-1] > ns[0]
    passed = all(checks[kk] for kk in ["length_exact_all_doses", "writes_queries_survive_all_doses",
                 "nested_monotonic", "no_answer_leak", "no_lexical_shortcut", "memory_bound_design",
                 "process_stable", "dose_monotonic"])
    res = dict(packet="RNN-05B-EXT2", component="nested memory-bound MQAR generator self-qualification",
               GENERATOR_SELFTEST="PASS" if passed else "FAIL", checks=checks,
               challengeGridSha256=cfg.challenge_grid_sha256(), numpy=np.__version__, python=sys.version.split()[0])
    if out_path:
        json.dump(res, open(out_path, "w"), indent=2)
    print(json.dumps({"GENERATOR_SELFTEST": res["GENERATOR_SELFTEST"], "checks": checks}, indent=2))
    return res


# ==================================================================================================
# 13. Main run
# ==================================================================================================
def _pools(cfg):
    tr = build_bank(cfg, 0, cfg.pool_train)
    dv = build_bank(cfg, cfg.pool_train, cfg.pool_eval)
    ho = build_bank(cfg, cfg.pool_train + cfg.pool_eval, cfg.pool_eval)
    return dict(train=tr, dev=dv, hold=ho)


def run(cfg: Ext2Config, outdir, artifacts=None, smoke=False):
    os.makedirs(outdir, exist_ok=True)
    art = artifacts or outdir
    os.makedirs(art, exist_ok=True)
    logf = open(os.path.join(outdir, "run.log"), "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    if smoke:
        cfg = Ext2Config(steps_bb=120, steps_rd=80, batch=16, pool_train=256, pool_eval=128,
                         dose_ladder=(0.0, 0.2, 0.4, 0.6), gdn_seeds=(42, 43), dn_seeds=(42,), la_seeds=(42,),
                         grade_hi=-1.0, grade_lo=1.1, mid_lo=-1.0, mid_hi=1.1, min_mid_doses=0,
                         boot_iters=200)  # force-qualify (all doses "graded") to exercise the MC/ablation/stats paths

    git_head = _git_head()
    vocab = mb_vocab(cfg)
    doses = list(cfg.doses())
    t_start = time.time()
    grid_sha = cfg.challenge_grid_sha256()
    gc = grid_selfcheck(cfg)

    R = dict(meta=dict(packet="RNN-05B-EXT2", device=DEV, torch=torch.__version__, numpy=np.__version__,
                       smoke=smoke, sourceGitHead=git_head, config=asdict(cfg),
                       challengeGridSha256=grid_sha, configSha256=cfg.config_sha256(),
                       grid_selfcheck=gc, artifacts_dir=art),
             substrate_sanity={}, backbones={}, base_qualification=None, graded_gate=None,
             snapshot_identity={}, mc={}, curves={}, ablation={}, stats={}, efficiency={}, outcomes={})

    def snap():
        json.dump(R, open(os.path.join(outdir, "rnn05bext2_results.json"), "w"), indent=2)

    log(f"[CFG] challengeGridSha256={grid_sha} selfcheck={gc['GRID_IDENTITY_SELFCHECK']} head={git_head[:12]}")
    if gc["GRID_IDENTITY_SELFCHECK"] != "PASS":
        R["outcomes"] = dict(STATUS="ABORTED_GRID_IDENTITY"); snap(); logf.close(); return

    pools = _pools(cfg)
    R["meta"]["exampleSetSha256"] = dict(train=pools["train"]["ids_hash"], dev=pools["dev"]["ids_hash"],
                                         holdout=pools["hold"]["ids_hash"])

    # ---- P0 substrate sanity (cheap reconfirm; already qualified in RNN-05B) ----
    log("[P0] substrate sanity: reference parity + full-module checkpoint/restore + request isolation")
    gm = build_model(cfg, vocab, seed=0)
    gcpu = torch.Generator().manual_seed(7)
    gids = torch.randint(0, vocab, (6, 192), generator=gcpu).to(DEV)
    gidsB = torch.randint(0, vocab, (6, 192), generator=torch.Generator().manual_seed(8)).to(DEV)
    for mode in MODES:
        par = reference_parity(gm, gids, mode, cfg.seg)
        life = checkpoint_restore_full_module(gm, gids, mode, 96, cfg.seg)
        iso = request_isolation(gm, gids, gidsB, mode, cfg.seg)
        R["substrate_sanity"][mode] = dict(REFERENCE_PARITY=par["PARITY"],
                                            FULL_MODULE_LIFECYCLE=life["FULL_MODULE_CHECKPOINT_RESTORE"],
                                            REQUEST_ISOLATION=iso["REQUEST_STATE_ISOLATION"])
        log(f"  {mode}: parity={par['PARITY']} lifecycle={life['FULL_MODULE_CHECKPOINT_RESTORE']} iso={iso['REQUEST_STATE_ISOLATION']}")
    del gm
    torch.cuda.empty_cache() if DEV == "cuda" else None
    snap()

    # ---- P1 TRAIN ONCE + FREEZE + HASH each seed (single-state mixture over dose ladder) ----
    log("[P1] train ONCE per seed (single-state, mixture over dose ladder) -> freeze -> SHA-256")
    seeds_by_mode = dict(gdn=list(cfg.gdn_seeds), dn=list(cfg.dn_seeds), la=list(cfg.la_seeds))
    for mode in MODES:
        R["backbones"][mode] = []
        for seed in seeds_by_mode[mode]:
            m = build_model(cfg, vocab, seed=seed)
            st = train_backbone(cfg, m, pools["train"], cfg.steps_bb, mode, seed, log=log, reader="single")
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()
            sha = backbone_sha256(m)
            ckpt = os.path.join(art, f"ext2_{mode}_seed{seed}_backbone.pt")
            torch.save({"state_dict": m.state_dict(), "mode": mode, "seed": seed, "vocab": vocab,
                        "config": asdict(cfg), "backbone_sha256": sha, "challengeGridSha256": grid_sha}, ckpt)
            file_sha = _sha256_bytes(open(ckpt, "rb").read())
            R["backbones"][mode].append(dict(seed=seed, backbone_sha256=sha, checkpoint_path=ckpt,
                                             checkpoint_file_sha256=file_sha, train_stat=st))
            log(f"  {mode} seed={seed}: trained+frozen loss={st['final_loss']} sha={sha[:12]} -> {os.path.basename(ckpt)}")
            del m
            torch.cuda.empty_cache() if DEV == "cuda" else None
            snap()

    # ---- P2 BASE QUALIFICATION (ALL doses, ALL seeds, FROZEN weights) -> persist -> gate ----
    log("[P2] BASE qualification: ALL doses x ALL seeds on FROZEN backbones (BASE-only, load-bearing gate)")
    base_curves = {}
    for mode in MODES:
        base_curves[mode] = {}
        for bb in R["backbones"][mode]:
            m = _reload_backbone(cfg, bb["checkpoint_path"], vocab, expect_sha=bb["backbone_sha256"])
            curve = [eval_acc(cfg, m, pools["hold"], mode, "single", d) for d in doses]
            base_curves[mode][bb["seed"]] = curve
            log(f"  {mode} seed={bb['seed']} BASE curve: " + " ".join(f"{d}:{a}" for d, a in zip(doses, curve)))
            del m
            torch.cuda.empty_cache() if DEV == "cuda" else None
    # graded-region gate on GDN (load-bearing), DN reported alongside
    gate = graded_region_gate(cfg, base_curves["gdn"], doses)
    R["graded_gate"] = gate
    base_qual = dict(packet="RNN-05B-EXT2", challengeGridSha256=grid_sha, configSha256=cfg.config_sha256(),
                     sourceGitHead=git_head, exampleSetSha256=R["meta"]["exampleSetSha256"],
                     stressAxis=cfg.stress_axis, dose_ladder=doses,
                     backboneSha256={mode: {bb["seed"]: bb["backbone_sha256"] for bb in R["backbones"][mode]}
                                     for mode in MODES},
                     per_seed_base_curves=base_curves,
                     gdn_graded_region=gate,
                     FIXED_BACKBONE_GRADED_REGION=gate["FIXED_BACKBONE_GRADED_REGION"],
                     H3_TESTABILITY=("BLOCKED_FIXED_BACKBONE" if gate["FIXED_BACKBONE_GRADED_REGION"] == "BLOCKED"
                                     else "QUALIFIED_FOR_MC"))
    qpath = os.path.join(outdir, "BASE_QUALIFICATION.json")
    json.dump(base_qual, open(qpath, "w"), indent=2)
    R["base_qualification"] = base_qual
    log(f"[P2] FIXED_BACKBONE_GRADED_REGION={gate['FIXED_BACKBONE_GRADED_REGION']} -> persisted {os.path.basename(qpath)}")
    snap()

    # ---- MC GATE: LOAD + VERIFY the persisted qualification artifact (control-flow invariant) ----
    ok, why = verify_qualification(qpath, grid_sha, cfg.config_sha256(),
                                   {mode: {str(bb["seed"]): bb["backbone_sha256"] for bb in R["backbones"][mode]}
                                    for mode in MODES})
    log(f"[GATE] qualification verify: ok={ok} ({why})")
    if not ok:
        R["outcomes"] = build_outcomes(cfg, R, blocked="QUALIFICATION_VERIFY_FAILED:" + why)
        finalize(R, cfg, outdir, t_start); logf.close(); return
    if base_qual["FIXED_BACKBONE_GRADED_REGION"] == "BLOCKED":
        log("[STOP] no common graded region on the frozen GDN backbones -> H3_TESTABILITY=BLOCKED_FIXED_BACKBONE; "
            "no MC, no EXT3 (Case A).")
        R["outcomes"] = build_outcomes(cfg, R, blocked="BLOCKED_FIXED_BACKBONE")
        finalize(R, cfg, outdir, t_start); logf.close(); return

    # ================= QUALIFIED: MC primary experiment (§10) =================
    log("[P3] MC primary experiment A/B/C on the FROZEN backbones (graded region QUALIFIED)")
    for mode in MODES:
        R["mc"][mode] = []
        R["curves"][mode] = {}
        for bb in R["backbones"][mode]:
            seed = bb["seed"]
            m = _reload_backbone(cfg, bb["checkpoint_path"], vocab, expect_sha=bb["backbone_sha256"])
            h_before = tensor_hashes(m)
            # C: train ONLY w_u reader on FROZEN backbone. build_model re-enabled grads on reload -> freeze ALL
            # params first, then unfreeze ONLY the reader connector (backbone mutation must be 0).
            for p in m.parameters():
                p.requires_grad_(False)
            m.blk.w_u.requires_grad_(True)
            rparams = [p for p in m.parameters() if p.requires_grad]
            rd_stat = train_backbone(cfg, m, pools["train"], cfg.steps_rd, mode, seed, log=log, reader="grm",
                                     params=rparams)
            changed = [n for n in h_before if n != "blk.w_u.weight" and h_before[n] != tensor_hashes(m)[n]]
            mutation = len(changed)
            # curves: A (base), B (param-free), C (reader) across all doses
            base_c = [base_curves[mode][seed][i] for i in range(len(doses))]
            pf_c = [eval_acc(cfg, m, pools["hold"], mode, "moving_average", d) for d in doses]
            rd_c = [eval_acc(cfg, m, pools["hold"], mode, "grm", d) for d in doses]
            R["curves"][mode][seed] = dict(dose=doses, base=base_c, paramfree=pf_c, reader=rd_c)
            m_base = retention_metrics(doses, base_c)
            m_pf = retention_metrics(doses, pf_c)
            m_rd = retention_metrics(doses, rd_c)
            # per-dose recovery/harm with counters + attribution (reader method)
            per_dose = []
            for d in doses:
                counters = dict(snapshotCandidates=0, snapshotReads=0, readerCalls=0)
                rec = eval_paired(cfg, m, pools["hold"], mode, d, "grm", capture=True, counters=counters)
                rec["path_counters"].update(counters)
                per_dose.append(rec)
            # durable reader save + SHA-256
            ck = os.path.join(art, f"ext2_{mode}_seed{seed}_reader.pt")
            torch.save({"state_dict": m.state_dict(), "mode": mode, "seed": seed, "vocab": vocab,
                        "reader_tensor": "blk.w_u.weight", "backbone_sha256": bb["backbone_sha256"],
                        "challengeGridSha256": grid_sha}, ck)
            reader_sha = _sha256_bytes(m.blk.w_u.weight.detach().cpu().numpy().tobytes())
            cell = dict(seed=seed, BACKBONE_WEIGHT_MUTATION=mutation, changed_tensors=changed,
                        FROZEN_BACKBONE_VALIDITY="PASS" if mutation == 0 else "FAIL",
                        AURC=dict(base=m_base["AURC_RETENTION"], paramfree=m_pf["AURC_RETENTION"],
                                  reader=m_rd["AURC_RETENTION"]),
                        DELTA_AURC_reader=round(m_rd["AURC_RETENTION"] - m_base["AURC_RETENTION"], 4),
                        DELTA_AURC_paramfree=round(m_pf["AURC_RETENTION"] - m_base["AURC_RETENTION"], 4),
                        D50=dict(base=m_base["D50"], reader=m_rd["D50"]),
                        DELTA_D50_reader=(round(m_rd["D50"] - m_base["D50"], 4)
                                          if (m_rd["D50"] is not None and m_base["D50"] is not None) else None),
                        D80_D20_WIDTH=dict(base=m_base["D80_D20_WIDTH"], reader=m_rd["D80_D20_WIDTH"]),
                        per_dose_recovery=per_dose, reader_checkpoint=ck, reader_weight_sha256=reader_sha,
                        reader_stat=rd_stat)
            R["mc"][mode].append(cell)
            log(f"  {mode} seed={seed}: mutation={mutation} AURC base={m_base['AURC_RETENTION']} "
                f"reader={m_rd['AURC_RETENTION']} DELTA_AURC={cell['DELTA_AURC_reader']}")
            del m
            torch.cuda.empty_cache() if DEV == "cuda" else None
            snap()

    # ---- P4 snapshot identity (§8) on one holdout example, GDN seed0, mid dose ----
    mid_dose = doses[len(doses) // 2]
    for mode in ("gdn", "dn", "la"):
        m = _reload_backbone(cfg, R["backbones"][mode][0]["checkpoint_path"], vocab,
                             expect_sha=R["backbones"][mode][0]["backbone_sha256"])
        R["snapshot_identity"][mode] = snapshot_identity(cfg, m, pools["hold"], mode, mid_dose, git_head)
        log(f"[P4] {mode} snapshot identity: {R['snapshot_identity'][mode]['SNAPSHOT_POSITION_IDENTITY']}")
        del m
        torch.cuda.empty_cache() if DEV == "cuda" else None
    snap()

    # ---- P5 target-aware ablation (§13) at mid dose, GDN + DN (reader loaded) ----
    log("[P5] target-aware ablation (FULL / DROP_TARGET_PROXIMAL / DROP_IRRELEVANT / DROP_RANDOM / SHAM)")
    for mode in ("gdn", "dn"):
        m = _reload_reader(cfg, R["mc"][mode][0]["reader_checkpoint"], vocab)
        R["ablation"][mode] = target_aware_ablation(cfg, m, pools["hold"], mode, mid_dose)
        log(f"  {mode}: signal={R['ablation'][mode]['HISTORICAL_SNAPSHOT_CAUSAL_SIGNAL']} "
            f"drop_prox={R['ablation'][mode]['drop_from_target_proximal']} "
            f"drop_irr={R['ablation'][mode]['drop_from_irrelevant']} drop_rnd={R['ablation'][mode]['drop_from_random']}")
        del m
        torch.cuda.empty_cache() if DEV == "cuda" else None
    snap()

    # ---- P6 hierarchical stats (§15) on GDN + DN DELTA_AURC ----
    for mode in ("gdn", "dn", "la"):
        curves = {seed: dict(base=R["curves"][mode][seed]["base"], mc=R["curves"][mode][seed]["reader"])
                  for seed in R["curves"][mode]}
        R["stats"][mode] = hierarchical_bootstrap_delta_aurc(curves, doses, cfg.boot_iters)
        log(f"[P6] {mode} DELTA_AURC per-seed={R['stats'][mode]['per_seed_DELTA_AURC']} "
            f"CI95={R['stats'][mode]['cluster_bootstrap_CI95']}")
    snap()

    # ---- P7 efficiency (§18-19) at mid dose ----
    for mode in MODES:
        m = _reload_reader(cfg, R["mc"][mode][0]["reader_checkpoint"], vocab)
        eff = efficiency(cfg, m, pools["hold"], mode, mid_dose)
        # cost-normalized (§18): use GDN/DN mean DELTA_AURC and mean recovery
        d_aurc = R["stats"][mode]["mean_DELTA_AURC"]
        add_bytes_kib = eff["bytes"]["historical_snapshots_at_full"] / 1024
        add_ms = round(eff["warm_ms"]["mc_total"] - eff["warm_ms"]["base_total"], 3)
        eff["derived"] = dict(DELTA_AURC=d_aurc,
                              DELTA_AURC_PER_KiB=round(d_aurc / add_bytes_kib, 6) if add_bytes_kib else None,
                              DELTA_AURC_PER_ADDED_ms=round(d_aurc / add_ms, 6) if add_ms > 0 else None,
                              added_snapshot_KiB=round(add_bytes_kib, 2), added_ms=add_ms)
        R["efficiency"][mode] = eff
        log(f"[P7] {mode}: mc_total_ms={eff['warm_ms']['mc_total']} base_total_ms={eff['warm_ms']['base_total']} "
            f"hist_KiB={round(add_bytes_kib,1)} DELTA_AURC/KiB={eff['derived']['DELTA_AURC_PER_KiB']}")
        del m
        torch.cuda.empty_cache() if DEV == "cuda" else None
    snap()

    R["outcomes"] = build_outcomes(cfg, R, blocked=None)
    finalize(R, cfg, outdir, t_start)
    logf.close()


# ==================================================================================================
# 14. Gate helpers, reload, outcomes, finalize
# ==================================================================================================
def graded_region_gate(cfg, gdn_curves, doses):
    """§7: a COMMON overlapping graded retention region across ALL GDN seeds (not one cell in a band)."""
    per_seed = {}
    mid_sets = []
    for seed, curve in gdn_curves.items():
        mx, mn = max(curve), min(curve)
        mid_doses = [doses[i] for i, a in enumerate(curve) if cfg.mid_lo <= a <= cfg.mid_hi]
        competent = mx >= cfg.grade_hi
        degrades = mn <= cfg.grade_lo
        resolved = len(mid_doses) >= cfg.min_mid_doses
        per_seed[seed] = dict(curve=curve, max=mx, min=mn, mid_doses=mid_doses,
                              competent=competent, degrades=degrades, resolved=resolved,
                              seed_graded=bool(competent and degrades and resolved))
        mid_sets.append(set(mid_doses))
    common = set.intersection(*mid_sets) if mid_sets else set()
    all_graded = all(v["seed_graded"] for v in per_seed.values())
    qualified = bool(all_graded and len(common) >= 1)
    return dict(per_seed=per_seed, common_mid_doses=sorted(common),
                ALL_SEEDS_GRADED=all_graded, COMMON_REGION_NONEMPTY=bool(common),
                thresholds=dict(grade_hi=cfg.grade_hi, grade_lo=cfg.grade_lo, mid_lo=cfg.mid_lo,
                                mid_hi=cfg.mid_hi, min_mid_doses=cfg.min_mid_doses),
                FIXED_BACKBONE_GRADED_REGION="QUALIFIED" if qualified else "BLOCKED")


def verify_qualification(qpath, grid_sha, config_sha, backbone_shas):
    if not os.path.exists(qpath):
        return False, "BASE_QUALIFICATION.json absent"
    q = json.load(open(qpath))
    if q.get("challengeGridSha256") != grid_sha:
        return False, "challengeGridSha256 mismatch"
    if q.get("configSha256") != config_sha:
        return False, "configSha256 mismatch"
    if q.get("backboneSha256") != backbone_shas:
        return False, "backboneSha256 mismatch"
    if "FIXED_BACKBONE_GRADED_REGION" not in q:
        return False, "verdict missing"
    return True, "verified"


def _reload_backbone(cfg, path, vocab, expect_sha=None):
    ck = torch.load(path, map_location=DEV)
    m = build_model(cfg, vocab, seed=0)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    if expect_sha is not None:
        got = backbone_sha256(m)
        assert got == expect_sha, f"backbone sha mismatch on reload: {got[:12]} != {expect_sha[:12]}"
    return m


def _reload_reader(cfg, path, vocab):
    ck = torch.load(path, map_location=DEV)
    m = build_model(cfg, vocab, seed=0)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def _classify_delta_aurc(cfg, ci, mean):
    lo, hi = ci
    s = cfg.sesoi_delta_aurc
    if lo >= s:
        return "MEANINGFUL_POSITIVE"
    if hi <= -s:
        return "REGRESSION"
    if -s <= lo and hi <= s:
        return "PRACTICALLY_EQUIVALENT"
    if hi > s and lo < s:
        return "INCONCLUSIVE_DIRECTIONAL"
    return "INCONCLUSIVE"


def build_outcomes(cfg, R, blocked=None):
    out = dict(packet="RNN-05B-EXT2", challengeGridSha256=R["meta"]["challengeGridSha256"],
               sourceGitHead=R["meta"]["sourceGitHead"], sesoi_delta_aurc=cfg.sesoi_delta_aurc,
               margin_OPERATOR_HEURISTIC=cfg.margin)
    if R.get("graded_gate"):
        out["FIXED_BACKBONE_GRADED_REGION"] = R["graded_gate"]["FIXED_BACKBONE_GRADED_REGION"]
        out["common_graded_doses"] = R["graded_gate"]["common_mid_doses"]
    else:
        out["FIXED_BACKBONE_GRADED_REGION"] = "NOT_COMPUTED"

    if blocked == "BLOCKED_FIXED_BACKBONE" or out["FIXED_BACKBONE_GRADED_REGION"] == "BLOCKED":
        out["H3_TESTABILITY"] = "BLOCKED_FIXED_BACKBONE"
        out["H3"] = "BLOCKED_FIXED_BACKBONE"
        out["QWEN_GDN_TRANSPLANT_GATE"] = "DEFER"
        out["SYNTHETIC_DENSE_MC"] = "PARK"
        out["decision_case"] = "A"
        out["note"] = "No common graded region on the frozen GDN backbones -> STOP; no MC; no EXT3."
        return out
    if blocked and blocked.startswith("QUALIFICATION_VERIFY_FAILED"):
        out["H3_TESTABILITY"] = "ABORTED_QUALIFICATION_VERIFY"
        out["H3"] = "ABORTED"
        out["QWEN_GDN_TRANSPLANT_GATE"] = "DEFER"
        out["decision_case"] = "ABORT"
        out["note"] = blocked
        return out

    out["H3_TESTABILITY"] = "QUALIFIED"
    # per-substrate summary
    per = {}
    for mode in MODES:
        cells = R["mc"].get(mode, [])
        if not cells:
            continue
        st = R["stats"].get(mode, {})
        mutation_ok = all(c["FROZEN_BACKBONE_VALIDITY"] == "PASS" for c in cells)
        # recovery vs harm pooled across seeds at the COMMON graded doses (denominators preserved)
        common = set(out.get("common_graded_doses") or [])
        rec_num = rec_den = harm_num = harm_den = 0
        for c in cells:
            for pd in c["per_dose_recovery"]:
                if not common or pd["dose"] in common:
                    rec_num += pd["n_recovered"]; rec_den += pd["n_base_wrong"]
                    harm_num += pd["n_harmed"]; harm_den += pd["n_base_correct"]
        per[mode] = dict(
            per_seed_DELTA_AURC=st.get("per_seed_DELTA_AURC"),
            mean_DELTA_AURC=st.get("mean_DELTA_AURC"),
            DELTA_AURC_CI95=st.get("cluster_bootstrap_CI95"),
            direction_all_positive=st.get("direction_all_positive"),
            FROZEN_BACKBONE_VALIDITY="PASS" if mutation_ok else "FAIL",
            pooled_recovery=dict(n_recovered=rec_num, n_base_wrong=rec_den,
                                 RECOVERY_RATE=round(rec_num / rec_den, 4) if rec_den else None,
                                 n_harmed=harm_num, n_base_correct=harm_den,
                                 HARM_RATE=round(harm_num / harm_den, 4) if harm_den else None,
                                 NET_RECOVERY_COUNT=rec_num - harm_num),
            DELTA_AURC_verdict=_classify_delta_aurc(cfg, st.get("cluster_bootstrap_CI95", [0, 0]),
                                                    st.get("mean_DELTA_AURC", 0.0)))
    out["per_substrate"] = per
    out["HISTORICAL_SNAPSHOT_CAUSAL_SIGNAL"] = R.get("ablation", {}).get("gdn", {}).get(
        "HISTORICAL_SNAPSHOT_CAUSAL_SIGNAL", "NOT_TESTED")
    la_delta = per.get("la", {}).get("mean_DELTA_AURC", 0.0) or 0.0
    la_verdict = per.get("la", {}).get("DELTA_AURC_verdict", "NA")
    out["LA_FALSIFICATION"] = ("CONTROL_HELD" if la_verdict in ("PRACTICALLY_EQUIVALENT", "REGRESSION", "INCONCLUSIVE")
                               and la_delta < cfg.sesoi_delta_aurc else "CONTROL_ALSO_POSITIVE_WEAKENS_SPECIFICITY")

    gdn = per.get("gdn", {})
    gdn_verdict = gdn.get("DELTA_AURC_verdict", "NA")
    prec = gdn.get("pooled_recovery", {})
    recovery_beats_harm = (prec.get("RECOVERY_RATE") is not None and prec.get("HARM_RATE") is not None
                           and prec["RECOVERY_RATE"] > prec["HARM_RATE"] + cfg.recovery_margin)
    strong = (gdn_verdict == "MEANINGFUL_POSITIVE" and gdn.get("direction_all_positive")
              and gdn.get("FROZEN_BACKBONE_VALIDITY") == "PASS" and recovery_beats_harm
              and out["HISTORICAL_SNAPSHOT_CAUSAL_SIGNAL"] == "SUPPORTED"
              and out["LA_FALSIFICATION"] == "CONTROL_HELD")
    if strong:
        out["H3"] = "POSITIVE_CANDIDATE"; out["QWEN_GDN_TRANSPLANT_GATE"] = "PASS_CANDIDATE"
        out["decision_case"] = "C"
        out["DENSE_POST_HOC_MEMORY_CACHING"] = "CANDIDATE"
        out["note"] = ("PASS_CANDIDATE authorizes ONLY the DESIGN of a separate real-Qwen qualification packet; "
                       "no automatic Qwen transplant or training.")
    elif gdn_verdict in ("PRACTICALLY_EQUIVALENT", "REGRESSION") or not recovery_beats_harm:
        out["H3"] = "NOT_DETECTED_IN_QUALIFIED_REGIME"; out["QWEN_GDN_TRANSPLANT_GATE"] = "DEFER"
        out["decision_case"] = "B"
        out["DENSE_POST_HOC_MEMORY_CACHING"] = "PARK"
        out["note"] = "Graded region qualified but MC DELTA_AURC not a defensible positive -> park; no EXT3."
    else:
        out["H3"] = "INCONCLUSIVE_DIRECTIONAL"; out["QWEN_GDN_TRANSPLANT_GATE"] = "DEFER"
        out["decision_case"] = "B"
        out["DENSE_POST_HOC_MEMORY_CACHING"] = "PARK"
        out["note"] = "MC effect directional but not clearly above SESOI or mechanism not fully supported -> defer."
    return out


def finalize(R, cfg, outdir, t_start):
    R["meta"]["wall_min"] = round((time.time() - t_start) / 60, 1)
    json.dump(R["outcomes"], open(os.path.join(outdir, "rnn05bext2_outcomes.json"), "w"), indent=2)
    json.dump(R, open(os.path.join(outdir, "rnn05bext2_results.json"), "w"), indent=2)
    _write_csv(R, os.path.join(outdir, "rnn05bext2_curves.csv"), cfg)
    print("[DONE] outcomes:\n" + json.dumps(R["outcomes"], indent=2))


def _write_csv(R, path, cfg):
    doses = list(cfg.doses())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["substrate", "seed", "method", *[f"dose_{d}" for d in doses], "AURC"])
        for mode in MODES:
            for seed, c in R.get("curves", {}).get(mode, {}).items():
                for method in ("base", "paramfree", "reader"):
                    acc = c[method]
                    aurc = retention_metrics(doses, acc)["AURC_RETENTION"]
                    w.writerow([mode, seed, method, *acc, aurc])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preregister", default=None, metavar="DIR")
    ap.add_argument("--selftest", default=None, metavar="JSON")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = Ext2Config()
    if a.preregister:
        m = write_preregistration(cfg, a.preregister)
        print(json.dumps(dict(wrote=a.preregister, challengeGridSha256=m["challengeGridSha256"],
                              grid_selfcheck=m["grid_selfcheck"]["GRID_IDENTITY_SELFCHECK"],
                              n_doses=len(cfg.doses())), indent=2))
    elif a.selftest is not None:
        generator_selftest(cfg, a.selftest)
    elif a.run:
        if not a.outdir:
            ap.error("--run requires --outdir")
        run(cfg, a.outdir, a.artifacts, a.smoke)
    else:
        ap.error("one of --preregister / --selftest / --run is required")
