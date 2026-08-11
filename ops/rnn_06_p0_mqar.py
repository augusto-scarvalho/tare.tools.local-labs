#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNN-06-P0 — Frozen-Checkpoint BASE Regime Scout (EXPLORATORY).

Bounded, inference-only MQAR-style associative-recall sweep on a FROZEN
pretrained recurrent LM. Answers ONE question:

  "On a frozen real pretrained recurrent LM, can we cheaply find a plausible
   non-ceiling, non-cliff memory-pressure regime suitable for later
   confirmatory qualification?"

This is P0. It is NOT RNN-06A (no lifecycle/checkpoint-restore), NOT RNN-06B
(no confirmatory qualification), and MUST NOT emit
FIXED_BACKBONE_GRADED_REGION = QUALIFIED. Its only band verdicts are:
  P0_GRADED_BAND = PLAUSIBLE | NOT_FOUND_WITHIN_BUDGET | MODEL_NOT_RUNNABLE

Design (token-id level, tokenizer-agnostic, deterministic):
  * Each key/value is exactly ONE token id (pools built per tokenizer), so
    every prompt at a given dose has identical length and the answer sits at a
    known position -> exact, low-variance constrained scoring, no padding.
  * Pressure knob = number of key/value pairs P (write-capacity / interference).
  * Nested-monotonic superset (EXT2 pattern): higher dose = same shared prefix
    of pairs + additional trailing distractors; the probed pair lives in the
    shared prefix at a fixed per-example index, so its write position is held
    constant while trailing interference (and the write->query gap) grows.
  * Evaluation = deterministic constrained argmax over the value vocabulary at
    the single answer position. Also records unconstrained top-1 (format
    adherence) and separates malformed/unparseable from memory failures.

Seeds: fixed integers via numpy PCG64 (process-stable, NOT python hash()).
Nothing here fine-tunes, modifies weights, restores/substitutes historical
state, or trains a probe/reader.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time

import numpy as np
import torch


# ----------------------------------------------------------------------------
# determinism helpers (process-stable integer-seed RNG; NOT hash()-based)
# ----------------------------------------------------------------------------
def rng_for(*ints):
    """Deterministic numpy Generator from a tuple of ints via SeedSequence."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(ints))))


def sha256_of_obj(obj):
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_of_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ----------------------------------------------------------------------------
# token pools (single-token keys + values), built per tokenizer
# ----------------------------------------------------------------------------
def build_token_pools(tokenizer, pool_size, seed):
    """Return (key_ids, value_ids, seps, meta) all single-token, keys/values DISJOINT.

    value tokens : prefer bare integers (2-4 digits, opt. leading space); if a
                   tokenizer (e.g. Llama/Mistral SentencePiece splits digits) has
                   too few, fall back to alphanumeric single-tokens.
    key tokens   : alphabetic word-tokens (opt. leading space, len>=3).
    sep          : "=" glue token and "\n" newline token (both single-token).
    Selection is deterministic (vocab order, then a fixed-seed permutation).
    Keys and values are drawn from disjoint id sets so no key == value.
    """
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    num_re = re.compile(r"^\s?\d{2,4}$")
    word_re = re.compile(r"^\s?[A-Za-z]{3,}$")
    alnum_re = re.compile(r"^\s?[A-Za-z0-9]{2,}$")

    num_ids, word_ids, alnum_ids = [], [], []
    for tid in range(vocab_size):
        try:
            s = tokenizer.decode([tid])
        except Exception:
            continue
        if num_re.match(s):
            num_ids.append(tid)
        if word_re.match(s):
            word_ids.append(tid)
        if alnum_re.match(s):
            alnum_ids.append(tid)

    def _filter_single(ids):
        seen, out = set(), []
        for tid in ids:
            s = tokenizer.decode([tid])
            enc = tokenizer.encode(s, add_special_tokens=False)
            if len(enc) == 1 and enc[0] == tid and s not in seen:
                seen.add(s)
                out.append(tid)
        return out

    num_ids = _filter_single(num_ids)
    word_ids = _filter_single(word_ids)
    alnum_ids = _filter_single(alnum_ids)

    def _perm(ids, salt):
        if not ids:
            return []
        r = rng_for(seed, salt)
        return [int(x) for x in np.array(ids)[r.permutation(len(ids))]]

    word_ids = _perm(word_ids, 0xB0B)
    # keys: alphabetic words
    key_ids = word_ids[:pool_size]
    key_set = set(key_ids)

    # values: prefer numeric; else alphanumeric disjoint from keys
    if len(num_ids) >= pool_size:
        value_ids = _perm(num_ids, 0xA11CE)[:pool_size]
        value_kind = "numeric_2to4_digit_single_token"
    else:
        alnum_perm = _perm(alnum_ids, 0x5A17)
        value_ids = [t for t in alnum_perm if t not in key_set][:pool_size]
        value_kind = ("alphanumeric_single_token_fallback (few multi-digit "
                      f"single-token numbers: only {len(num_ids)})")
    value_set = set(value_ids)
    # ensure disjoint
    key_ids = [t for t in key_ids if t not in value_set][:pool_size]

    def _one(tok):
        enc = tokenizer.encode(tok, add_special_tokens=False)
        return enc[-1] if enc else None

    meta = {
        "value_kind": value_kind,
        "n_numeric_single_token": len(num_ids),
        "n_word_single_token": len(word_ids),
        "n_alnum_single_token": len(alnum_ids),
    }
    return key_ids, value_ids, {"eq": _one("="), "nl": _one("\n")}, meta


# ----------------------------------------------------------------------------
# abstract calibration spec (tokenizer-independent): indices + probe position
# ----------------------------------------------------------------------------
def build_calibration_spec(master_seed, n_examples, p_max, p_min, pool_size):
    """Per-example: ordered key/value SLOT lists (perm prefix) + probe index.

    Nesting: dose P uses the first P (key_slot,value_slot). probe_index<p_min so
    the probed pair is inside every dose's shared prefix.
    """
    examples = []
    for i in range(n_examples):
        r = rng_for(master_seed, 0x5EED, i)
        key_slots = r.permutation(pool_size)[:p_max].tolist()
        val_slots = r.permutation(pool_size)[:p_max].tolist()
        probe_index = int(r.integers(0, p_min))
        examples.append({"key_slots": key_slots, "val_slots": val_slots,
                         "probe_index": probe_index})
    spec = {
        "kind": "mqar_assoc_recall_nested_monotonic_v1",
        "master_seed": master_seed,
        "n_examples": n_examples,
        "p_max": p_max,
        "p_min": p_min,
        "pool_size": pool_size,
        "examples": examples,
    }
    return spec


# ----------------------------------------------------------------------------
# materialize one dose to flat token-id prompts (all equal length)
# ----------------------------------------------------------------------------
def materialize_dose(spec, P, key_ids, value_ids, seps):
    eq, nl = seps["eq"], seps["nl"]
    prompts, golds = [], []
    for ex in spec["examples"]:
        ks = ex["key_slots"][:P]
        vs = ex["val_slots"][:P]
        toks = []
        for k, v in zip(ks, vs):
            toks += [key_ids[k], eq, value_ids[v], nl]
        pk = ex["key_slots"][ex["probe_index"]]
        pv = ex["val_slots"][ex["probe_index"]]
        toks += [key_ids[pk], eq]          # query: "<probe_key> ="
        prompts.append(toks)
        golds.append(value_ids[pv])        # gold value token id
    return prompts, golds


# ----------------------------------------------------------------------------
# forward + constrained scoring (deterministic, single forward per prompt)
# ----------------------------------------------------------------------------
@torch.no_grad()
def eval_dose(model, prompts, golds, value_id_set, device, batch_size):
    """Return per-dose metrics. No sampling; pure logits.

    constrained_correct : argmax over the value vocabulary == gold
    unconstrained_top1   : global argmax (any token)
    format_ok            : unconstrained top1 is inside the value vocabulary
    """
    value_ids = torch.tensor(sorted(value_id_set), device=device, dtype=torch.long)
    n = len(prompts)
    n_constrained_correct = 0
    n_unconstrained_correct = 0
    n_format_ok = 0
    for b in range(0, n, batch_size):
        chunk = prompts[b:b + batch_size]
        gold_chunk = golds[b:b + batch_size]
        ids = torch.tensor(chunk, device=device, dtype=torch.long)  # equal length
        out = model(ids)
        logits = out.logits if hasattr(out, "logits") else out[0]
        last = logits[:, -1, :].float()                             # [B, V]
        # unconstrained
        uncon = last.argmax(dim=-1)                                 # [B]
        # constrained to value vocabulary
        sub = last.index_select(1, value_ids)                       # [B, |Vvocab|]
        con_local = sub.argmax(dim=-1)                              # [B]
        con = value_ids[con_local]                                  # [B]
        for j in range(len(chunk)):
            g = int(gold_chunk[j])
            if int(con[j]) == g:
                n_constrained_correct += 1
            if int(uncon[j]) == g:
                n_unconstrained_correct += 1
            if int(uncon[j]) in value_id_set:
                n_format_ok += 1
    return {
        "n": n,
        "constrained_acc": n_constrained_correct / n,
        "unconstrained_exact_acc": n_unconstrained_correct / n,
        "format_adherence": n_format_ok / n,
        "n_constrained_correct": n_constrained_correct,
        "n_unconstrained_correct": n_unconstrained_correct,
        "n_format_ok": n_format_ok,
        "chance": 1.0 / len(value_id_set),
        "value_vocab_size": len(value_id_set),
    }


# ----------------------------------------------------------------------------
# model loading + identity
# ----------------------------------------------------------------------------
def load_model(model_id, revision, dtype, device, config_overrides=None):
    # importing fla registers gated_deltanet / delta_net / gla / rwkv7 with
    # AutoConfig/AutoModel; harmless (and skipped gracefully) for mamba2.
    try:
        import fla  # noqa: F401
    except Exception:
        pass
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    applied = {}
    # Some fla checkpoints under-specify intermediate_size (config sets it None
    # and the installed fla recomputes a value that mismatches the trained
    # weights). Overrides here reconstruct the architecture the *weights* define;
    # they are recorded in MODEL_IDENTITY, not silent.
    for k, v in (config_overrides or {}).items():
        applied[k] = {"was": getattr(cfg, k, None), "now": v}
        setattr(cfg, k, v)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, config=cfg, trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()
    return model, tok, cfg, applied


def resolve_revision(model_id):
    from huggingface_hub import HfApi
    try:
        info = HfApi().model_info(model_id)
        return info.sha
    except Exception:
        return None


def file_hashes(model_id):
    """Return {filename: {size, sha256}} for config + small json files from cache."""
    from huggingface_hub import try_to_load_from_cache
    out = {}
    for fn in ["config.json", "tokenizer_config.json", "tokenizer.json",
               "generation_config.json", "special_tokens_map.json"]:
        p = try_to_load_from_cache(model_id, fn)
        if isinstance(p, str) and os.path.isfile(p):
            with open(p, "rb") as f:
                data = f.read()
            out[fn] = {"size": len(data), "sha256": sha256_of_bytes(data)}
    return out


def env_versions():
    import transformers
    ver = {"python": sys.version.split()[0], "platform": platform.platform(),
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "transformers": transformers.__version__}
    for mod in ["fla", "flash_linear_attention", "triton", "mamba_ssm",
                "causal_conv1d", "accelerate", "numpy", "huggingface_hub"]:
        try:
            m = __import__(mod)
            ver[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            ver[mod] = None
    if torch.cuda.is_available():
        ver["gpu"] = torch.cuda.get_device_name(0)
        ver["gpu_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return ver


# ----------------------------------------------------------------------------
# main sweep
# ----------------------------------------------------------------------------
def run_candidate(args, dtype, device):
    tag = args.tag
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    t_all0 = time.time()

    record = {
        "packet": "RNN-06-P0",
        "candidate_tag": tag,
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "resolved_revision": None,
        "status": {},
        "env": env_versions(),
        "config": {
            "dose_ladder_pairs": args.doses,
            "n_eval": args.n_eval,
            "master_seed": args.master_seed,
            "pool_size": args.pool_size,
            "p_min": min(args.doses),
            "p_max": max(args.doses),
            "dtype": str(dtype),
            "batch_size": args.batch_size,
            "competence_tau_hi": args.tau_hi,
            "low_dose": min(args.doses),
        },
        "notes": [],
    }

    # resolve revision (best effort, before load)
    resolved = resolve_revision(args.model_id)
    record["resolved_revision"] = resolved

    # -- load ---------------------------------------------------------------
    overrides = json.loads(args.config_overrides) if args.config_overrides else {}
    record["config"]["config_overrides_requested"] = overrides
    t0 = time.time()
    try:
        model, tok, cfg, applied = load_model(args.model_id, args.revision, dtype,
                                              device, overrides)
    except Exception as e:
        record["status"]["MODEL_RUNNABLE"] = "NO"
        record["status"]["P0_GRADED_BAND"] = "MODEL_NOT_RUNNABLE"
        record["load_error"] = f"{type(e).__name__}: {e}"
        record["load_traceback"] = _tb()
        with open(os.path.join(outdir, f"P0_RESULTS_{tag}.json"), "w") as f:
            json.dump(record, f, indent=2)
        print(f"[{tag}] MODEL_NOT_RUNNABLE: {e}", file=sys.stderr)
        return record
    load_secs = time.time() - t0
    record["status"]["MODEL_RUNNABLE"] = "YES"
    record["load_seconds"] = round(load_secs, 2)
    record["config_overrides_applied"] = applied
    record["env"].update(env_versions())  # refresh now that fla/mamba imported

    n_params = sum(p.numel() for p in model.parameters())
    record["n_params"] = int(n_params)

    # file hashes (now that cache is populated)
    record["file_hashes"] = file_hashes(args.model_id)
    record["tokenizer_identity"] = {
        "name_or_path": getattr(tok, "name_or_path", None),
        "vocab_size": int(getattr(tok, "vocab_size", 0) or len(tok)),
        "class": type(tok).__name__,
    }
    # freeze machine-readable model identity artifact (before substantive sweep)
    identity = {
        "packet": "RNN-06-P0",
        "candidate_tag": tag,
        "model_id": args.model_id,
        "resolved_revision": record["resolved_revision"],
        "model_class": type(model).__name__,
        "model_type": getattr(cfg, "model_type", None),
        "n_params": int(n_params),
        "dtype": str(dtype),
        "quantization": "NONE",
        "config_overrides_applied": applied,
        "config_public": {k: v for k, v in cfg.to_dict().items()
                          if isinstance(v, (int, float, str, bool, type(None)))},
        "file_hashes": record["file_hashes"],
        "tokenizer_identity": record["tokenizer_identity"],
        "backend_env": record["env"],
        "frozen_assertion": "same frozen checkpoint used across ALL pressure conditions; no per-condition model/seed variation",
    }
    with open(os.path.join(outdir, f"MODEL_IDENTITY_{tag}.json"), "w") as f:
        json.dump(identity, f, indent=2)

    # -- token pools --------------------------------------------------------
    key_ids, value_ids, seps, pool_meta = build_token_pools(
        tok, args.pool_size, args.master_seed)
    record["pools"] = {
        "n_key_ids": len(key_ids),
        "n_value_ids": len(value_ids),
        "eq_id": seps["eq"],
        "nl_id": seps["nl"],
        "eq_str": tok.decode([seps["eq"]]) if seps["eq"] is not None else None,
        "sample_keys": [tok.decode([i]) for i in key_ids[:8]],
        "sample_values": [tok.decode([i]) for i in value_ids[:8]],
        "value_ids_sha256": sha256_of_obj(sorted(value_ids)),
        "key_ids_sha256": sha256_of_obj(sorted(key_ids)),
        **pool_meta,
    }
    p_max = max(args.doses)
    if len(key_ids) < p_max or len(value_ids) < p_max or seps["eq"] is None or seps["nl"] is None:
        record["status"]["TASK_COMPETENT"] = "N/A"
        record["status"]["P0_GRADED_BAND"] = "MODEL_NOT_RUNNABLE"
        record["notes"].append(
            f"insufficient single-token pools: keys={len(key_ids)} values={len(value_ids)} "
            f"eq={seps['eq']} nl={seps['nl']} (need >= p_max={p_max})")
        with open(os.path.join(outdir, f"P0_RESULTS_{tag}.json"), "w") as f:
            json.dump(record, f, indent=2)
        return record

    # -- calibration spec (abstract, model-independent) ---------------------
    spec = build_calibration_spec(args.master_seed, args.n_eval, p_max,
                                  min(args.doses), args.pool_size)
    abstract_sha = sha256_of_obj({k: spec[k] for k in
                                  ["kind", "master_seed", "n_examples", "p_max",
                                   "p_min", "pool_size", "examples"]})
    record["calibrationSetSha256"] = abstract_sha
    # save abstract spec once (shared across candidates) + a readable sample
    spec_path = os.path.join(outdir, "calibration_examples.json")
    if not os.path.isfile(spec_path):
        with open(spec_path, "w") as f:
            json.dump({"abstract_spec_sha256": abstract_sha, **spec}, f)

    value_id_set = set(value_ids)

    # -- sweep --------------------------------------------------------------
    curves = []
    per_dose_prompt_sha = {}
    peak_vram = 0
    n_forward_examples = 0
    for P in args.doses:
        prompts, golds = materialize_dose(spec, P, key_ids, value_ids, seps)
        per_dose_prompt_sha[str(P)] = sha256_of_obj(prompts)
        seq_len = len(prompts[0])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        td0 = time.time()
        m = eval_dose(model, prompts, golds, value_id_set, device, args.batch_size)
        dt = time.time() - td0
        n_forward_examples += m["n"]
        if torch.cuda.is_available():
            peak_vram = max(peak_vram, torch.cuda.max_memory_allocated())
        row = {"pairs": P, "seq_len_tokens": seq_len, "eval_seconds": round(dt, 2), **m}
        curves.append(row)
        print(f"[{tag}] P={P:4d} len={seq_len:5d} "
              f"con_acc={m['constrained_acc']:.3f} fmt={m['format_adherence']:.3f} "
              f"uncon={m['unconstrained_exact_acc']:.3f} chance={m['chance']:.4f} "
              f"({dt:.1f}s)", file=sys.stderr)

    record["curves"] = curves
    record["per_dose_prompt_sha256"] = per_dose_prompt_sha
    record["materialized_sweep_sha256"] = sha256_of_obj(per_dose_prompt_sha)
    record["peak_vram_gb"] = round(peak_vram / 1e9, 3)
    record["n_forward_examples"] = n_forward_examples
    record["total_seconds"] = round(time.time() - t_all0, 2)

    # -- verdicts -----------------------------------------------------------
    low_acc = curves[0]["constrained_acc"]
    competent = low_acc >= args.tau_hi
    record["status"]["TASK_COMPETENT"] = "YES" if competent else "NO"
    record["status"]["low_dose_constrained_acc"] = low_acc

    band = classify_band(curves, args.tau_hi, args.tau_lo, competent)
    record["status"]["P0_GRADED_BAND"] = band["verdict"]
    record["band_analysis"] = band

    with open(os.path.join(outdir, f"P0_RESULTS_{tag}.json"), "w") as f:
        json.dump(record, f, indent=2)
    return record


def classify_band(curves, tau_hi, tau_lo, competent):
    """Exploratory band classifier. PLAUSIBLE only if competent-high at low
    pressure AND a material-but-non-total drop appears across >=1 interior
    transition without being a single-cell cliff to the floor.
    """
    accs = [c["constrained_acc"] for c in curves]
    pairs = [c["pairs"] for c in curves]
    hi = accs[0]
    lo = min(accs)
    lo_at = pairs[accs.index(lo)]
    analysis = {
        "low_dose_acc": hi, "min_acc": lo, "min_acc_at_pairs": lo_at,
        "monotone_nonincreasing": all(accs[i] >= accs[i + 1] - 0.02
                                      for i in range(len(accs) - 1)),
        "n_mid_band_doses": sum(1 for a in accs if tau_lo < a < tau_hi),
        "tau_hi": tau_hi, "tau_lo": tau_lo,
    }
    if not competent:
        analysis["verdict"] = "NOT_FOUND_WITHIN_BUDGET"
        analysis["reason"] = "TASK_NOT_COMPETENT at low pressure (see status)"
        return analysis
    # flat-high?
    if lo >= tau_hi:
        analysis["verdict"] = "NOT_FOUND_WITHIN_BUDGET"
        analysis["reason"] = f"flat-high: min acc {lo:.3f} never dropped below tau_hi={tau_hi}"
        return analysis
    # find first index that drops below tau_hi, and whether there is a mid rung
    drop_idx = next((i for i, a in enumerate(accs) if a < tau_hi), None)
    # cliff = jump from >=tau_hi straight to <=tau_lo in one rung with no mid rung
    mid = analysis["n_mid_band_doses"]
    if mid >= 1:
        analysis["verdict"] = "PLAUSIBLE"
        analysis["reason"] = (f"{mid} interior dose(s) in ({tau_lo},{tau_hi}); "
                              f"competent-high {hi:.3f} -> material loss {lo:.3f}")
    else:
        analysis["verdict"] = "NOT_FOUND_WITHIN_BUDGET"
        analysis["reason"] = ("no interior mid-band dose (possible single-cell "
                              "cliff or too-coarse ladder); finer sweep needed")
    return analysis


def _tb():
    import traceback
    return traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--doses", type=int, nargs="+",
                    default=[4, 8, 16, 32, 64, 128])
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--master-seed", type=int, default=20260811)
    ap.add_argument("--pool-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tau-hi", type=float, default=0.75)
    ap.add_argument("--tau-lo", type=float, default=0.45)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--config-overrides", default="",
                    help="JSON dict of AutoConfig attribute overrides (recorded in identity)")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    rec = run_candidate(args, dtype, device)
    print(json.dumps({"tag": args.tag,
                      "status": rec["status"]}, indent=2))


if __name__ == "__main__":
    main()
