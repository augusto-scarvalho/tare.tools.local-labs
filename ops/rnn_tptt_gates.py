#!/usr/bin/env python
"""
RNN-08b pre-run gates (§5,§9,§11). ALL must PASS before training, else STOP.
  INDEPENDENT_SAMPLE_ISOLATION  — B-alone == A->B through the lifecycle wrapper (BIT_EXACT)
  INTRA_SEQUENCE_RECURRENCE     — state accumulates across internal chunks; reset clears it
  TPTT_SAVE_RELOAD              — save->fresh-load->deterministic logits equivalence
  TRAINING_RESET                — per-batch reset clears state; grads still flow
"""
import argparse, json, os
import torch
from transformers import AutoTokenizer, AutoConfig
import tptt
from rnn_tptt_lifecycle import build_tptt, IndependentSequenceTPTT, MODEL

DEV, DT = "cuda", torch.float32


def maxdiff(a, b):
    return float((a - b).abs().max())


def verdict(d):
    if d == 0.0:
        return "BIT_EXACT"
    if d <= 1e-4:
        return f"NUMERICALLY_EQUIVALENT_WITHIN_TOLERANCE(<=1e-4, {d:.2e})"
    return f"DIFFERENT({d:.2e})"


def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(MODEL)
    A = tok("Alpha alpha unrelated context number 111.", return_tensors="pt").to(DEV)
    B = tok("What is two plus two?", return_tensors="pt").to(DEV)
    fixed = tok("Deterministic gate prompt for save reload.", return_tensors="pt").to(DEV)
    R = {}

    m, lc = build_tptt(cfg, DEV, DT)
    iso = IndependentSequenceTPTT(lc)

    # ---- ISOLATION (with wrapper) + leak sanity (without) ----
    m.eval()
    with torch.no_grad():
        iso.reset(); lgB0 = m(**B).logits
        iso.reset(); _ = m(**A)
        iso.reset(); lgB1 = m(**B).logits
        iso_diff = maxdiff(lgB0, lgB1)
        # sanity: WITHOUT reset it should leak (reproduce 08)
        lc.reset(); lgN0 = m(**B).logits
        _ = m(**A)
        lgN1 = m(**B).logits
        leak_diff = maxdiff(lgN0, lgN1)
    R["INDEPENDENT_SAMPLE_ISOLATION"] = dict(
        with_wrapper_max_abs_diff=iso_diff, without_wrapper_leak_diff=leak_diff,
        status="PASS" if iso_diff == 0.0 else "FAIL", verdict=verdict(iso_diff))

    # ---- INTRA-SEQUENCE RECURRENCE (state accumulates across chunks; reset clears) ----
    long_ids = torch.randint(0, cfg.vocab_size, (1, 128), generator=torch.Generator().manual_seed(3)).to(DEV)
    with torch.no_grad():
        iso.reset()
        rep_empty = iso.state_report()               # after reset, before forward
        iso.reset(); _ = m(input_ids=long_ids[:, :32])
        rep32 = iso.state_report()                   # 1 chunk (32 = max_chunk)
        iso.reset(); _ = m(input_ids=long_ids[:, :128])
        rep128 = iso.state_report()                  # 4 chunks
        iso.reset()
        rep_reset = iso.state_report()               # cleared again
    active = (rep128["l2_norm"] > 0 and rep32["l2_norm"] > 0
              and abs(rep128["l2_norm"] - rep32["l2_norm"]) > 1e-6)
    clears = (rep_reset["l2_norm"] <= rep128["l2_norm"] * 1e-3 + 1e-6)
    R["INTRA_SEQUENCE_RECURRENCE"] = dict(
        state_after_reset=rep_empty, state_32tok=rep32, state_128tok=rep128,
        state_after_final_reset=rep_reset,
        accumulates_across_chunks=bool(active), reset_clears=bool(clears),
        status="PASS" if (active and clears) else "FAIL")
    # state-size evidence (§16)
    R["TPTT_STATE_SIZE"] = dict(total_state_bytes=rep128["total_bytes"],
                                n_state_tensors=rep128["n_tensors"],
                                sample_shapes=rep128["sample_shapes"],
                                note="TPTT recurrent LCache state, separate from full-attn KV cache")

    # ---- SAVE / RELOAD (perturb LoRA first so it is non-trivial) ----
    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=5e-3)
    batch = tok(["The capital of France is Paris and it is lovely."], return_tensors="pt").to(DEV)
    for _ in range(3):
        iso.reset()
        loss = m(input_ids=batch["input_ids"], labels=batch["input_ids"]).loss
        loss.backward(); opt.step(); opt.zero_grad()
    m.eval()
    with torch.no_grad():
        iso.reset(); lg_before = m(**fixed).logits
    sdir = os.path.join(args.outdir, "gate_tptt_adapter")
    os.makedirs(sdir, exist_ok=True)
    save_ok, save_err = True, None
    try:
        tptt.save_tptt_safetensors(m, sdir, "adapter_model.safetensors")
    except Exception as e:
        save_ok, save_err = False, repr(e)[:200]
    reload_diff = None
    try:
        m2, lc2 = build_tptt(cfg, DEV, DT)
        m2 = tptt.load_tptt_safetensors(sdir, m2)
        iso2 = IndependentSequenceTPTT(lc2)
        m2.eval()
        with torch.no_grad():
            iso2.reset(); lg_after = m2(**fixed).logits
        reload_diff = maxdiff(lg_before, lg_after)
        del m2
    except Exception as e:
        save_err = (save_err or "") + " | reload:" + repr(e)[:200]
    torch.cuda.empty_cache()
    sr_status = "QUALIFIED" if (save_ok and reload_diff is not None and reload_diff <= 1e-4) else "FAILED"
    R["TPTT_SAVE_RELOAD"] = dict(save_ok=save_ok, reload_max_abs_diff=reload_diff,
                                 verdict=verdict(reload_diff) if reload_diff is not None else "N/A",
                                 status=sr_status, err=save_err,
                                 adapter_dir=os.path.basename(sdir))

    # ---- TRAINING-RESET probe (§11): train-mode isolation via LOSS (robust to whether the
    # checkpointed train path stores cache state); + reset must not disable gradients. ----
    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    b1 = tok(["First independent training batch about oceans."], return_tensors="pt").to(DEV)
    # (i) grad-flow AFTER a reset -> reset does not detach/disable the current graph's grads
    opt.zero_grad(); iso.reset()
    lg = m(input_ids=b1["input_ids"], labels=b1["input_ids"]).loss
    lg.backward()
    grads = sum(1 for n, p in m.named_parameters()
                if p.requires_grad and p.grad is not None and "lora" in n.lower())
    opt.zero_grad()
    # (ii) train-mode contamination test at fixed weights (loss of B must not depend on A)
    with torch.no_grad():
        lB0 = float(m(input_ids=B["input_ids"], labels=B["input_ids"]).loss)
        _ = m(**A)
        lB1 = float(m(input_ids=B["input_ids"], labels=B["input_ids"]).loss)
        leak_train = abs(lB1 - lB0)
        iso.reset(); lB0r = float(m(input_ids=B["input_ids"], labels=B["input_ids"]).loss)
        iso.reset(); _ = m(**A)
        iso.reset(); lB1r = float(m(input_ids=B["input_ids"], labels=B["input_ids"]).loss)
        iso_train = abs(lB1r - lB0r)
    R["TRAINING_RESET"] = dict(
        train_mode_leak_without_reset=round(leak_train, 8),
        train_mode_isolation_with_wrapper=round(iso_train, 8),
        lora_grads_after_reset=grads, grad_flow_after_reset=bool(grads > 0),
        note=("Under use_linear_checkpoint the train forward does not retain LCache state, so "
              "train-mode forwards are already isolated; the wrapper reset is an explicit no-op "
              "safeguard and does not disable gradients."),
        status="PASS" if (grads > 0 and iso_train == 0.0) else "FAIL")

    gates = {k: R[k].get("status") for k in ["INDEPENDENT_SAMPLE_ISOLATION",
             "INTRA_SEQUENCE_RECURRENCE", "TPTT_SAVE_RELOAD", "TRAINING_RESET"]}
    allpass = all(v in ("PASS", "QUALIFIED") for v in gates.values())
    R["GATES_SUMMARY"] = dict(gates=gates, ALL_PASS=allpass)
    with open(os.path.join(args.outdir, "rnn08b_gates.json"), "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps({"GATES_SUMMARY": R["GATES_SUMMARY"],
                      "isolation": R["INDEPENDENT_SAMPLE_ISOLATION"],
                      "recurrence": {k: R["INTRA_SEQUENCE_RECURRENCE"][k] for k in
                                     ["state_32tok", "state_128tok", "accumulates_across_chunks",
                                      "reset_clears", "status"]},
                      "save_reload": R["TPTT_SAVE_RELOAD"],
                      "training_reset": R["TRAINING_RESET"]}, indent=2))
    return 0 if allpass else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    raise SystemExit(run(ap.parse_args()))
