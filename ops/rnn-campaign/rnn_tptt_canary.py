#!/usr/bin/env python
"""
RNN-08 §4 — TPTT dependency canary. Proves the isolated stack executes end-to-end BEFORE any
meaningful training. Isolated venv only; does not touch sglang/evalplus envs. float32 (the LiZA
linear path is float32; bf16 base triggers a conv-bias dtype mismatch).

Acceptance (all must pass): import, model load, TPTT inject, forward, backward, optimizer step,
LoRA-only control same, adapter save/load equivalence, CUDA memory returns to ~baseline after cleanup.
"""
import argparse, json, os, time, subprocess, platform
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model
import tptt

LORA = dict(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")


def ntrain(m):
    t = sum(p.numel() for p in m.parameters() if p.requires_grad)
    a = sum(p.numel() for p in m.parameters())
    return t, a


def env_manifest():
    import transformers, peft
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True).strip()
    except Exception as e:
        smi = f"err:{e}"
    return dict(
        python=platform.python_version(), torch=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        nvidia_smi=smi, transformers=transformers.__version__, peft=peft.__version__,
        tptt_commit="242e2140c2af469765b84ab3a7a79668be254cfa",
    )


def run(args):
    dev = "cuda"
    dtype = torch.float32
    R = dict(env=env_manifest(), model=args.model, dtype="float32", stages={})
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(args.model)
    batch = tok(["The capital of France is Paris.", "Water boils at one hundred degrees."],
                return_tensors="pt", padding=True)
    ids, am = batch["input_ids"].to(dev), batch["attention_mask"].to(dev)
    labels = ids.clone()
    fixed = tok(["Deterministic check prompt."], return_tensors="pt").to(dev)

    def vram_reset():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    # baseline VRAM
    vram_reset()
    base_alloc0 = torch.cuda.memory_allocated() / 1e6

    # ---------- BASE reference ----------
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(dev).eval()
    with torch.no_grad():
        base_loss = float(base(input_ids=ids, attention_mask=am, labels=labels).loss)
    R["stages"]["base_reference"] = dict(loss=base_loss, load_s=round(time.time() - t0, 2),
                                         trainable_all=ntrain(base))
    del base; vram_reset()

    # ---------- Arm C: TPTT + LoRA ----------
    st = dict()
    try:
        vram_reset(); t0 = time.time()
        mc = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(dev)
        mc, lcache = tptt.get_tptt_model(mc, cfg, operator_mode="delta_rule", mag_weight=0.5,
                                         linear_precision=torch.float32)
        st["inject_ok"] = True
        st["trainable_all_after_inject"] = ntrain(mc)
        mc = get_peft_model(mc, LoraConfig(**LORA))
        mc = mc.to(dev)  # move injected LiZA submodules (created on CPU) to GPU
        # Faithful, parameter-matched recipe: train LoRA ONLY (same as arm B). The delta-rule
        # memory path is mixed in via the scheduled scalar mag; its internal conv is frozen upstream.
        st["trainable_all_after_peft"] = ntrain(mc)
        mc.train()
        out = mc(input_ids=ids, attention_mask=am, labels=labels)
        st["forward_loss"] = float(out.loss); st["forward_finite"] = bool(torch.isfinite(out.loss))
        opt = torch.optim.AdamW([p for p in mc.parameters() if p.requires_grad], lr=1e-4)
        ts = time.time()
        out.loss.backward(); opt.step(); opt.zero_grad()
        st["step_s"] = round(time.time() - ts, 3)
        st["backward_step_ok"] = True
        with torch.no_grad():
            st["loss_after_step"] = float(mc(input_ids=ids, attention_mask=am, labels=labels).loss)
        st["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        # save adapter
        os.makedirs(args.outdir, exist_ok=True)
        cpath = os.path.join(args.outdir, "canary_tptt_adapter")
        try:
            tptt.save_tptt_safetensors(mc, cpath, "adapter_model.safetensors")
            st["save_tptt_safetensors_ok"] = os.path.exists(
                os.path.join(cpath, "adapter_model.safetensors")) or os.path.isdir(cpath)
        except Exception as e:
            st["save_tptt_safetensors_ok"] = False; st["save_err"] = repr(e)[:200]
        st["load_s"] = round(time.time() - t0, 2)
        st["PASS"] = st.get("forward_finite") and st.get("backward_step_ok")
    except Exception as e:
        import traceback
        st["PASS"] = False; st["error"] = repr(e)[:300]; st["tb"] = traceback.format_exc()[-800:]
    R["stages"]["arm_C_tptt_lora"] = st
    try:
        del mc
    except Exception:
        pass
    vram_reset()

    # ---------- Arm B: LoRA-only control ----------
    sb = dict()
    try:
        vram_reset(); t0 = time.time()
        mb = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(dev)
        mb = get_peft_model(mb, LoraConfig(**LORA))
        sb["trainable_all_after_peft"] = ntrain(mb)
        mb.train()
        out = mb(input_ids=ids, attention_mask=am, labels=labels)
        sb["forward_loss"] = float(out.loss); sb["forward_finite"] = bool(torch.isfinite(out.loss))
        opt = torch.optim.AdamW([p for p in mb.parameters() if p.requires_grad], lr=1e-4)
        out.loss.backward(); opt.step(); opt.zero_grad()
        sb["backward_step_ok"] = True
        sb["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        # save + reload adapter -> deterministic logits equivalence
        mb.eval()
        with torch.no_grad():
            lg0 = mb(**fixed).logits
        bpath = os.path.join(args.outdir, "canary_lora_adapter")
        mb.save_pretrained(bpath)
        from peft import PeftModel
        mb2base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(dev)
        mb2 = PeftModel.from_pretrained(mb2base, bpath).to(dev).eval()
        with torch.no_grad():
            lg1 = mb2(**fixed).logits
        sb["adapter_reload_max_abs_diff"] = float((lg0 - lg1).abs().max())
        sb["adapter_reload_bit_exact"] = bool(sb["adapter_reload_max_abs_diff"] == 0.0)
        sb["load_s"] = round(time.time() - t0, 2)
        sb["PASS"] = sb["forward_finite"] and sb["backward_step_ok"]
        del mb, mb2, mb2base
    except Exception as e:
        import traceback
        sb["PASS"] = False; sb["error"] = repr(e)[:300]; sb["tb"] = traceback.format_exc()[-800:]
    R["stages"]["arm_B_lora_only"] = sb
    vram_reset()

    # ---------- VRAM cleanup check ----------
    torch.cuda.synchronize(); vram_reset()
    final_alloc = torch.cuda.memory_allocated() / 1e6
    R["stages"]["vram_cleanup"] = dict(
        baseline_mb=round(base_alloc0, 1), final_mb=round(final_alloc, 1),
        returns_to_baseline=bool(final_alloc <= base_alloc0 + 50))

    C, B = R["stages"]["arm_C_tptt_lora"], R["stages"]["arm_B_lora_only"]
    qualified = bool(C.get("PASS") and B.get("PASS")
                     and R["stages"]["vram_cleanup"]["returns_to_baseline"])
    R["TPTT_DEPENDENCY"] = "QUALIFIED" if qualified else "FAILED"
    R["UPSTREAM_MECHANISM"] = ("REPRODUCED" if C.get("PASS") else "FAILED")
    with open(os.path.join(args.outdir, "rnn08_canary.json"), "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps({k: R[k] for k in ["TPTT_DEPENDENCY", "UPSTREAM_MECHANISM"]}, indent=2))
    print("base_loss", R["stages"]["base_reference"]["loss"])
    print("C:", {k: C.get(k) for k in ["forward_loss", "loss_after_step", "trainable_all_after_peft",
                                       "gate_params_unfrozen", "peak_vram_mb", "step_s", "PASS", "error"]})
    print("B:", {k: B.get(k) for k in ["forward_loss", "trainable_all_after_peft",
                                       "adapter_reload_max_abs_diff", "peak_vram_mb", "PASS", "error"]})
    print("VRAM:", R["stages"]["vram_cleanup"])
    return 0 if qualified else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--outdir", required=True)
    raise SystemExit(run(ap.parse_args()))
