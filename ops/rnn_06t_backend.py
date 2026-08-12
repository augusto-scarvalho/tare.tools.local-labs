#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06T Section 1 — official Mamba backend + fast-path FIRING qualification.

Resolves the official state-spaces/mamba2-1.3b checkpoint, loads it via the official mamba_ssm
MambaLMHeadModel (NOT transformers), and PROVES the CUDA/Triton fast path actually fires by
instrumenting the kernel entry points in mamba_ssm.modules.mamba2 with call counters:
  prefill: mamba_split_conv1d_scan_combined | (causal_conv1d_fn + mamba_chunk_scan_combined)
  step   : causal_conv1d_update + selective_state_update
"package installed" is not evidence: we assert the step kernels fire exactly n_layer per decoded
token and that no fallback branch (kernel is None) is reachable. Emits OFFICIAL_MAMBA_FASTPATH +
full environment/kernel identity. No outcomes-bearing recovery here.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import torch

REPO = "/mnt/c/projects/local-model-lifecycle"
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06T")
REPO_ID = "state-spaces/mamba2-1.3b"
DEVICE, DTYPE = "cuda", torch.bfloat16


def git(*a):
    try:
        return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"<git-error:{e}>"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    runner = os.path.abspath(__file__)
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    info = api.model_info(REPO_ID)
    revision = info.sha
    cfg_path = hf_hub_download(REPO_ID, "config.json", revision=revision)
    config = json.load(open(cfg_path))

    import mamba_ssm
    import causal_conv1d
    import triton
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.utils.generation import InferenceParams
    import mamba_ssm.modules.mamba2 as m2

    # ---- instrument kernel entry points (patch the names bound inside mamba2 module) ----
    counters = {"mamba_split_conv1d_scan_combined": 0, "mamba_chunk_scan_combined": 0,
                "causal_conv1d_fn": 0, "causal_conv1d_update": 0, "selective_state_update": 0}
    originals = {}

    def wrap(name):
        orig = getattr(m2, name)
        originals[name] = orig
        if orig is None:
            return
        def wrapped(*a, **k):
            counters[name] += 1
            return orig(*a, **k)
        setattr(m2, name, wrapped)

    for nm in list(counters):
        wrap(nm)
    fallback_reachable = {"causal_conv1d_fn_is_None": m2.causal_conv1d_fn is None,
                          "causal_conv1d_update_is_None": m2.causal_conv1d_update is None,
                          "selective_state_update_is_None": m2.selective_state_update is None}

    # ---- load official model ----
    t0 = time.time()
    model = MambaLMHeadModel.from_pretrained(REPO_ID, device=DEVICE, dtype=DTYPE)
    model.eval()
    load_s = time.time() - t0
    n_layer = len(model.backbone.layers)
    # NOTE: the fully-fused prefill kernel (mamba_split_conv1d_scan_combined) is gated by
    # `use_mem_eff_path` AND requires inference_params is None. The state-capture trajectory ALWAYS
    # passes inference_params, so it uses the (still-Triton/CUDA) chunk_scan + causal_conv1d_fn
    # prefill and the selective_state_update + causal_conv1d_update step kernels. The fused flag is
    # therefore irrelevant to the capture fast path; recorded descriptively only.
    mixer0 = model.backbone.layers[0].mixer
    use_mem_eff_path = bool(getattr(mixer0, "use_mem_eff_path", False))
    weights_id = hashlib.sha256(
        "|".join(f"{n}:{float(p.float().sum()):.4e}" for n, p in model.named_parameters()).encode()
    ).hexdigest()

    # ---- exercise prefill + step to force fast path ----
    torch.manual_seed(0)
    B, PREFILL, NSTEP = 2, 8, 5
    ids = torch.randint(0, config["vocab_size"], (B, PREFILL), device=DEVICE)
    inf = InferenceParams(max_seqlen=64, max_batch_size=B)
    inf.key_value_memory_dict = model.allocate_inference_cache(B, 64, dtype=DTYPE)

    c_before = dict(counters)
    with torch.no_grad():
        # prefill (seqlen_offset == 0): fused/prefill kernels
        out = model(ids, inference_params=inf)
        inf.seqlen_offset += PREFILL
        prefill_counts = {k: counters[k] - c_before[k] for k in counters}
        # step NSTEP single tokens (decode): step kernels fire n_layer per token
        c_step0 = dict(counters)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        for _ in range(NSTEP):
            out = model(nxt, inference_params=inf)
            inf.seqlen_offset += 1
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        step_counts = {k: counters[k] - c_step0[k] for k in counters}

    # ---- fast-path firing proof ----
    step_ssu = step_counts["selective_state_update"]
    step_ccu = step_counts["causal_conv1d_update"]
    prefill_fast = (prefill_counts["mamba_split_conv1d_scan_combined"] +
                    prefill_counts["mamba_chunk_scan_combined"]) >= n_layer
    step_fast = (step_ssu == n_layer * NSTEP) and (step_ccu == n_layer * NSTEP)
    no_fallback = not any(fallback_reachable.values())
    # Fast path for the state-capture trajectory = real Triton/CUDA kernels fire with exact expected
    # counts (n_layer per step for the decode kernels) and no fallback branch is reachable. The
    # fused mem-eff prefill flag is NOT required (it is bypassed under inference_params by design).
    fast_path_active = bool(prefill_fast and step_fast and no_fallback)

    env = {
        "packet": "RNN-06T-BACKEND", "kind": "official_mamba_fastpath_qualification",
        "repo_id": REPO_ID, "revision": revision, "config": config,
        "model_weights_identity": weights_id, "n_layer": n_layer, "d_model": config["d_model"],
        "vocab_size": config["vocab_size"], "ssm_layer": config["ssm_cfg"].get("layer"),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "cxx11abi": torch._C._GLIBCXX_USE_CXX11_ABI,
        "cuda_device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "driver_version": subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
        "mamba_ssm_version": mamba_ssm.__version__, "causal_conv1d_version": causal_conv1d.__version__,
        "triton_version": triton.__version__, "dtype": str(DTYPE), "device": DEVICE,
        "use_mem_eff_path_fused_prefill_flag": use_mem_eff_path,
        "kernel_path": "mamba_ssm-native-triton-ssd + causal_conv1d-cuda (inference_params capture path)",
        "fallback_reachable": fallback_reachable, "prefill_kernel_counts": prefill_counts,
        "step_kernel_counts": step_counts, "n_step_probe": NSTEP,
        "step_selective_state_update_expected": n_layer * NSTEP,
        "step_causal_conv1d_update_expected": n_layer * NSTEP,
        "prefill_fast_path_fired": prefill_fast, "step_fast_path_fired": step_fast,
        "no_fallback": no_fallback, "FAST_PATH_ACTIVE": fast_path_active,
        "model_load_s": round(load_s, 1), "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "runner_sha256": sha256_file(runner), "runner_git_blob": git("hash-object", runner),
        "runner_dirty": git("status", "--porcelain", "--", runner), "git_head": git("rev-parse", "HEAD"),
    }
    env["OFFICIAL_MAMBA_FASTPATH"] = "RUNNABLE" if fast_path_active else "NOT_RUNNABLE_WITHIN_BUDGET"

    with open(os.path.join(OUTDIR, "OFFICIAL_MAMBA_ENV.json"), "w") as f:
        json.dump(env, f, indent=2, default=str)

    print(f"repo={REPO_ID} revision={revision}")
    print(f"mamba_ssm={mamba_ssm.__version__} causal_conv1d={causal_conv1d.__version__} "
          f"triton={triton.__version__} torch={torch.__version__} cxx11abi={torch._C._GLIBCXX_USE_CXX11_ABI}")
    print(f"n_layer={n_layer} use_mem_eff_path(fused_prefill)={use_mem_eff_path} fallback_reachable={fallback_reachable}")
    print(f"prefill_counts={prefill_counts}")
    print(f"step_counts={step_counts} expected selective_state_update={n_layer*NSTEP}")
    print(f"prefill_fast={prefill_fast} step_fast={step_fast} no_fallback={no_fallback}")
    print(f"FAST_PATH_ACTIVE={fast_path_active}")
    print(f"OFFICIAL_MAMBA_FASTPATH = {env['OFFICIAL_MAMBA_FASTPATH']}")


if __name__ == "__main__":
    main()
