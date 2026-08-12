# RNN-06T — Official Mamba fast-path environment provenance

Project-local/isolated dependency work to make the **official** `mamba_ssm` fast path runnable on
this host. No global Windows/driver policy changed. Additive install into the existing project venv
`/home/augus/rnn06_env` (torch/transformers untouched, `--no-deps`).

## Host / base

- WSL Ubuntu 24.04, RTX 3090 (compute capability 8.6), driver 591.86.
- Python 3.12.3, torch 2.6.0+cu124 (CUDA 12.4), **`torch._C._GLIBCXX_USE_CXX11_ABI = False`** (old
  ABI; `libc10.so` exports `torchCheckFail(...RKSs)`), triton 3.2.0.
- No `nvcc` / CUDA toolkit → build-from-source not possible; prebuilt wheels required.

## ABI wheel-matching (the non-obvious part)

The official mamba/causal-conv1d release matrices ship `cxx11abiTRUE` and `cxx11abiFALSE` wheels per
torch row, but for the torch2.6 row the **1.6.x / 2.2.5+ "abiFALSE" wheels are mislabeled** — their
`.so` references the new-ABI symbol `std::__cxx11::basic_string` (`torchCheckFail(...__cxx11...)`),
which this old-ABI torch does not export (`undefined symbol`). The correctly old-ABI-built wheels are:

- **causal_conv1d 1.5.0.post8** — `causal_conv1d-1.5.0.post8+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl`
- **mamba_ssm 2.2.4** — `mamba_ssm-2.2.4+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl`

(2.2.3.post1 also worked; 2.2.5 / 2.3.x abiFALSE and all 1.6.x abiFALSE did NOT import against this
torch.) `causal_conv1d_cuda.so` resolves `libc10.so` only after `import torch` (RPATH), which is the
normal usage order.

## Pinned fast-path stack (frozen for this train)

| component | version |
|---|---|
| official checkpoint | `state-spaces/mamba2-1.3b` @ `c5b59d00ec85d313adea86a08cad2a43c962dd3b` |
| mamba_ssm | 2.2.4 (+cu12torch2.6cxx11abiFALSE-cp312) |
| causal_conv1d | 1.5.0.post8 (+cu12torch2.6cxx11abiFALSE-cp312) |
| triton | 3.2.0 |
| torch | 2.6.0+cu124 (cxx11abi=False) |
| python | 3.12.3 |

## Fast-path firing proof (not "installed" — executed; see OFFICIAL_MAMBA_ENV.json)

Loaded via the official `mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel` (48 layers, d_model 2048,
Mamba2, bf16). Kernel entry points in `mamba_ssm.modules.mamba2` were wrapped with call counters:

- **prefill** (inference_params, seqlen_offset 0): `mamba_chunk_scan_combined` ×48 (1/layer) +
  `causal_conv1d_fn` ×48 — real Triton SSD + CUDA conv.
- **step** (decode, 5 tokens): `selective_state_update` ×240 and `causal_conv1d_update` ×240 =
  exactly `n_layer(48) × n_step(5)` — real Triton/CUDA per layer per token.
- **no fallback reachable** (all three kernel symbols non-None).
- `use_mem_eff_path` (fused `mamba_split_conv1d_scan_combined`) = True, but it is bypassed by design
  whenever `inference_params` is passed (the state-capture path); it is available for the FINAL-only
  economics baseline.

⇒ `FAST_PATH_ACTIVE = True`, `OFFICIAL_MAMBA_FASTPATH = RUNNABLE`.
