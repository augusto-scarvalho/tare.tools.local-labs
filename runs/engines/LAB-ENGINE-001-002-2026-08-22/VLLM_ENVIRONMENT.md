# vLLM isolated environment receipt

Frozen before vLLM generation on 2026-08-22.

- Installation procedure: new Python 3.12 venv at `/home/augus/vllm-venv`; `uv 0.12.5`; official
  documented command shape `uv pip install vllm --torch-backend=auto`.
- Resolved engine: `vllm 0.27.1`.
- Resolved compute stack: `torch 2.13.0+cu132`, CUDA runtime 13.2,
  `flashinfer-python 0.6.16.post3`, `transformers 5.15.1`.
- Preflight: `torch.cuda.is_available() == True`; device identified as
  `NVIDIA GeForce RTX 3090`.
- This environment is isolated from `/home/augus/sglang-venv` and from the canonical llama.cpp
  service. No server defaults were changed.

