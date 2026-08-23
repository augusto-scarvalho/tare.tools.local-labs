# LAB-ENGINE-001/002 — same-snapshot BF16 engine control

Status: **FROZEN BEFORE MODEL DOWNLOAD AND GENERATION**  
Date: 2026-08-22

## Question

On this RTX 3090, how do llama.cpp, SGLang and vLLM compare when serving the same official model
snapshot in the same BF16 precision class under an identical bounded client workload?

This is a small-model/BF16 control, not a universal engine ranking and not a proxy for the
production Qwen3.8 quantized workload.

## Frozen artifact and runtimes

- Source: `Qwen/Qwen3-4B`, Apache-2.0, Hugging Face revision
  `1cfa9a7208912126459214e8b04321603b3df60c`.
- Source tensors:
  - `model-00001-of-00003.safetensors`, 3,957,900,840 bytes, LFS SHA-256
    `328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223`;
  - `model-00002-of-00003.safetensors`, 3,987,450,520 bytes, LFS SHA-256
    `6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5`;
  - `model-00003-of-00003.safetensors`, 99,630,640 bytes, LFS SHA-256
    `e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1`.
- SGLang: isolated existing environment, version `0.5.16`, PyTorch `2.11.0+cu130`.
- llama.cpp: local binary version `10134`, commit `14d65fc45`; BF16 GGUF converted from the frozen
  snapshot by converter repository commit `87a416bd75d5a64e66e55846b779c0a54eca21bd`.
- vLLM: install into a new isolated environment using the current official CUDA-wheel procedure;
  freeze the resolved vLLM/PyTorch versions in the receipt before generation.
- Hardware: one NVIDIA GeForce RTX 3090, 24,576 MiB; NVIDIA driver `591.86`.

The GGUF format conversion changes the container and tokenizer metadata but must retain BF16 tensor
precision. If converter inspection reports quantized tensors or tensor-shape divergence, stop the
llama.cpp arm as invalid.

## Fixed workload

- Sequential engine blocks; only one generation server resident at a time. The embedding service on
  port 8081 remains untouched.
- Context length 8,192; one request at a time; OpenAI-compatible chat endpoint.
- Greedy decoding: temperature 0, top-p 1, fixed seed 4242 where supported.
- Two probes per measured round:
  - prefill: deterministic approximately 1,000-token passage, `max_tokens=16`;
  - decode: fixed short prompt, `max_tokens=256`.
- One warm-up round followed by five retained rounds per engine.
- Client records wall-clock time to first streamed token, total time, prompt/completion token counts,
  output text, finish reason, server command, startup time, and GPU residency.
- Engine order: llama.cpp, SGLang, vLLM. Sequential order/thermal drift is a stated limitation; a
  performance delta smaller than 10% is treated as unresolved and requires alternating server swaps.

## Gates and interpretation

Each engine arm is valid only if startup succeeds, all ten retained probes finish, outputs are
non-empty, token counts are sane, and no server/runtime error occurs. Report medians and every raw
round for TTFT, prefill tokens/s, decode tokens/s, total latency, and peak VRAM.

- `COMPARABLE_COMPLETE`: at least llama.cpp and SGLang are valid on the frozen pair.
- `VLLM_COMPLETE`: the isolated official wheel installs and the vLLM arm is valid.
- `BLOCKED_RUNTIME`: an engine cannot run the admitted snapshot/config; preserve logs and do not
  substitute a different model or quantization post hoc.
- No production default changes from this packet. A winner must exceed the peer by at least 10% in
  the relevant median and pass the validity gates; otherwise call that metric unresolved.

