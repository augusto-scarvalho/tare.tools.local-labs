# LAB-ENGINE-001/002 — result

Decision: **COMPARABLE_COMPLETE / VLLM_COMPLETE / REGIME-SPECIFIC**  
Date: 2026-08-22

All three engines completed five qualified fresh-prefill rounds and five decode rounds on the same
official Qwen3-4B revision and BF16 precision class. No qualified server log contains a traceback,
CUDA error, fatal error, or OOM.

| Engine | Startup s | Peak VRAM MiB | Fresh prefill tok/s | Prefill TTFT s | Decode tok/s | Decode TTFT s | Decode total s |
|---|---:|---:|---:|---:|---:|---:|---:|
| llama.cpp `14d65fc45` | 4.562 | 9,700 | 5,482.43 | 0.313 | 90.45 | 0.141 | 1.141 |
| SGLang 0.5.16 | 24.266 | 18,618 | 6,451.13 | 0.266 | 86.32 | 0.032 | 1.063 |
| vLLM 0.27.1 | 101.234 | 17,692 | 6,475.47 | 0.265 | 85.00 | 0.047 | 1.094 |

## Interpretation

- SGLang and vLLM improved fresh prefill throughput over llama.cpp by 17.67% and 18.11%, clearing
  the frozen 10% resolution threshold. They were mutually unresolved at 0.38% apart.
- Decode throughput was unresolved: llama.cpp led SGLang by 4.78% and vLLM by 6.40%, both below the
  10% threshold. Decode total latency also remained within 10% across all engines.
- SGLang/vLLM reduced median decode TTFT materially, but paid 1.92x/1.82x llama.cpp peak VRAM and
  5.3x/22.2x startup time. For this small BF16, concurrency-one regime, llama.cpp is the residency and
  startup winner; SGLang/vLLM are the fresh-prefill winners.
- Token accounting was identical on input (1,716 prefill and 50 decode tokens per round). Greedy
  decode was byte-identical across all engines in 2/5 nonce-matched rounds; each Python engine had
  occasional alternate valid continuations. This prevents a claim of bitwise cross-engine identity
  but does not violate the non-empty/sane-accounting performance gate.

The initial identical-prompt blocks were invalidated because prefix caching contaminated prefill.
They remain as `.pre-amendment` receipts. SGLang required a Pydantic compatibility repair before
model load. vLLM's first WSL startup failed on unavailable UVA; the documented
`VLLM_WSL2_ENABLE_PIN_MEMORY=1` recovery succeeded and is the qualified arm.

This result does not change the production Qwen3.8/llama.cpp default. It is evidence only for a
Qwen3-4B BF16, 8k-context, single-request regime.

## Evidence seals

- Model GGUF SHA-256: `6209b3a01d69a53fba10670ba002da976543e73d2d7152be309593615749818b`.
- Frozen packet SHA-256: `292f14a2093b89ea7455e06901bdcfd6811c3be6f6b2e71ab06d88833058aa7a`.
- Cache amendment SHA-256: `cc3cfec45b17bfddaf9f6b1fbc734097dcb8cc179505445a59a3b3c00fa53994`.
- UVA amendment SHA-256: `e2320b8427cfbec92f5059ebdb8289ee1228663e052761443b33a63097d8ac3c`.
- Qualified JSON SHA-256: llama.cpp `d0b3f4ce...`, SGLang `84ec938c...`, vLLM `56b7647f...`.

