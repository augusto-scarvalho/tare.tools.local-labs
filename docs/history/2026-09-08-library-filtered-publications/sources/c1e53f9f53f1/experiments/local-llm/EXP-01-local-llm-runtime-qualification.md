# 🧪 Experiment Card: EXP-01 — Local LLM Runtime Qualification & Hardware Acceleration

- **ID:** `EXP-01`
- **Cluster:** `11 Research frontier` / `local-models-evaluation`
- **Node Alvo:** `aaaaa` (NVIDIA GeForce RTX 3090 24GB VRAM / WSL2 Ubuntu 24.04 / CUDA 12.x)
- **Status:** `ACTIVE` (Execution & Benchmark Phase)
- **Prerequisites:** [ADR-048](file:///C:/projects/tare.tools.os/docs/ADR-048_LOCAL_INFERENCE_SUBSTRATE_AND_AGENT_HARNESS.md)
- **Author:** Antigravity Mediator & Engineering Lead

---

## 1. Hipótese & Motivação (Hypothesis)

> **Hipótese:** A execução de inferência local via `slop.cpp` com as 4 alavancas CUDA habilitadas (`GGML_KV_PIN_HOST=1`, FlashAttention `-fa on`, MTP Speculative Decoding $n=3$, e MoE hot-expert VRAM cache) no nó `aaaaa` entrega throughput $\ge 650\text{ prefill tokens/s}$ e $\ge 32\text{ decode tokens/s}$ para modelos de raciocínio de 27B/32B (ex: Qwen 3.6 Fable TC e Qwen 3.8 27B) em contextos de até 64k tokens, garantindo custo zero incremental de tokens e ausência de OOM na VRAM de 24 GB.

---

## 2. Métricas & Variáveis (Metrics)

| Métrica | Descrição | Unidade | Target de Sucesso |
| :--- | :--- | :--- | :--- |
| **`TTFT_cold`** | Time-To-First-Token em cache frio (15k-18k tokens de prompt inicial) | segundos | $\le 25.0\text{s}$ |
| **`TTFT_warm`** | Time-To-First-Token em cache quente ($R_{\text{cache}} \ge 0.90$) | segundos | $\le 2.0\text{s}$ |
| **`Prefill_Throughput`** | Taxa de avaliação do prompt não cacheado | tokens/s | $\ge 600\text{ t/s}$ |
| **`Decode_Throughput`** | Taxa de geração de tokens | tokens/s | $\ge 30\text{ t/s}$ |
| **`VRAM_Peak`** | Pico de memória alocado na RTX 3090 durante contexto de 64k | GB | $\le 23.5\text{ GB}$ (sem spill / OOM) |
| **`MTP_Acceptance_Rate`** | Taxa de aceitação de tokens do rascunho especulativo ($n=3$) | % | $\ge 80.0\%$ |

---

## 3. Baseline de Comparação (Baseline)

- **Baseline:** `llama.cpp` upstream padrão sem alocação page-locked (`GGML_KV_PIN_HOST=0`), sem FlashAttention e sem MTP draft.
  - Baseline TTFT Frio (18k tokens): ~45.2s.
  - Baseline Decode: ~16.5 t/s.
  - Baseline Prefill: ~350 t/s.

---

## 4. Critérios de Sucesso (Success Criteria)

1. Atingimento de $\ge 600\text{ t/s}$ de prefill e $\ge 30\text{ t/s}$ de decode sob quantização $Q4\_0$ / $Q4\_K\_M$.
2. Conexão e handshake estável na porta 8080 via túnel Tailscale com latência de rede $\le 15\text{ms}$.
3. Zero travamentos (`INFRA_CRASH`) ou estouros de VRAM (`INFRA_OOM`) em 20 requisições sequenciais de teste.

---

## 5. Critérios de Engavetamento / Abandono (Abandon Criteria)

- Prefill throughput $< 350\text{ t/s}$ ou decode $< 20\text{ t/s}$, indicando gargalo de barramento PCIe ou contenção de CPU no WSL2.
- Taxa de falha de conexão HTTP $> 5\%$ via Tailscale sob carga.

---

## 6. Plano de Reversão (Reversal Plan)

- Reverter para flags conservadoras no `llama-server` (`-fa off`, sem speculative decoding) ou redirecionar tráfego para fallback em nuvem $T_1$.
