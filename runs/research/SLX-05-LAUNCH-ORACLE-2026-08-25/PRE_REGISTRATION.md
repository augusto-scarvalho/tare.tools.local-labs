# SLX-05 Megakernel & Launch-Overhead Oracle - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: Modelos compactos sub-3B (como `Qwen/Qwen3.5-0.8B-Base`) operam com footprints de memória muito pequenos e latências de computação em GPU extremamente baixas por token ($< 2 \text{ ms}$). Em regime de decodificação sequencial (*batch=1*), o overhead de lançamento de múltiplos kernels individuais na CPU (CUDA driver API launch queue) compete diretamente com o tempo de execução física na GPU. O challenger **Lucebox** reporta ganho de até ~2× na RTX 3090 através de um megakernel persistente. Este oracle mede o teto teórico máximo recuperável pela eliminação do launch overhead via CUDA Events pareados e captura de CUDA Graphs antes de qualquer investimento em desenvolvimento de kernels persistentes.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Modelo Alvo**: `Qwen/Qwen3.5-0.8B-Base` (`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`)
* **Hardware**: NVIDIA GeForce RTX 3090 (24GB GDDR6X, `sm_86`), Intel Core i7-13700K
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824` (PyTorch 2.5.1 + CUDA 12.4)
* **Condições de Teste**:
  - Matriz de Batches: $B \in \{1, 2, 4\}$
  - Comprimento de Contexto: $L \in \{128, 512, 2048\}$
  - Regime de Execução: Eager Mode (padrão) vs CUDA Graphs (captura integral do grafo de decode)
* **Métricas Principais**:
  1. `cpu_launch_latency_ms`: Tempo de CPU gasto enfileirando os kernels de um passo de decodificação.
  2. `gpu_kernel_execution_ms`: Tempo real de GPU medido por `torch.cuda.Event(enable_timing=True)`.
  3. `cuda_graph_speedup_ratio`: Razão de aceleração entre Eager e CUDA Graphs (teto teórico do megakernel).
  4. `launch_overhead_fraction`: Proporção do tempo de inferência atribuível exclusivamente ao overhead de launch.

---

## 🛑 2. Critérios de Corte e Decisão (Kill Gates)

1. **Gate de Relevância ($< 5\%$)**: Se o overhead de launch for inferior a 5% do tempo total de decodificação em $B=1$, a linha de megakernels persistentes é **`REJECTED_MARGINAL`** (ganho insuficiente para justificar o blast radius de reescrever o motor).
2. **Gate de Qualificação ($\ge 15\%$)**: Se o overhead de launch for superior a 15%, o gargalo de CPU é **`CONFIRMED_LAUNCH_BOUND`**, autorizando o design de fusão de blocos ou extensão de CUDA Graphs no `slop.cpp`.
