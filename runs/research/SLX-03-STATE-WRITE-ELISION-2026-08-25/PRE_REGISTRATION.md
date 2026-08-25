# SLX-03 ReplaySSM State-Write Elision - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: Em arquiteturas híbridas recorrentes (como as 18 camadas Gated DeltaNet do Qwen3.5), atualizar e gravar o estado recorrente completo ($S_t \in \mathbb{R}^{18 \times d_{\text{in}} \times d_{\text{out}}}$) na memória DRAM global a cada token de decodificação gera um estrangulamento severo de largura de banda de memória. A elisão de escrita de estado (*State-Write Elision*), mantendo o estado recorrente em registradores/SRAM e persistindo na DRAM global apenas a cada $N$ tokens ou em limites de transação, reduz o tráfego de escrita em $\ge 70\%$ com speedup de decodificação $\ge 1.30\times$ e zero divergência numérica.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Modelo Alvo**: `Qwen/Qwen3.5-0.8B-Base` (18 camadas Gated DeltaNet)
* **Hardware**: NVIDIA GeForce RTX 3090 (24GB VRAM, 936 GB/s de banda)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Políticas de Escrita de Estado**:
  1. `PERSIST_EVERY_TOKEN` ($N=1$): Gravação síncrona na DRAM global a cada passo (Baseline Padrão).
  2. `ELISION_N4` ($N=4$): Gravação na DRAM a cada 4 tokens.
  3. `ELISION_N16` ($N=16$): Gravação na DRAM a cada 16 tokens.
  4. `EPHEMERAL_REGISTER_ONLY`: Estado puramente mantido em SRAM durante toda a geração.
* **Métricas**:
  1. `state_bytes_written_mb`: Volume total de bytes de estado transferidos para a memória global.
  2. `decode_latency_per_token_ms`: Latência média de decodificação por token.
  3. `logits_divergence_mse`: Erro numérico vs baseline (meta: 0.0).

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Fidelidade Numérica ($0.0$ MSE)**: Zero distorção nos logits gerados pelo modelo.
2. **Gate de Redução de Tráfego de Estado ($\ge 70\%$)**: Redução de pelo menos 70% no volume de IO de estado.
3. **Gate de Aceleração ($\ge 1.20\times$)**: Speedup mensurável no loop de decodificação de atenção linear.
