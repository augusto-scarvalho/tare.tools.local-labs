# REP-04 KVarN Native Attention Kernel - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: A fusão do codec KVarN (Transformação de Walsh-Hadamard + Corpo INT4 + Cauda Recente FP16 $T=64$) diretamente dentro do pipeline de atenção FlashAttention elimina o gargalo de largura de banda de memória DRAM na RTX 3090, proporcionando **$\ge 2.0\times$ de aceleração na latência do kernel de atenção em $L=8192$** com **$\ge 0.998$ de similaridade de cosseno** nos mapas de probabilidade de atenção.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Hardware**: NVIDIA GeForce RTX 3090 (24GB GDDR6X, 936 GB/s)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Configuração de Atenção**: $L=8192$, $H=16$, $d_{\text{head}}=128$
* **Regimes Comparados**:
  1. `STANDARD_FLASH_ATTN_FP16`: Atenção padrão carregando $1.5\text{ GB}$ de KV cache em FP16.
  2. `FUSED_KVARN_HYBRID_INT4`: Kernel fundido KVarN (Corpo INT4 rotacionado + Cauda FP16 $T=64$).
* **Métricas**:
  1. `kernel_latency_us`: Latência de execução do kernel de atenção por token.
  2. `effective_speedup_factor`: Aceleração relativa vs FlashAttention FP16.
  3. `attention_softmax_cosine_sim`: Fidelidade dos mapas de atenção softmax.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Speedup ($\ge 1.80\times$)**: Redução de tempo de atenção $\ge 1.80\times$ em $L=8192$.
2. **Gate de Fidelidade ($\ge 0.995$)**: $\text{Cosine Sim} \ge 0.995$.
3. **Economia de Tráfego DRAM ($\ge 70\%$)**: Redução $\ge 70\%$ nos bytes trafegados por token.
