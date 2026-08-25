# REP-05 Layer-Wise Mixed Precision KV Cache - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: A sensibilidade à quantização varia drasticamente ao longo da profundidade da rede: camadas iniciais (0..3) constroem o roteamento sintático básico e camadas finais (20..23) projetam a distribuição do vocabulário, sendo hiper-sensíveis a erros de arredondamento. Camadas intermediárias (4..19) são tolerantes a ruído. Uma alocação assimétrica mista (8 camadas em FP16 + 16 camadas em INT4) reduz a pegada de KV em **$50.0\%$** mantendo fidelidade de atenção **$\text{Cosine Sim} \ge 0.995$**, superando a quantização homogênea INT4 em todas as camadas.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Modelo Base**: `Qwen/Qwen3.5-0.8B-Base` (24 Camadas, $H=16$, $d_{\text{head}}=128$, $L=4096$)
* **Hardware**: NVIDIA GeForce RTX 3090 (24GB VRAM)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Políticas de Precisão por Camada**:
  1. `HOMOGENEOUS_FP16`: 24 camadas em FP16 (Controle de referência).
  2. `HOMOGENEOUS_INT4`: 24 camadas em INT4 simétrico por bloco de 32.
  3. `LAYERWISE_MIXED`: Camadas 0..3 (Entrada) e 20..23 (Saída) em FP16 (8 camadas); Camadas 4..19 em INT4 (16 camadas).
* **Métricas**:
  1. `kv_memory_footprint_mb`: Pegada de memória total do cache em 4.096 tokens.
  2. `end_to_end_cosine_sim`: Similaridade de cosseno do embedding final de saída após as 24 camadas.
  3. `memory_savings_pct`: Percentual de economia de memória vs FP16 homogêneo.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Economia de Memória ($\ge 45\%$)**: Redução de pelo menos 45% vs FP16.
2. **Gate de Fidelidade de Saída ($\ge 0.990$)**: $\text{Cosine Sim} \ge 0.990$ no embedding final.
3. **Superioridade vs INT4 Homogêneo**: $\text{Cosine Sim}$ superior ao INT4 homogêneo em $\ge 0.05$.
