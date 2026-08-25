# SLX-11 Granite 4 Hybrid Lab - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: Em arquiteturas de 24 camadas, a topologia híbrida 3:1 (18 camadas recorrentes SSM / Gated DeltaNet + 6 camadas de Atenção Plena quadrática, como em Granite 4 e Qwen3.5) reduz o consumo de KV cache em **$75\%$** e acelera a decodificação em $\ge 2.0\times$ em contextos longos ($L=8192$) mantendo **$100\%$ de capacidade de cópia associativa e recuperação de induction heads**, superando tanto a atenção pura (gargalo de VRAM) quanto o SSM puro (incapaz de recall exato de sub-sequências longas).

---

## 🎯 1. Contrato e Protocolo Experimental

* **Hardware**: NVIDIA GeForce RTX 3090 (24GB GDDR6X)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Topologias Comparadas (24 Camadas, $d_{\text{model}}=2048$, $H=16$, $d_{\text{head}}=128$, $L=8192$)**:
  1. `PURE_FULL_ATTENTION`: 24 camadas de atenção densa MHA.
  2. `HYBRID_3_TO_1` (Granite 4 / Qwen3.5): 18 camadas Gated DeltaNet + 6 camadas Full Attention.
  3. `PURE_SSM_MAMBA`: 24 camadas puramente recorrentes sem KV cache.
* **Métricas**:
  1. `kv_cache_size_mb_at_8k`: Tamanho físico do buffer de KV cache em 8.192 tokens.
  2. `induction_head_recall_pct`: Acurácia na tarefa sintética de cópia associativa de longo alcance ($A \dots B \dots A \rightarrow B$).
  3. `decode_throughput_tokens_sec`: Throughput de geração sequencial.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Economia de KV Cache ($\ge 70\%$)**: Topologia híbrida deve consumir $\le 30\%$ da VRAM de atenção plena.
2. **Gate de Retenção Associativa (Induction Head $\ge 95\%$)**: Manter capacidade associativa onde SSM puro falha.
3. **Speedup de Decodificação ($\ge 1.50\times$)**: Aceleração sustentada em 8.192 tokens.
