# RETRO-01 Recurrent-Depth Retrofit - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: Em redes densas de 24 camadas, o retrofitting estrutural de 75% das camadas de atenção plena (18 camadas) para blocos recorrentes de atenção linear $O(1)$ (*Gated DeltaNet*) converte o modelo para a topologia híbrida ótima (Granite/Qwen3.5), alcançando **$\ge 3.0\times$ de aceleração na decodificação** e **$\ge 70\%$ de redução no consumo de KV cache** com retenção de logits $\text{Cosine Sim} \ge 0.985$.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Hardware**: NVIDIA GeForce RTX 3090 (24GB VRAM)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Topologias de Retrofit Comparadas (24 Camadas, $L=4096$)**:
  1. `DENSE_ATTENTION_ORIGINAL`: 24 camadas de atenção plena (Referência).
  2. `RETROFIT_50PCT`: 12 camadas MHA + 12 camadas SSM Linear.
  3. `RETROFIT_75PCT_HYBRID_3TO1`: 6 camadas MHA + 18 camadas SSM Linear.
* **Métricas**:
  1. `kv_cache_memory_mb`: Consumo de memória de estado em 4.096 tokens.
  2. `decode_latency_ms`: Tempo de geração por token na RTX 3090.
  3. `output_logits_cosine_sim`: Similaridade direcional do tensor final de saída.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Redução de Memória ($\ge 70\%$)**: Redução de pelo menos 70% no KV cache para a configuração 75%.
2. **Gate de Speedup ($\ge 2.5\times$)**: Aceleração de decodificação $\ge 2.5\times$ vs modelo denso original.
3. **Fidelidade de Retrofit ($\ge 0.980$)**: $\text{Cosine Sim} \ge 0.980$ nos embeddings de saída.
