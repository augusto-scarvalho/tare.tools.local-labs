# SLX-07 Hierarchical KV Cache Eviction (H2O) - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: Em sequências extensas de contexto ($L=4096$ tokens), a acumulação de escores de atenção segue uma distribuição lei de potência, onde uma fração mínima de tokens (*Heavy-Hitters*) concentra mais de $90\%$ da relevância semântica. Um mecanismo de evicção hierárquica H2O composto por $S=4$ sinks de atenção fixos, $R=64$ tokens locais recentes e $H=128$ tokens Heavy-Hitters dinâmicos reduz o tamanho do KV cache em **$95.2\%$** (de 4096 para 196 posições) mantendo **$100\%$ de recall em agulha no palheiro** (*Needle-in-a-Haystack*).

---

## 🎯 1. Contrato e Protocolo Experimental

* **Modelo Base**: `Qwen/Qwen3.5-0.8B-Base` (`head_dim=128`, $H=16$)
* **Hardware**: NVIDIA GeForce RTX 3090 (24GB VRAM)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Configuração de Contexto ($L=4096$)**:
  - Inserção de 3 fatos-chave (*Needles*) em profundidades $10\%$, $50\%$ e $90\%$ da sequência.
* **Políticas de Cache Comparadas**:
  1. `FULL_KV_CACHE`: $L=4096$ tokens (Sem evicção - Referência).
  2. `H2O_HIERARCHICAL`: $4\text{ Sinks} + 64\text{ Recent} + 128\text{ HeavyHitters} = 196\text{ tokens}$.
  3. `RANDOM_EVICTION`: 196 tokens amostrados aleatoriamente (Controle negativo).
* **Métricas**:
  1. `needle_retrieval_recall_pct`: Taxa de recuperação exata dos fatos inseridos.
  2. `attention_distribution_cosine_sim`: Similaridade direcional do softmax vs Full KV.
  3. `kv_memory_reduction_pct`: Redução percentual da pegada de memória do cache.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Recuperação de Fatos (Needle Recall $100\%$)**: 3 de 3 agulhas recuperadas com atenção dominante.
2. **Gate de Economia de Memória ($\ge 85\%$)**: Redução de pelo menos 85% no consumo de KV cache.
3. **Gate de Fidelidade vs Controle**: Similaridade de atenção do H2O superior em $\ge 30\%$ vs evicção aleatória.
