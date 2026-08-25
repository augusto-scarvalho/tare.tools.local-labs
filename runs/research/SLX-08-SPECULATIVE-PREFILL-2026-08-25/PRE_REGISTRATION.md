# SLX-08 Speculative Prefill (PFlash) - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: Em contextos extensos ($L \in [1024, 8192]$ tokens), o Time-To-First-Token (TTFT) cresce quadraticamente com o comprimento da sequência em camadas de atenção densa. Um pipeline de *Speculative Prefill* (PFlash), que utiliza um modelo draft de baixo custo para computar a distribuição esparsa de relevância de blocos e direcionar a computação pesada do target apenas para os blocos críticos de atenção, reduz o TTFT em $\ge 1.40\times$ em sequências $\ge 4096$ tokens mantendo fidelidade de representação ($\text{Cosine Sim} \ge 0.98$).

---

## 🎯 1. Contrato e Protocolo Experimental

* **Hardware**: NVIDIA GeForce RTX 3090 (24GB GDDR6X)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Sequências Avaliadas**: $L \in \{1024, 2048, 4096, 8192\}$ tokens.
* **Políticas Avaliadas**:
  1. `DENSE_STANDARD_PREFILL`: Atenção densa $O(L^2)$ padrão.
  2. `CHUNKED_SPECULATIVE_PREFILL`: Prefill especulativo em blocos de 256 tokens com filtragem top-50% de blocos pelo draft.
* **Métricas**:
  1. `ttft_ms`: Tempo até a geração do primeiro token em milissegundos.
  2. `speedup_factor`: Razão $\frac{\text{TTFT}_{\text{dense}}}{\text{TTFT}_{\text{speculative}}}$.
  3. `logits_cosine_sim`: Fidelidade dos logits resultantes no passo $L$.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Aceleração em Contexto Longo ($L=8192$)**: Speedup no TTFT $\ge 1.40\times$.
2. **Gate de Fidelidade de Representação**: Cosine Similarity $\ge 0.95$ nos logits finais.
3. **Escalabilidade Monótona**: Speedup crescente com o aumento de $L$.
