# GDN-02 Gated DeltaNet-2 Erase & Retention Lab - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: Em memórias recorrentes $O(1)$, a atualização ou esquecimento seletivo de uma chave específica $(K_5)$ através de Delta-Rule clássica ou decaimento estático causa esquecimento colateral em chaves com projeções similares. A formulação do **Gated DeltaNet-2** (com vetor de gating não-linear condicionado na query e chave: $g_t = \sigma(W_q q_t + W_k k_t) \odot \beta_t$) desacopla a rota de apagamento, atingindo **$\ge 95\%$ de supressão do fato obsoleto** com **$\ge 90\%$ de preservação dos fatos colaterais não-relacionados**.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Hardware**: NVIDIA GeForce RTX 3090 (24GB VRAM)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Mecanismos Recorrentes Comparados ($d_{\text{state}}=64 \times 64$, 50 fatos associativos)**:
  1. `STATIC_DECAY_SSM`: Matriz de estado com decaimento temporal escalar $\alpha \in [0.9, 0.99]$.
  2. `CLASSIC_DELTANET`: Delta-rule matricial padrão sem gating dependente de query.
  3. `QUERY_GATED_DELTANET2`: Gated DeltaNet-2 com porta de esquecimento ortogonal seletiva.
* **Métricas**:
  1. `erased_fact_leakage_pct`: Percentual de resíduo do fato obsoleto recuperado após o comando de apagamento (meta: $\le 5\%$).
  2. `collateral_memory_retention_pct`: Taxa de acerto na recuperação dos outros 49 fatos intocados (meta: $\ge 90\%$).
  3. `updated_fact_fidelity_pct`: Fidelidade na recuperação do novo valor substituído (meta: $\ge 95\%$).

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Esquecimento Seletivo ($\le 5\%$ de Vazamento)**: $\text{Leakage} \le 5.0\%$.
2. **Gate de Retenção Colateral ($\ge 90\%$)**: Preservação $\ge 90.0\%$ nas chaves não-alvo.
3. **Fidelidade de Atualização ($\ge 95\%$)**: Recuperação $\ge 95.0\%$ do novo fato.
