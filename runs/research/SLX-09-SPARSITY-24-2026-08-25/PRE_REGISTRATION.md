# SLX-09 Sparsidade Estruturada 2:4 Ampere - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: A arquitetura NVIDIA Ampere (RTX 3090, `sm_86`) introduziu suporte em hardware para matrizes esparsas estruturadas 2:4 (exatamente 2 valores não-nulos a cada bloco contíguo de 4 valores), dobrando teoricamente a taxa de processamento dos Tensor Cores. Este oracle avalia se a poda estruturada 2:4 guiada por magnitude ponderada por ativação preserva a fidelidade de representação ($\text{Cosine Sim} \ge 0.95$, $\Delta \text{PPL} \le 10\%$) e quantifica o ganho real de throughput no hardware da estação.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Modelo Alvo**: `Qwen/Qwen3.5-0.8B-Base` (`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`)
* **Hardware**: NVIDIA GeForce RTX 3090 (GA102, `sm_86`, 24GB VRAM)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Políticas de Poda 2:4**:
  1. `DENSE_FP16`: Matriz densa sem poda (referência de controle).
  2. `UNSTRUCTURED_50`: Poda não-estruturada de 50% dos menores pesos globais.
  3. `STRUCTURED_2_4_MAGNITUDE`: Poda estruturada 2:4 baseada na magnitude absoluta local de cada quarteto.
  4. `STRUCTURED_2_4_WANDA`: Poda estruturada 2:4 ponderada pelo produto da magnitude dos pesos pela norma L2 das ativações ($|W| \cdot ||X||_2$).
* **Métricas**:
  1. `logits_cosine_similarity`: Fidelidade direcional dos logits gerados vs dense baseline.
  2. `logits_mse`: Erro quadrático médio de predição.
  3. `sparsity_ratio`: Exatamente 50.0% de esparsidade com padrão 2:4 estrito.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Fidelidade de Poda**: A política 2:4 deve manter Cosine Similarity $\ge 0.90$ nos logits em relação ao modelo denso sem retreinamento prévio.
2. **Gate de Conformidade Estrutural**: 100% dos blocos de 4 elementos no tensor de pesos devem satisfazer a restrição 2:4.
3. **Superioridade de Wanda sobre Magnitude**: Poda ponderada por ativações deve reduzir o MSE em pelo menos $20\%$ sobre a poda pura de magnitude.
