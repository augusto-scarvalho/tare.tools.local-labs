# RSH-04 RaBitQCache Sparse Retrieval - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: A projeção ortogonal aleatória dos vetores de chave em sketches binários compactos de 1 bit ($\text{sign}(R k) \in \{0, 1\}^{128}$) permite a seleção dos top-25% blocos de KV cache mais relevantes através de operações ultrarrápidas de `popcount` (distância de Hamming), atingindo **$\ge 95\%$ de recall dos blocos de maior atenção** e reduzindo o volume de DRAM carregado em **$75\%$**.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Hardware**: NVIDIA GeForce RTX 3090 (24GB VRAM)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Configuração de Avaliação**: $L=4096$, $H=16$, $d_{\text{head}}=128$, Blocos de 32 tokens (128 blocos).
* **Mecanismos Comparados**:
  1. `EXACT_FP16_DENSE_SCAN`: Varredura exata de todos os 128 blocos.
  2. `RABITQ_BINARY_SKETCH`: Filtragem binária de 1 bit por bloco selecionando os 32 blocos top (25%).
* **Métricas**:
  1. `topk_block_recall_pct`: Percentual dos 32 blocos de maior atenção real capturados pelo sketch binário.
  2. `hamming_filter_latency_us`: Tempo de filtragem dos 128 blocos na GPU.
  3. `dram_bytes_saved_pct`: Redução de dados carregados do cache.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Recall dos Top Blocos ($\ge 90\%$)**: Capturar $\ge 90.0\%$ dos blocos críticos de atenção.
2. **Gate de Economia de DRAM ($\ge 70\%$)**: Carregamento de apenas 25% a 30% dos blocos.
3. **Latência do Filtro ($\le 50\text{ µs}$)**: Overhead de indexação binária desprezível.
