# RSH-04 RaBitQCache Sparse Retrieval - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `REJECTED` — A indexação binária de 1 bit por projeção ortogonal aleatória capturou apenas **37.97% dos blocos de maior atenção** (perdendo 62% dos blocos críticos) devido ao descarte das normas de ativação, refutando o uso de sketches binários puros para filtragem de KV cache.

---

## 🎯 1. Resumo Executivo

O experimento avaliou a viabilidade de filtrar os top-25% blocos de KV cache mais relevantes em sequências de 4.096 tokens utilizando assinaturas binárias rotacionadas de 1 bit ($\text{sign}(R k)$) através de [`tools/probes/rsh04_rabitq_cache.py`](../../tools/probes/rsh04_rabitq_cache.py) na RTX 3090.

A hipótese de recuperação precisa via distância de Hamming foi **FALSIFICADA**:
- O sketch binário de 1 bit preserva apenas o ângulo direcional, descartando a magnitude absoluta das chaves ($||k||_2$).
- Como a atenção da Transformer é exponencial ($\text{softmax}(q^T k)$), picos de norma governam a distribuição de probabilidade; o filtro binário falhou em identificar blocos com normas dominantes, atingindo apenas **37.97% de recall** (muito abaixo do gate de 90.0%).
- Em contrapartida, o método hierárquico `H2O` (`SLX-07`) comprovou ser a solução correta com 100% de recall.

---

## 📊 2. Tabela de Métricas de Indexação Binária ($L=4096$, 128 Blocos)

| Método de Indexação | Blocos Selecionados | Recall dos Top Blocos | Latência do Filtro (µs) | Economia de DRAM | Veredito |
|---|:---:|:---:|:---:|:---:|:---:|
| **`EXACT_DENSE_SCAN`** | 128 (Todos) | 100.0% (Exato) | 0.00 µs (Base) | 0.0% | Referência |
| **`RABITQ_1BIT_SKETCH`**| **32 (Top-25%)** | **37.97% (FAIL GATE)** | **62.98 µs** | **75.00%** | `REJECTED` |

---

## 🔬 3. Diretriz de Projeto

1. **Descarte de Filtragem Binária de 1-bit no `slop.cpp`**:
   - Não utilizar hashing binário simples para evicção ou filtragem de blocos de atenção.
   - Adotar evicção acumulada de Heavy-Hitters (`H2O` / `SLX-07`) ou precisão mista por camada (`REP-05`).

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/RSH-04-RABITQ-CACHE-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Script da Prova**: [`tools/probes/rsh04_rabitq_cache.py`](../../tools/probes/rsh04_rabitq_cache.py)
- **Agente Executor**: Antigravity
