# REP-05 Layer-Wise Mixed Precision KV Cache - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `PROMOTED` — A política de precisão assimétrica por camada (Camadas Críticas 0..3 e 20..23 em FP16 + Camadas Intermediárias 4..19 em INT4) atingiu **49.00% de economia de memória no KV cache** mantendo **fidelidade quase perfeita ($\text{Cosine Sim} = 0.99976$, $\text{MSE} = 0.000886$)**, eliminando o colapso observado no INT4 homogêneo ($\text{Sim} = 0.93892$).

---

## 🎯 1. Resumo Executivo

O experimento comparou a sensibilidade à quantização através da profundidade de uma rede de 24 camadas em sequências de 4.096 tokens utilizando [`tools/probes/rep05_layerwise_kv_precision.py`](../../tools/probes/rep05_layerwise_kv_precision.py) na RTX 3090.

A hipótese de proteção por alocação assimétrica foi **CONFIRMADA**:
- O regime homogêneo INT4 em todas as 24 camadas acumulou ruído de arredondamento inter-camadas, degradando a similaridade direcional de saída para **0.93892** ($\text{MSE} = 0.514194$).
- A política **`LAYERWISE_MIXED`** protegeu as 8 camadas mais sensíveis (entrada sintática e projeção de vocabulário) enquanto comprimiu as 16 camadas intermediárias em INT4, reduzindo a pegada de memória de **768 MB para 391.7 MB (-49%)** com **0.99976 de similaridade direcional** com o FP16 de referência.

---

## 📊 2. Tabela de Comparação de Políticas de Precisão de KV ($L=4096$, 24 Camadas)

| Política de Precisão | Camadas FP16 | Camadas INT4 | Pegada de Memória | Redução de VRAM | Fidelidade de Saída (Cosseno) | Erro $\text{MSE}$ | Veredito |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`HOMOGENEOUS_FP16`** | 24 | 0 | 768.0 MB | 0.0% (Base) | 1.00000 | 0.000000 | Referência |
| **`HOMOGENEOUS_INT4`** | 0 | 24 | 203.5 MB | -73.50% | 0.93892 | 0.514194 | Degradação |
| **`LAYERWISE_MIXED`**  | **8** | **16** | **391.7 MB** | **-49.00% (PASS)** | **0.99976 (PASS)** | **0.000886** | **PROMOTED** |

---

## 🔬 3. Diretriz para o `slop.cpp`

1. **Alocação Mista de Precisão em Buffers de Atenção**:
   - Configurar o alocador de KV cache do `slop.cpp` para alocar tensores FP16 nas primeiras 4 camadas e nas últimas 4 camadas do modelo, enquanto as camadas intermediárias utilizam quantização INT4 em blocos de 32.

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/REP-05-LAYERWISE-PRECISION-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Script da Prova**: [`tools/probes/rep05_layerwise_kv_precision.py`](../../tools/probes/rep05_layerwise_kv_precision.py)
- **Agente Executor**: Antigravity
