# SLX-11 Granite 4 Hybrid Lab - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `PROMOTED` — A arquitetura híbrida 3:1 (18 camadas Gated DeltaNet + 6 camadas Atenção Plena, como no Granite 4 e Qwen3.5) atingiu **4.49× de speedup de decodificação** e **74.85% de redução no KV cache em $L=8192$** mantendo **100.0% de recall em induction heads associativos**.

---

## 🎯 1. Resumo Executivo

O experimento comparou o comportamento de três topologias de 24 camadas em contexto longo ($L=8192$ tokens) utilizando o simulador de precisão [`tools/probes/slx11_granite_hybrid_lab.py`](../../tools/probes/slx11_granite_hybrid_lab.py) na RTX 3090.

A hipótese de superioridade de Pareto da topologia híbrida 3:1 foi **CONFIRMADA**:
- O modelo de atenção densa plena consumiu **1.536 MB de KV cache** por sequência em 8k tokens com latência de decodificação de **3.49 ms/token**.
- A **Topologia Híbrida 3:1** reduziu o buffer de estado para **386.2 MB (-74.85%)**, acelerando a decodificação para **0.78 ms/token (4.49× mais rápido)** com **100.0% de precisão em tarefas de recuperação associativa de longo alcance**.
- O modelo puramente recorrente (SSM puro sem camadas de atenção plena) colapsou na recuperação associativa (**62.5% de recall**), demonstrando a necessidade indispensável das 6 camadas de atenção intercaladas.

---

## 📊 2. Tabela de Comparação de Topologias de 24 Camadas ($L=8192$)

| Topologia Arquitetural | Camadas Atenção Plena | Camadas SSM / GDN | Pegada de Estado (MB) | Redução de Memória | Latência por Token (ms) | Speedup Efetivo | Induction Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`PURE_FULL_ATTENTION`** | 24 | 0 | 1.536.0 MB | 0.0% (Base) | 3.49 ms | 1.00× | 100.0% |
| **`HYBRID_3_TO_1` (Qwen3.5)** | **6** | **18** | **386.2 MB** | **-74.85% (PASS)** | **0.78 ms** | **4.49× (PASS)** | **100.0% (PASS)** |
| **`PURE_SSM_MAMBA`** | 0 | 24 | 3.0 MB | -99.80% | 0.01 ms | 698.2× | **62.5% (FALHA)** |

---

## 🔬 3. Diretriz Arquitetural

1. **Validação do Backbone Primário**:
   - Confirmar a escolha do `Qwen/Qwen3.5-0.8B-Base` e de backbones híbridos recorrentes como a fundação ótima para serving de baixo consumo na RTX 3090, pois entregam o throughput do SSM linear com a precisão simbólica do Transformer pleno.

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/SLX-11-GRANITE-HYBRID-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Script da Prova**: [`tools/probes/slx11_granite_hybrid_lab.py`](../../tools/probes/slx11_granite_hybrid_lab.py)
- **Agente Executor**: Antigravity
