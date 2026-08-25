# REP-06 Online Dynamic Precision KV - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `REJECTED` — A precisão dinâmica guiada por entropia elevou a fidelidade de atenção em relação ao INT4 estático (**0.97482 vs 0.91737**) com **68.84% de economia de VRAM (4.99 bpw)**, mas a quantização de 2 bits nos tokens de baixa entropia violou o gate estrito ($\ge 0.992$), demonstrando a superioridade da precisão por camada (`REP-05`).

---

## 🎯 1. Resumo Executivo

O experimento testou a alocação dinâmica de precisão por token (2 bits para baixa entropia, 4 bits para média, FP16 para alta entropia) em sequências de 2.048 tokens na RTX 3090 através de [`tools/probes/rep06_dynamic_entropy_precision.py`](../../tools/probes/rep06_dynamic_entropy_precision.py).

A hipótese de fidelidade perfeita via triagem de entropia foi **FALSIFICADA**:
- O método dinâmico atingiu **68.84% de economia de memória** (consumindo em média 4.99 bits/elemento) e superou com folga o INT4 estático homogêneo ($\text{Sim} = 0.91737$).
- Contudo, a perda residual nos 44.4% de tokens quantizados em 2 bits impediu que a similaridade atingisse o teto de 0.992, ficando em **0.97482**.
- A alocação mista por camada (`REP-05`), que protege camadas inteiras em vez de tokens individuais, permanece a estratégia recomendada com 0.9998 de similaridade.

---

## 📊 2. Tabela de Comparação de Políticas de Precisão de KV ($L=2048$)

| Política de Precisão | Distribuição de Bits | Média de Bits / Elem | Economia de VRAM | Similaridade Cosseno | Veredito |
|---|:---:|:---:|:---:|:---:|:---:|
| **`STATIC_FP16`** | 100% FP16 | 16.00 bpw | 0.0% (Base) | 1.00000 | Referência |
| **`STATIC_INT4`** | 100% INT4 | 4.00 bpw | 75.00% | 0.91737 | Degradação |
| **`DYNAMIC_ENTROPY`**| **2b (44%), 4b (40%), 16b (16%)** | **4.99 bpw** | **68.84% (PASS)** | **0.97482 (FAIL GATE)** | `REJECTED` |

---

## 🔬 3. Diretriz de Projeto

1. **Preferência por Estratificação por Camada (`REP-05`)**:
   - Evitar alocação heterogênea de bits por token na mesma camada (fragmentação de registradores e perda em 2 bits).
   - Utilizar precisão mista por camada (`REP-05`), que mantém formato uniforme dentro de cada kernel de atenção.

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/REP-06-ENTROPY-PRECISION-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Script da Prova**: [`tools/probes/rep06_dynamic_entropy_precision.py`](../../tools/probes/rep06_dynamic_entropy_precision.py)
- **Agente Executor**: Antigravity
