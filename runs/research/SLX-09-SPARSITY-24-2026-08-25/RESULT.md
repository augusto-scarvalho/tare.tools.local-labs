# SLX-09 Sparsidade Estruturada 2:4 Ampere - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `REJECTED` — A poda 2:4 zero-shot ponderada por ativações (Wanda) reduziu o erro em **87.7%** sobre a poda de magnitude, mas manteve Cosine Sim de 0.777 (abaixo do limiar de 0.90), confirmando que a aceleração 2:4 em hardware exige calibração iterativa de 2ª ordem (SparseGPT) ou fine-tuning esparso.

---

## 🎯 1. Resumo Executivo

O experimento avaliou o impacto de aplicar poda semi-estruturada 2:4 nos Tensor Cores da RTX 3090 (`sm_86`) sobre os 186 módulos lineares do `Qwen/Qwen3.5-0.8B-Base`.

A hipótese de preservação direta de logits ($\text{Cosine Sim} \ge 0.90$) em poda pós-treinamento de um único disparo foi **FALSIFICADA**:
- A poda 2:4 por magnitude pura colapsou a saída do modelo ($\text{Cosine Sim} = 0.233$, $\text{MSE} = 16.88$).
- O método **Wanda** (pesos ponderados pela norma $L_2$ das ativações de calibração) melhorou drasticamente a retenção ($\text{Cosine Sim} = 0.777$, $\text{MSE} = 2.08$, ganho de **87.7%** no MSE), com **100.0% de conformidade estrutural 2:4**.
- Contudo, a distorção residual ainda é proibitiva para inferência sem perda de qualidade em regime *zero-shot*.

---

## 📊 2. Tabela de Comparação de Poda 2:4

| Método de Poda 2:4 | Conformidade 2:4 | Logits Cosine Sim | Logits MSE | Redução de Erro vs Magnitude | Veredito |
|---|:---:|:---:|:---:|:---:|:---:|
| **`DENSE_BASELINE`** | — | 1.0000 | 0.0000 | Referência | Controle |
| **`MAGNITUDE_2_4`** | 100.0% | 0.2334 | 16.8750 | 0.0% | `REJECTED` |
| **`WANDA_2_4` (Ativações)** | **100.0%** | **0.7773** | **2.0781** | **+87.7% (PASS)** | `REJECTED (GATE SIM < 0.90)` |

---

## 🔬 3. Diretriz para o `slop.cpp`

1. **Rejeição de Poda Zero-Shot**:
   - Não converter matrizes densas para esparsas 2:4 no carregamento sem pesos pré-calibrados por Hessian/SparseGPT ou fine-tuning esparso prévio.
2. **Revisitação Futura (Faixa D)**:
   - Reabrir a linha de sparsidade 2:4 caso seja introduzido um pipeline de calibração com inversão de Hessian em bloco (SparseGPT).

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/SLX-09-SPARSITY-24-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Script da Prova**: [`tools/probes/slx09_sparsity_oracle.py`](../../tools/probes/slx09_sparsity_oracle.py)
- **Agente Executor**: Antigravity
