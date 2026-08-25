# RSH-03 KVLinC Residual Compensation - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `REJECTED` — A compensação residual de baixo rank ($r=4$) recuperou apenas **1.62% do erro quadrático da quantização INT4**, refutando a hipótese de cancelamento de ruído e comprovando que o erro de quantização possui espectro isotrópico de rank pleno que não pode ser comprimido via SVD ultra-baixo.

---

## 🎯 1. Resumo Executivo

O experimento testou a viabilidade de cancelar o ruído de quantização INT4 adicionando adaptadores residuais de baixo rank ($C = U V^T$, rank $r=4$, adicionando 0.78% de parâmetros) otimizados via SVD na matriz de erro residual através de [`tools/probes/rsh03_kvlinc_compensation.py`](../../tools/probes/rsh03_kvlinc_compensation.py).

A hipótese de recuperação substancial de fidelidade foi **FALSIFICADA**:
- O erro de quantização $E = W_{\text{orig}} - W_{\text{quant}}$ apresentou distribuição espectralmente plana (ruído branco ortogonal).
- O subespaço de rank $r=4$ capturou apenas uma fração desprezível da energia do resíduo, reduzindo o MSE de saída de **48.09 para 47.31 (recuperação marginal de 1.62%)**, distante do piso de 50.0%.

---

## 📊 2. Tabela de Métricas de Compensação Residual (Matriz $1024 \times 1024$)

| Regime de Avaliação | Rank do Resíduo | Sobrecarga de Parâmetros | $\text{MSE}$ de Saída | Similaridade Cosseno | Recuperação de $\text{MSE}$ | Veredito |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`UNQUANTIZED_FP16`** | — | 0.0% | 0.000000 | 1.00000 | Referência | — |
| **`UNCOMPENSATED_INT4`** | — | 0.0% | 48.090313 | 0.99536 | 0.0% | Baseline |
| **`KVLINC_COMPENSATED`** | **$r=4$** | **0.78%** | **47.313122** | **0.99543** | **+1.62% (FAIL)** | `REJECTED` |

---

## 🔬 3. Implicação Teórica

1. **Rejeição de Corretores Low-Rank para Quantização**:
   - Não utilizar matrizes de baixo rank ($r \le 16$) como corretor pós-hoc de quantização; o ruído de quantização é intrinsecamente de rank alto.
   - Utilizar preservação esparsa de outliers (`SpQR`) ou rotação de Walsh-Hadamard (`QuaRot` / `REP-03`).

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/RSH-03-KVLINC-COMPENSATION-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Script da Prova**: [`tools/probes/rsh03_kvlinc_compensation.py`](../../tools/probes/rsh03_kvlinc_compensation.py)
- **Agente Executor**: Antigravity
