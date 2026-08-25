# RSH-03 KVLinC Residual Compensation - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: O erro de quantização INT4 em matrizes lineares $W_{\text{quant}}$ pode ser modelado como uma perturbação estocástica com componentes sistemáticos de baixa dimensão. A introdução de uma matriz residual de rank ultra-baixo ($C = U V^T$ com rank $r=4$, adicionando apenas $0.78\%$ de parâmetros em FP16) treinado em ativações de calibração recupera $\ge 60\%$ do erro quadrático médio ($\text{MSE}$), elevando a similaridade direcional de saída para $\ge 0.998$.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Hardware**: NVIDIA GeForce RTX 3090 (24GB VRAM)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Matriz Avaliada**: Projeções lineares $1024 \times 1024$ do `Qwen/Qwen3.5-0.8B-Base`.
* **Regimes Comparados**:
  1. `UNQUANTIZED_FP16`: Referência sem perdas.
  2. `UNCOMPENSATED_INT4`: Matriz quantizada em INT4 simétrico por blocos de 32.
  3. `KVLINC_COMPENSATED_INT4`: $W_{\text{quant}} + U V^T$ ($r=4$, 8.192 parâmetros adicionais / 16 KB).
* **Métricas**:
  1. `reconstruction_mse`: Erro quadrático médio de saída ($y = W x$).
  2. `mse_recovery_pct`: Percentual do erro de quantização eliminado pela compensação residual.
  3. `output_cosine_sim`: Similaridade de cosseno do vetor de ativação de saída.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Recuperação de MSE ($\ge 50\%$)**: Eliminação de pelo menos 50% do erro quadrático da quantização.
2. **Gate de Fidelidade ($\ge 0.998$)**: $\text{Cosine Sim} \ge 0.998$ no vetor de ativação de saída.
3. **Pegada de Parâmetros ($\le 1.0\%$)**: O overhead de parâmetros residuais deve ser inferior a 1.0% da matriz original.
