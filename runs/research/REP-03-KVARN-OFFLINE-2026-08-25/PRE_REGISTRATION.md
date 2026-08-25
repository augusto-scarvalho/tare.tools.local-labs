# REP-03 KVarN Offline Codec (Hadamard Outlier Suppression) - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: A quantização direta em baixa precisão (INT4 / INT2) dos tensores de Key e Value no KV cache sofre com picos pontuais de magnitude (*outliers de ativação*), que esticam a faixa de escala min/max e causam distorção no restante dos elementos. A aplicação de uma rotação ortogonal exata de Walsh-Hadamard ($H_{128}$) nos vetores $K$ e $V$ antes da quantização dispersa a energia dos outliers uniformemente em todas as 128 dimensões da cabeça, reduzindo o erro quadrático médio ($\text{MSE}$) de atenção em $\ge 50\%$ em comparação com a quantização direta não-rotacionada.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Modelo Base**: `Qwen/Qwen3.5-0.8B-Base` (`head_dim=128`, $H=16$)
* **Hardware**: NVIDIA GeForce RTX 3090 (24GB VRAM)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Políticas de Codec KV**:
  1. `UNQUANTIZED_FP16`: Referência exata de controle.
  2. `DIRECT_INT4`: Quantização uniforme simétrica INT4 por bloco de 32 elementos sem rotação.
  3. `KVARN_HADAMARD_INT4`: Rotação rápida de Walsh-Hadamard $H_{128} \times K$ seguida de quantização INT4.
  4. `KVARN_HADAMARD_INT2`: Rotação de Walsh-Hadamard seguida de quantização INT2.
* **Métricas**:
  1. `kv_reconstruction_mse`: Erro quadrático médio de reconstrução dos tensores $K$ e $V$.
  2. `attention_scores_cosine_sim`: Similaridade de cosseno nos mapas de atenção softmax gerados.
  3. `mse_reduction_pct`: Percentual de redução do MSE em relação à quantização direta.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Supressão de Outliers ($\ge 50\%$ Redução de MSE)**: O codec Hadamard INT4 deve reduzir o MSE em pelo menos 50% vs Direct INT4.
2. **Gate de Fidelidade de Atenção**: $\text{Cosine Sim} \ge 0.99$ nos mapas de atenção em $L=2048$.
3. **Economia de Memória de KV ($\ge 70\%$)**: Pegada física $\le 30\%$ do FP16.
