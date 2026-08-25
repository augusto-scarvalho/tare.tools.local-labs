# REP-06 Online Dynamic Precision KV (Entropy-Guided) - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: A incerteza preditiva de cada token durante a geração (medida pela entropia de Shannon do softmax de saída $H(p)$) reflete a importância de informação do seu vetor de Key e Value no contexto futuro. Alocar dinamicamente 2 bits para tokens de baixa entropia ($H < 0.8$), 4 bits para entropia média ($0.8 \le H < 2.0$) e FP16 para tokens de alta entropia ($H \ge 2.0$) reduz a pegada de KV cache em **$\ge 60\%$** (média de $\approx 3.8\text{ bits/token}$) mantendo fidelidade de atenção **$\text{Cosine Sim} \ge 0.995$**.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Hardware**: NVIDIA GeForce RTX 3090 (24GB VRAM)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Configuração de Contexto**: $L=2048$ tokens, $H=16$, $d_{\text{head}}=128$.
* **Políticas Comparadas**:
  1. `STATIC_FP16`: 16.0 bits/valor (Referência).
  2. `STATIC_INT4`: 4.0 bits/valor homogêneo.
  3. `DYNAMIC_ENTROPY_PRECISION`: Triagem por limiares de entropia (INT2 / INT4 / FP16).
* **Métricas**:
  1. `average_bits_per_token`: Média ponderada de bits consumidos por token no cache.
  2. `memory_savings_pct`: Economia de VRAM vs FP16.
  3. `attention_scores_cosine_sim`: Similaridade dos escores de atenção softmax gerados.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Economia de Memória ($\ge 55\%$)**: Bits médios por elemento $\le 7.0$ bits.
2. **Gate de Fidelidade de Atenção ($\ge 0.992$)**: $\text{Cosine Sim} \ge 0.992$.
3. **Superioridade vs INT4 Homogêneo**: Fidelidade superior ao INT4 estático com economia comparável.
