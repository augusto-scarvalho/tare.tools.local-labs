# RSH-01 FibQuant Vector Quantization Simulation - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: As ativações e pesos em modelos de linguagem apresentam densidade centrada em zero com caudas longas (distribuição Gaussiana / Laplaciana). A quantização linear uniforme desperdiça níveis de quantização em valores extremos de baixa probabilidade. A alocação de codebooks não-lineares baseados na série de Fibonacci ($\pm \{0, 1, 2, 3, 5, 8, 13, 21\} / 21$ em 4 bits) aloca alta densidade de representação próximo de zero, reduzindo o erro quadrático médio ($\text{MSE}$) em $\ge 30\%$ e elevando a relação sinal-ruído ($\text{SQNR}$) em $\ge 3.0\text{ dB}$ em relação à grade linear uniforme com os mesmos 4 bits por valor.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Hardware**: NVIDIA GeForce RTX 3090 (24GB VRAM)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Tensores Avaliados**:
  - Matrizes reais de pesos e ativações $1024 \times 1024$ do `Qwen/Qwen3.5-0.8B-Base`.
* **Grades de Quantização Comparadas (4 bits = 16 níveis de código)**:
  1. `UNIFORM_LINEAR_4BIT`: 16 níveis simétricos linearmente espaçados em $[-1.0, +1.0]$.
  2. `FIBONACCI_NONLINEAR_4BIT`: 16 níveis baseados na razão de Fibonacci normalizada.
  3. `LOGARITHMIC_4BIT`: 16 níveis logarítmicos $\pm 2^{-i}$.
  4. `OPTIMAL_LLOYD_MAX_4BIT`: Codebook ótimo iterativo de Lloyd-Max (Referência teórica máxima).
* **Métricas**:
  1. `reconstruction_mse`: Erro quadrático médio de quantização.
  2. `sqnr_db`: Signal-to-Quantization-Noise Ratio em decibéis ($10 \log_{10} \frac{\text{Var}(X)}{\text{MSE}}$).
  3. `cosine_similarity`: Fidelidade direcional do tensor reconstruído.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Redução de Erro de Quantização ($\ge 30\%$ vs Linear)**: $\text{MSE}_{\text{fib}} \le 0.70 \times \text{MSE}_{\text{linear}}$.
2. **Gate de Ganho de SQNR ($\ge 2.5\text{ dB}$)**: $\text{SQNR}_{\text{fib}} - \text{SQNR}_{\text{linear}} \ge 2.5\text{ dB}$.
3. **Fidelidade Direcional**: $\text{Cosine Sim} \ge 0.995$.
