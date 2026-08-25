# BEE-L3 Adaptive MTP Profit Controller - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: Em decodificação especulativa / Multi-Token Prediction (MTP), fixar estaticamente a profundidade de rascunho ($K$) provoca penalidade de latência quando a entropia local do texto se eleva e a taxa de aceitação cai abaixo do limiar de rentabilidade ($\alpha < \alpha_{\text{breakeven}}$). Um controlador adaptativo em malha fechada baseado em janela deslizante de aceitação que ajusta $K \in \{0, 1, 2, 3, 4\}$ em tempo de execução maximiza o speedup global em domínios fáceis ($\alpha \ge 0.75$) e evita perdas de throughput em domínios estocásticos ($\alpha < 0.40$).

---

## 🎯 1. Contrato e Protocolo Experimental

* **Modelo Teórico**:
  $$\text{Speedup}(\alpha, K) = \frac{1 + \alpha + \alpha^2 + \dots + \alpha^K}{1 + K \cdot \gamma}$$
  onde $\gamma = \frac{\text{Custo do Passo Draft}}{\text{Custo do Passo Target}}$.
* **Política de Controle**:
  - $\alpha_{\text{window}} \ge 0.75 \implies K = 4$ (Profundidade Máxima)
  - $0.50 \le \alpha_{\text{window}} < 0.75 \implies K = 2$ (Profundidade Moderada)
  - $0.35 \le \alpha_{\text{window}} < 0.50 \implies K = 1$ (Profundidade Mínima)
  - $\alpha_{\text{window}} < 0.35 \implies K = 0$ (Draft Desabilitado - Pass-through)
* **Condições de Teste**:
  - Simulação estocástica de 1.000 passos com regimes alternados de aceitação ($\alpha_{\text{high}} = 0.85$, $\alpha_{\text{low}} = 0.15$).
  - Comparativo pareado:
    1. `STATIC_MTP_K4`: Profundidade $K=4$ fixa.
    2. `STATIC_MTP_K2`: Profundidade $K=2$ fixa.
    3. `NO_MTP_K0`: Sem decodificação especulativa.
    4. `ADAPTIVE_CONTROLLER`: Controle dinâmico com janela $W=16$.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Proteção em Baixa Previsibilidade**: O throughput no regime de baixa aceitação ($\alpha=0.15$) não pode ser inferior a 95% do baseline sem MTP ($K=0$).
2. **Gate de Eficiência Global**: O speedup médio integrado através de todos os regimes deve superar o melhor MTP estático em pelo menos $\ge 15\%$.
3. **Passagem na Suite de Testes**: 100% de cobertura nos testes unitários do controlador.
