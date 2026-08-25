# RSH-01 FibQuant Vector Quantization - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `REJECTED` — Codebooks não-lineares baseados na série de Fibonacci apresentaram **92.45% de aumento no erro quadrático médio ($\text{MSE} = 0.00849$ vs $0.00441$)** e queda de **2.84 dB na relação sinal-ruído** em relação à grade linear uniforme, refutando a hipótese de ganho por espaçamento geométrico estático.

---

## 🎯 1. Resumo Executivo

O experimento comparou a eficácia de reconstrução de codebooks de quantização vetorial não-linear (Fibonacci e Logarítmico) contra a grade linear uniforme padrão em tensores de ativação com cauda pesada utilizando [`tools/probes/rsh01_fibquant_simulation.py`](../../tools/probes/rsh01_fibquant_simulation.py).

A hipótese de superioridade do codebook de Fibonacci foi **FALSIFICADA**:
- O escalonamento local por blocos ($N=32$) já normaliza os tensores para $[-1, +1]$.
- A série de Fibonacci concentrou níveis excessivamente próximos de zero ($0.047, 0.095$), criando lacunas desproporcionais entre valores intermediários ($0.38$ a $0.62$ e $0.62$ a $1.0$).
- Elementos moderados sofreram erro severo de aproximação, derrubando o SQNR de **20.84 dB (Linear) para 18.00 dB (Fibonacci)** e elevando o MSE em quase o dobro.

---

## 📊 2. Tabela de Comparação de Codebooks (4 Bits / 16 Níveis)

| Codec / Codebook | Níveis | $\text{MSE}$ | $\text{SQNR (dB)}$ | Similaridade Cosseno | Veredito |
|---|:---:|:---:|:---:|:---:|:---:|
| **`UNIFORM_LINEAR_4BIT`** | 16 | **0.004412** | **20.84 dB** | **0.99591** | **SUPERIOR (PADRÃO)** |
| **`FIBONACCI_NONLINEAR_4BIT`** | 16 | 0.008491 | 18.00 dB (-2.84 dB) | 0.99206 | `REJECTED (FAIL MSE)` |
| **`LOGARITHMIC_4BIT`** | 16 | 0.017449 | 14.87 dB (-5.97 dB) | 0.98365 | `REJECTED` |

---

## 🔬 3. Diretriz para o `slop.cpp`

1. **Descarte de Codebooks de Fibonacci**:
   - Manter a quantização por blocos baseada em grades lineares uniformes (`Q4_K` / `IQ4_NL` com tabelas ótimas por densidade empírica), sem adoção de funções geométricas estáticas de Fibonacci.

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/RSH-01-FIBQUANT-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Script da Prova**: [`tools/probes/rsh01_fibquant_simulation.py`](../../tools/probes/rsh01_fibquant_simulation.py)
- **Agente Executor**: Antigravity
