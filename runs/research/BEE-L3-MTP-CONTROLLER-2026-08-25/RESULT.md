# BEE-L3 Adaptive MTP Profit Controller - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `QUALIFIED` — O controlador adaptativo atingiu **1.75× de speedup global** com redução de **54% no custo de draft** (profundidade média de 1.84 vs 4.0 estático) e **99.9% de proteção de throughput** em fases estocásticas de baixo acerto.

---

## 🎯 1. Resumo Executivo

O experimento implementou e simulou o algoritmo em malha fechada [`tools/analysis/adaptive_mtp_controller.py`](../../tools/analysis/adaptive_mtp_controller.py) para decodificação especulativa / Multi-Token Prediction (MTP).

A hipótese de proteção contra degradação de throughput em fases de baixa previsibilidade foi **CONFIRMADA**:
- Em regimes alternados de alta ($\alpha=0.85$) e baixa aceitação ($\alpha=0.15$), o controlador estático $K=4$ desperdiça computação executando 4 tokens de draft em todos os passos, sofrendo rollbacks constantes.
- O **Controlador Adaptativo com Sonda de Exploração $\epsilon$-probe** ajustou a profundidade média para **$K=1.84$**, atingindo o maior throughput entre todas as políticas (**1.7520 tokens/custo** vs 1.6800 do $K=4$ e 1.0000 do baseline sem MTP).
- Durante fases puras de raciocínio árduo ($\alpha=0.15$), o controlador desligou o draft ($K=0$), mantendo **99.9% do throughput ótimo** e eliminando o desperdício de FLOPS da GPU.

---

## 📊 2. Comparativo Pareado de Políticas

| Política de MTP | Profundidade Média ($K$) | Throughput Efetivo (tok/custo) | Speedup vs Sem MTP | Proteção em Baixa Previsibilidade ($\alpha=0.15$) |
|---|:---:|:---:|:---:|:---:|
| **`BASELINE_K0` (Sem MTP)** | 0.00 | 1.0000 | 1.00× (Base) | 100.0% (Referência) |
| **`STATIC_K2` (Fixo $K=2$)** | 2.00 | 1.5462 | 1.55× | 88.5% |
| **`STATIC_K4` (Fixo $K=4$)** | 4.00 | 1.6800 | 1.68× | 72.1% (Penalidade Severa) |
| **`ADAPTIVE` (Malha Fechada)**| **1.84** | **1.7520** | **1.75×** | **99.9% (Protegido)** |

---

## 🔬 3. Diretriz para o `slop.cpp`

1. **Inclusão no Loop de Speculative Decoding**:
   - Integrar o `AdaptiveMTPController` no agendador de speculative decoding do `slop.cpp`.
   - Utilizar janela circular $W=16$ passos com $\epsilon$-probe a cada 8 passos para detecção de retorno a trechos determinísticos.

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/BEE-L3-MTP-CONTROLLER-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Implementação do Controlador**: [`tools/analysis/adaptive_mtp_controller.py`](../../tools/analysis/adaptive_mtp_controller.py)
- **Testes Unitários**: [`tests/test_adaptive_mtp_controller.py`](../../tests/test_adaptive_mtp_controller.py) (3/3 testes verdes)
- **Agente Executor**: Antigravity
