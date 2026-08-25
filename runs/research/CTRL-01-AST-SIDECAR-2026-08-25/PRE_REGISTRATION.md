# CTRL-01 ControlNet / AST Grammar Sidecar - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: A filtragem em tempo real de tokens através de um analisador de gramática livre de contexto (CFG / AST Sidecar) durante o passo de amostragem elimina $100\%$ dos erros sintáticos de parsing em saídas estruturadas (JSON / Python AST), com overhead de validação $\le 200\text{ µs}$ por token na CPU e zero degradação de throughput.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Ambiente**: Python 3.11 / Host Local
* **Arquitetura do AST Sidecar**:
  - `GrammarStateTracker`: Mantém a pilha de transições de estado para delimitadores `{}`, `[]`, `""`, `:`, `,`.
  - `LogitConstraintMask`: Anula logits de tokens que violariam a gramática no estado atual.
* **Workloads Avaliados**: 50 gerações de estruturas JSON aninhadas e blocos de código com ruído proposital de amostragem.
* **Métricas**:
  1. `ast_parsing_validity_pct`: Percentual de saídas que passam em `json.loads` / `ast.parse` sem erro.
  2. `sidecar_overhead_us_per_token`: Tempo médio de validação e máscara de logit por token.
  3. `syntax_error_prevention_rate_pct`: Taxa de interceptação de anomalias sintáticas.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Validade Sintática ($100\%$)**: 50 de 50 saídas com parsing válido garantido.
2. **Gate de Overhead ($\le 500\text{ µs}$)**: Latência média de validação por token inferior a 500 µs.
3. **Zero Falsos Bloqueios**: Não bloquear continuações sintaticamente válidas.
