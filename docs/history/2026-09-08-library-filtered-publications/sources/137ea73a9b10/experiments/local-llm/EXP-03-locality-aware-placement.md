# 🧪 Experiment Card: EXP-03 — Locality-Aware Placement & Cost-to-Trust Dispatcher

- **ID:** `EXP-03`
- **Cluster:** `10 Resource / Assurance research` / `res-exp-locality`
- **Node Alvo:** `Acer-Augusto` (Console/Router) $\to$ `aaaaa` (Workstation RTX 3090) vs Cloud $T_1$
- **Status:** `ACTIVE`
- **Prerequisites:** [EXP-01](file:///C:/projects/tare.tools.research/experiments/local-llm/EXP-01-local-llm-runtime-qualification.md), [ADR-048](file:///C:/projects/tare.tools.os/docs/ADR-048_LOCAL_INFERENCE_SUBSTRATE_AND_AGENT_HARNESS.md)
- **Author:** Antigravity Mediator

---

## 1. Hipótese & Motivação (Hypothesis)

> **Hipótese:** O roteamento dinâmico de tarefas de implementação e refatoração de código com prioridades P3 e P4 para o substrato local no nó `aaaaa` (respeitando o orçamento determinístico $N_{\text{total}} \le 5$ e failover para $T_1$ sob saturação de hash de erro ou timeout de 600s) zera os custos de tokens para $\ge 70\%$ do backlog rotineiro sem degradar a taxa final de aprovação de auditoria em comparação com a execução direta em nuvem.

---

## 2. Métricas & Variáveis (Metrics)

| Métrica | Descrição | Unidade | Target de Sucesso |
| :--- | :--- | :--- | :--- |
| **`Incremental_Token_Cost`** | Gasto adicional em dólares por tarefa concluída | USD ($) | `$0.00` em $\ge 70\%$ das tarefas |
| **`Pass_At_1_Local`** | Taxa de sucesso no primeiro turno de testes unitários locais | % | $\ge 65.0\%$ |
| **`Repair_Convergence_Rate`** | Taxa de sucesso nos reparos $N \le 4$ antes do failover | % | $\ge 80.0\%$ |
| **`Cloud_Failover_Rate`** | Frequência de acionamento do fallback Cloud $T_1$ | % | $\le 20.0\%$ |
| **`Mean_Wall_Clock`** | Tempo médio de ponta a ponta da execução da tarefa | segundos | $\le 180.0\text{s}$ |

---

## 3. Baseline de Comparação (Baseline)

- **Baseline:** Roteamento 100% Cloud Frontier ($T_1$ — Claude 3.5/3.7 / GPT-5 / Gemini Pro) para todas as tarefas de código.
  - Custo médio por tarefa: ~$0.15 a $0.45.
  - Latência média de cold start API: 3.5s.

---

## 4. Critérios de Sucesso (Success Criteria)

1. Custo zero incremental comprovado em $\ge 70\%$ dos jobs executados.
2. Zero violações do limite rígido $N_{\text{total}} \le 5$ tentativas locais.
3. Failover determinístico e logging terminal sem travamento do orquestrador quando acionado.

---

## 5. Critérios de Engavetamento / Abandono (Abandon Criteria)

- Taxa de failover para nuvem $> 50\%$, evidenciando capacidade insuficiente do modelo local para tarefas P3/P4.
- Latência de execução local ultrapassando frequentemente o teto de wall-clock de 600s.

---

## 6. Plano de Reversão (Reversal Plan)

- Reverter a política de roteamento no `relay_mesh.py` para despacho exclusivo em nuvem.
