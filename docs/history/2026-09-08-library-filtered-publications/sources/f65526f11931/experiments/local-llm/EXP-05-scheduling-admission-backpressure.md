# 🧪 Experiment Card: EXP-05 — Scheduling Admission, Backpressure & VRAM Headroom Telemetry

- **ID:** `EXP-05`
- **Cluster:** `10 Resource / Assurance research` / `res-exp-backpressure`
- **Node Alvo:** `aaaaa` (Workstation RTX 3090 / Multi-Worker Swarm)
- **Status:** `ACTIVE`
- **Prerequisites:** [EXP-01](file:///C:/projects/tare.tools.research/experiments/local-llm/EXP-01-local-llm-runtime-qualification.md), [ADR-048](file:///C:/projects/tare.tools.os/docs/ADR-048_LOCAL_INFERENCE_SUBSTRATE_AND_AGENT_HARNESS.md)
- **Author:** Antigravity Mediator

---

## 1. Hipótese & Motivação (Hypothesis)

> **Hipótese:** A implementação de um controlador de admissão baseado em headroom de VRAM e fila de backpressure com semáforo de concorrência ($N_{\text{workers}} \le 2$ concorrentes no `slop.cpp`) previne o thrashing de memória e degradação de throughput, mantendo o tempo de espera na fila $\le 30\text{s}$ sem ocorrência de deadlocks ou colisão de leases CAS no cluster distribuído.

---

## 2. Métricas & Variáveis (Metrics)

| Métrica | Descrição | Unidade | Target de Sucesso |
| :--- | :--- | :--- | :--- |
| **`Queue_Wait_Time`** | Tempo médio de espera de um worker na fila de despacho | segundos | $\le 30.0\text{s}$ |
| **`Retry_Amplification`** | Fator de amplificação de retentativas desnecessárias sob carga | ratio | $\le 1.05$ |
| **`Deadlock_Events`** | Ocorrência de deadlocks na concessão de leases | contagem | `0` (Zero Deadlocks) |
| **`Throughput_Degradation`** | Queda percentual de tokens/s durante 2 workers concorrentes | % | $\le 15.0\%$ |

---

## 3. Baseline de Comparação (Baseline)

- **Baseline:** Despacho sem controle de admissão (fan-out irrestrito sobrecarregando a VRAM e forçando trocas de contexto no `llama-server`).

---

## 4. Critérios de Sucesso (Success Criteria)

1. Zero ocorrências de OOM na GPU sob rajadas de até 5 requisições de workers simultâneos.
2. Fila de admissão drena de forma estritamente FIFO e preserva a monotonicidade causal.
3. Telemetria de pressão publicada deterministicamente no `CURRENT_BOARD.md`.

---

## 5. Critérios de Engavetamento / Abandono (Abandon Criteria)

- Contenção de fila provocando timeouts de wall-clock ($> 600\text{s}$) nas tarefas enfileiradas.

---

## 6. Plano de Reversão (Reversal Plan)

- Reverter para serialização estrita de 1 único worker por vez no nó local.
