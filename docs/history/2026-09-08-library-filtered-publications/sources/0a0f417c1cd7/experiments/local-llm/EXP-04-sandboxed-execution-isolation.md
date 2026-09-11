# 🧪 Experiment Card: EXP-04 — 3-Process Sandboxed Test Harness Isolation & Egress Proof

- **ID:** `EXP-04`
- **Cluster:** `10 Resource / Assurance research` / `res-exp-isolation`
- **Node Alvo:** `aaaaa` (WSL2 Ubuntu 24.04 / Bubblewrap)
- **Status:** `ACTIVE`
- **Prerequisites:** [ADR-048](file:///C:/projects/tare.tools.os/docs/ADR-048_LOCAL_INFERENCE_SUBSTRATE_AND_AGENT_HARNESS.md)
- **Author:** Antigravity Mediator

---

## 1. Hipótese & Motivação (Hypothesis)

> **Hipótese:** A topologia de 3 processos segregados (Processo 1: Host Orchestrator; Processo 2: `AiderDriver` com egress HTTP restrito a `slop.cpp:8080`; Processo 3: Execução de testes encapsulada via Bubblewrap com `--unshare-net`, `--ro-bind` para toolchains do sistema e workspace efêmero `rw`) impede 100% das tentativas de vazamento de credenciais e comunicação não autorizada com a internet durante a execução de testes gerados por LLM local, mantendo o overhead de inicialização do sandbox $\le 100\text{ms}$ por comando.

---

## 2. Métricas & Variáveis (Metrics)

| Métrica | Descrição | Unidade | Target de Sucesso |
| :--- | :--- | :--- | :--- |
| **`Egress_Leak_Blocked`** | Taxa de bloqueio de tentativas de conexão externa durante testes | % | `100.0%` (Zero vazamentos) |
| **`Secret_Leak_Blocked`** | Taxa de bloqueio de leitura de `.git/config`, `~/.ssh` e variáveis de ambiente | % | `100.0%` |
| **`Sandbox_Overhead_ms`** | Tempo adicional introduzido pelo encapsulamento Bubblewrap | ms | $\le 100\text{ms}$ |
| **`Git_Worktree_Integrity`** | Sucesso em operações `git status`/`git diff` sem colisão de lock | % | `100.0%` |

---

## 3. Baseline de Comparação (Baseline)

- **Baseline:** Execução direta de `pytest` pelo processo do driver no host sem contenção de namespace de rede ou isolamento de sistema de arquivos.

---

## 4. Critérios de Sucesso (Success Criteria)

1. Validação de 10 probes adversariais de injeção de código tentando abrir sockets externos e ler arquivos sensíveis do host.
2. Overhead medido no ciclo de feedback TDD menor que 5% do tempo total de execução do pytest.
3. Conformidade estrita com a Seção 2.1 da [ADR-048](file:///C:/projects/tare.tools.os/docs/ADR-048_LOCAL_INFERENCE_SUBSTRATE_AND_AGENT_HARNESS.md).

---

## 5. Critérios de Engavetamento / Abandono (Abandon Criteria)

- Falha no isolamento de rede (ex: pacotes TCP conseguindo sair do namespace de teste).
- Quebra de comandos normais de teste do pytest dependentes de ferramentas locais padrão (`/bin`, `/usr/bin`).

---

## 6. Plano de Reversão (Reversal Plan)

- Reverter para container OCI leve (Docker / Podman) isolado caso Bubblewrap apresente incompatibilidades de kernel no WSL2.
