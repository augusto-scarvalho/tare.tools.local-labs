# 🧪 Registro Mestre de Experimentos & Benchmarks Empíricos

Este diretório abriga todos os ensaios técnicos, benchmarks de hardware, testes de estresse de LLMs locais e validações de hipóteses arquiteturais do ecossistema TARE, em conformidade com as diretrizes da **ADR-051** e **ADR-052**.

---

## 📋 Tabela Central de Ensaios & Vereditos

| ID | Título do Experimento | Substrato / Hardware | Data | Veredito | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`EXP-01`** | Qualificação de Runtime LLM Local (slop.cpp) | RTX 3090 @ `aaaaa` | 2026-08-16 | `ADOPT` | Concluído |
| **`EXP-02`** | Retenção de Thinking & KV-Cache em Long Context | RTX 3090 @ `aaaaa` | 2026-08-16 | `ADOPT` | Concluído |
| **`EXP-03`** | Locality-Aware Task Placement & VRAM Tiering | Cluster Local + API | 2026-08-16 | `ADOPT` | Concluído |
| **`EXP-04`** | Isolamento de Execução em Sandbox de Processos | Windows Job Objects | 2026-08-16 | `ADOPT` | Concluído |
| **`EXP-05`** | Controle de Admissão & Backpressure sob Carga | Event Loop AsyncIO | 2026-08-16 | `ADOPT` | Concluído |

---

## 📐 Padrão Enxuto para Novos Experimentos

Para propor ou registrar um novo experimento:
1. Copie o template oficial em [`docs/templates/EXP-template.md`](file:///C:/projects/tare.tools.research/docs/templates/EXP-template.md).
2. Salve no formato `experiments/<categoria>/EXP-XXX-slug-do-experimento.md`.
3. Registre o identificador e o veredito na tabela acima.
4. Mantenha os relatórios objetivos com dados brutos, gráficos/tabelas e um veredito claro (`ADOPT`, `ADAPT` ou `RETIRE`). Zero hipertrofia burocrática!
