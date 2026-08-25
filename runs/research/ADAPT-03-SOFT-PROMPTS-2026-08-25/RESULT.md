# ADAPT-03 Learned Semantic Tokens (Soft Prompts) - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `REJECTED` — Os tokens semânticos contínuos (8 pseudo-tokens, 8.192 parâmetros / 16 KB) atingiram **15/32 no GSM8K e 84.4% de conformidade de formato**, mas induziram colapso modal na suite geral (**0/16 em QA protegida**), comprovando que soft prompts prefixados sequestram a rota de atenção de tarefas genéricas.

---

## 🎯 1. Resumo Executivo

O experimento avaliou o método de Prompt Tuning injetando 8 pseudo-tokens contínuos aprendidos ($8 \times 1024 = 8.192$ parâmetros em FP16) no embedding do `Qwen/Qwen3.5-0.8B-Base` utilizando o script [`tools/probes/adapt03_soft_prompts.py`](../../tools/probes/adapt03_soft_prompts.py).

A hipótese de modulação comportamental com preservação passiva foi **FALSIFICADA**:
- **Capacidade de Indução de Formato**: Com apenas 16 KB de armazenamento, os 8 tokens virtuais direcionaram o modelo base para **15/32 (46.9%) de acertos matemáticos** e **84.4% (27/32) de aderência à sintaxe estrita `#### <valor>`**.
- **Colapso em Domínio Cruzado**: Como o prefixo virtual foi anexado estaticamente a todas as entradas, a atenção do modelo foi monopolizada pelo modo de resolução matemática, levando a **0/16 de acertos na suite protegida de QA** (violação total do gate de retenção).

---

## 📊 2. Tabela de Métricas do Experimento

| Métrica | Valor Observado | Meta / Gate | Veredito |
|---|:---:|:---:|:---:|
| **GSM8K Correto (32)** | **15/32 (46.9%)** | $\ge 12/32$ | **PASS** |
| **Conformidade de Formato (`####`)** | **27/32 (84.4%)** | $\ge 75.0\%$ | **PASS** |
| **QA Geral Protegida (16)** | **0/16 (0.0%)** | $\ge 3/16$ | **FAIL (KILL GATE)** |
| **Término Natural (EOS)** | 39/48 | $\ge 36/48$ | **PASS** |
| **Pegada Física de Armazenamento** | **16.00 KB** | $\le 32.0\text{ KB}$ | **PASS** |
| **Parâmetros Treináveis** | **8.192** (0.001% do modelo) | — | — |

---

## 🔬 3. Diretriz para o `slop.cpp`

1. **Restrição de Uso de Soft Prompts**:
   - Nunca aplicar soft prompts de maneira global ou estática no servidor de inferência.
   - Restringir soft prompts a requisições com tag explícita de habilidade (`skill="math_solver"`), combinando com roteamento de adaptadores (`SLOP-L1..L7`).

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/ADAPT-03-SOFT-PROMPTS-2026-08-25/raw/results.json`](raw/results.json)
- **Pesos do Soft Prompt**: `runs/research/ADAPT-03-SOFT-PROMPTS-2026-08-25/raw/adapter/adapter_model.safetensors`
- **Script da Prova**: [`tools/probes/adapt03_soft_prompts.py`](../../tools/probes/adapt03_soft_prompts.py)
- **Agente Executor**: Antigravity
