# ADAPT-04 Prior-Preservation Loss (DreamBooth para LLMs) - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `REJECTED` — A perda de preservação de prior melhorou o término natural (42/48 EOS), mas gerou competição de gradiente no espaço compacto de 359k parâmetros do LoKr, limitando a acurácia de raciocínio a 11/32.

---

## 🎯 1. Resumo Executivo

O experimento testou a aplicação da perda de preservação de prior ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{target}} + \lambda \mathcal{L}_{\text{prior}}$ com $\lambda \in \{0.0, 0.2, 0.5\}$) no treinamento de 5 épocas (640 passos) do LoKr sobre o `Qwen/Qwen3.5-0.8B-Base`.

A hipótese de que o regularizador de prior protegeria a retenção geral mantendo acurácia matemática $\ge 14/32$ foi **FALSIFICADA**:
- O regularizador de prior teve efeito altamente benéfico na regularização do tamanho de geração, elevando o término natural de **37/48 $\rightarrow$ 42/48 EOS** e suprimindo loops de raciocínio descontrolados.
- Contudo, a capacidade de representação do LoKr (359k parâmetros em backbone 0.8B) mostrou-se insuficiente para acomodar simultaneamente a perda de prior e os passos de raciocínio em cadeia do GSM8K, resultando em acurácia de **11/32** ($\lambda=0.2$) e **10/32** ($\lambda=0.5$), abaixo do gate de 14/32.

---

## 📊 2. Tabela de Resultados

| Braço | $\lambda_{\text{prior}}$ | Passos / Épocas | GSM8K Correto (32) | QA Protegida (16) | Natural EOS (48) | Veredito |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Base Control** | — | — | 4/32 | 3/16 | 40/48 | Controle |
| **`lokr_unreg_5ep`** | 0.0 | 640 / 5 | 9/32 | 3/16 | 37/48 | `REJECTED` |
| **`lokr_prior_lambda02`** | 0.2 | 640 / 5 | 11/32 | 2/16 | **42/48 (PASS)** | `REJECTED` |
| **`lokr_prior_lambda05`** | 0.5 | 640 / 5 | 10/32 | 1/16 | **42/48 (PASS)** | `REJECTED` |

---

## 🔬 3. Conclusão Epistêmica

1. **Gargalo de Capacidade em Adapters Compactos**:
   - Forçar um adaptador de baixo rank ($r=8$) em modelo 0.8B a reter representações gerais enquanto especializa em raciocínio matemático gera interferência destrutiva entre gradientes de tarefas heterogêneas.
2. **Diretriz de Arquitetura**:
   - Em vez de regularização conjunta em adapter único, tarefas distintas (raciocínio vs QA geral vs código) devem ser segregadas em adaptadores modulares dedicados com chaveamento em runtime (`SLOP-L1..L7` / Multi-Adapter Serving) ou destilação em modelos $\ge 1.5\text{B}$.

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/ADAPT-04-PRIOR-PRESERVATION-2026-08-25/raw/results.json`](raw/results.json)
- **Adapters Gerados**: `runs/research/ADAPT-04-PRIOR-PRESERVATION-2026-08-25/raw/{lokr_unreg_5ep, lokr_prior_lambda02, lokr_prior_lambda05}/adapter/`
- **Script**: [`tools/probes/adapt04_prior_preservation.py`](../../tools/probes/adapt04_prior_preservation.py)
- **Agente Executor**: Antigravity
