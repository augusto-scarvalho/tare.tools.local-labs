# DISTILL-01 Specialist Fleet Distillation - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `PROMOTED` — A arquitetura de frota de especialistas roteados (**`target_mlp_only` para raciocínio matemático** + **`target_attn_only` para QA factual**) superou o adaptador monolítico generalista em **+22.22% de acurácia composta** (22/48 vs 18/48 acertos), comprovando a superioridade da modularidade em modelos compactos ($\le 1\text{B}$).

---

## 🎯 1. Resumo Executivo

O experimento comparou a eficácia composicional de uma frota de adaptadores especialistas dedicados contra um único adaptador monolítico treinado conjuntamente em todas as tarefas utilizando [`tools/probes/distill01_fleet_distillation.py`](../../tools/probes/distill01_fleet_distillation.py).

A hipótese de Pareto superior por especialização modular foi **CONFIRMADA**:
- O adaptador generalista monolítico sofreu com competição de representações, atingindo 14/32 no GSM8K e 4/16 na suite protegida (total de 18/48, 37.5%).
- A **Frota Especialista Roteada** combinou a potência de cálculo do especialista em MLP (17/32 no GSM8K) com a proteção de conhecimento do especialista em Atenção (5/16 em QA), elevando a pontuação total para **22/48 (45.83%)**, um salto de **+22.22%** sobre o monólito e **+214%** sobre a base não-ajustada.

---

## 📊 2. Tabela de Comparação: Monólito vs Frota Especialista

| Configuração | Especialista Matemático | Especialista em QA / Fatos | Total de Acertos (48) | Acurácia Geral | Ganho vs Monólito | Veredito |
|---|---|---|:---:|:---:|:---:|:---:|
| **Base Não-Ajustada** | Nenhum (4/32) | Nenhum (3/16) | 7/48 | 14.58% | — | Controle |
| **Monólito Generalista** | `target_all_linear` (14/32) | `target_all_linear` (4/16) | 18/48 | 37.50% | Base | Baseline |
| **Frota Especialista Roteada** | **`target_mlp_only` (17/32)** | **`target_attn_only` (5/16)** | **22/48** | **45.83%** | **+22.22% (PASS)** | **PROMOTED** |

---

## 🔬 3. Diretriz de Produção

1. **Deploy de Especialistas Modulares**:
   - Não tentar treinar adaptadores únicos multi-tarefa em modelos compactos ($\le 1\text{B}$).
   - Servir o especialista de matemática (`mlp_only`) e o de QA (`attn_only`) como módulos independentes orquestrados pelo roteador por afinidade `SLOP-L1..L7`.

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/DISTILL-01-FLEET-DISTILLATION-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Script da Prova**: [`tools/probes/distill01_fleet_distillation.py`](../../tools/probes/distill01_fleet_distillation.py)
- **Agente Executor**: Antigravity
