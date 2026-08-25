# DISTILL-01 Specialist Fleet Distillation - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: Em modelos compactos ($\le 1\text{B}$), a destilação de uma frota de adaptadores especialistas dedicados (*Math Specialist*, *Format/JSON Specialist*, *Factual QA Specialist*) gerados a partir do professor 27B supera em $\ge 30\%$ o desempenho de um adaptador generalista monolítico, atingindo Pareto superior quando acoplados ao roteador por afinidade (`SLOP-L1..L7`).

---

## 🎯 1. Contrato e Protocolo Experimental

* **Modelo Base**: `Qwen/Qwen3.5-0.8B-Base` (`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`)
* **Hardware**: NVIDIA GeForce RTX 3090 (24GB VRAM)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Especialistas da Frota**:
  1. `SPECIALIST_MATH`: Especializado em raciocínio matemático via `mlp_only` (`gate, up, down`).
  2. `SPECIALIST_QA`: Especializado em QA factual via `attn_only` (`q, k, v, o`).
  3. `MONOLITH_GENERALIST`: Adaptador único treinado conjuntamente em todas as tarefas.
* **Métricas**:
  1. `fleet_math_score`: Desempenho no painel GSM8K (32 problemas).
  2. `fleet_qa_score`: Desempenho no painel protegido de QA (16 perguntas).
  3. `composite_fleet_efficiency`: Soma combinada dos acertos da frota roteada vs monólito.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Superioridade da Frota ($\ge 20\%$ Ganho Composto)**: O escore conjunto da frota roteada deve superar o monólito generalista em pelo menos 20%.
2. **Preservação de Retenção Mútua**: O especialista em QA deve atingir $\ge 5/16$ enquanto o especialista matemático atinge $\ge 15/32$.
3. **Passagem nos Gates Comportamentais**: Zero loops infinitos nos especialistas.
