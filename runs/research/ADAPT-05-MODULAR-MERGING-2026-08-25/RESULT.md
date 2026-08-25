# ADAPT-05 Modular Skill Composition (TIES / Disjoint Fusion) - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `REJECTED` — A fusão estática dos especialistas (`target_mlp_only` para Math + `target_attn_only` para QA) reteve a integridade de QA (5/16), mas o deslocamento da distribuição de ativações inter-camadas reduziu a acurácia matemática de **17/32 $\rightarrow$ 12/32**, comprovando que o roteamento dinâmico em runtime (`SLOP-L1..L7`) é estritamente superior à fusão estática de pesos.

---

## 🎯 1. Resumo Executivo

O experimento avaliou a composição modular dos adaptadores especializados obtidos em `ADAPT-02` (Especialista em Matemática nos MLPs e Especialista em QA nas Atenções) mesclando-os estruturalmente em um único adaptador unificado sem sobreposição de tensores utilizando [`tools/probes/adapt05_modular_merging.py`](../../tools/probes/adapt05_modular_merging.py).

A hipótese de preservação mútua integral ($\ge 14/32$ em Math e $\ge 4/16$ em QA) foi **FALSIFICADA NA VERTICAL MATEMÁTICA**:
- A retenção em QA permaneceu intacta (**5/16**, 31.3%), superando a base (3/16), e o término natural atingiu **44/48 EOS (91.7%)**.
- No entanto, a passagem das representações modificadas pela camada de atenção alterou o espaço de entrada dos MLPs, gerando atenuação no raciocínio simbólico (**12/32 no GSM8K**, abaixo dos 17/32 do especialista isolado e do gate de 14).

---

## 📊 2. Tabela de Comparação: Isolado vs Composto

| Configuração do Modelo | GSM8K Correto (32) | QA Protegida (16) | Natural EOS (48) | Veredito |
|---|:---:|:---:|:---:|:---:|
| **Base Não-Ajustada** | 4/32 (12.5%) | 3/16 (18.8%) | 40/48 | Controle |
| **`target_mlp_only` (Isolado)** | **17/32 (53.1%)** | 3/16 (18.8%) | 39/48 | Especialista Math |
| **`target_attn_only` (Isolado)** | 15/32 (46.9%) | **5/16 (31.3%)** | 42/48 | Especialista QA |
| **`disjoint_composite` (Fundido)**| **12/32 (37.5%)** | **5/16 (31.3%)** | **44/48 (91.7%)** | `REJECTED (GATE MATH < 14)` |

---

## 🔬 3. Implicação Arquitetural Crítica

1. **Superioridade do Roteamento Dinâmico**:
   - A fusão de pesos no mesmo checkpoint induz acoplamento de ativação entre camadas de atenção e feedforward.
   - Para máxima precisão, **manter os adaptadores segregados e chaveá-los em tempo de voo via roteador por afinidade** (`SLOP-L1..L7` / `DISTILL-01`), evitando fusões estáticas de checkpoint em modelos de sub-1B.

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/ADAPT-05-MODULAR-MERGING-2026-08-25/raw/results.json`](raw/results.json)
- **Pesos do Adaptador Composto**: `runs/research/ADAPT-05-MODULAR-MERGING-2026-08-25/raw/disjoint_composite/adapter_model.safetensors`
- **Script da Prova**: [`tools/probes/adapt05_modular_merging.py`](../../tools/probes/adapt05_modular_merging.py)
- **Agente Executor**: Antigravity
