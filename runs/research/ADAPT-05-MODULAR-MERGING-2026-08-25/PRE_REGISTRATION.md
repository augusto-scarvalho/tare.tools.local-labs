# ADAPT-05 Modular Skill Composition (TIES-Merging & DARE) - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: A fusão ingênua por média linear de múltiplos adaptadores heterogêneos (`Adapter_Math` e `Adapter_QA`) sofre com interferência de sinais conflitantes de gradiente, degradando o raciocínio matemático. A aplicação de **TIES-Merging** (poda de deltas de baixa magnitude + consenso de sinal) ou **DARE** (*Drop And REscale*) elimina o ruído redundante, permitindo a consolidação dos dois especialistas em um único adaptador unificado que retém $\ge 80\%$ da acurácia matemática ($\ge 14/32$) mantendo a retenção em QA ($\ge 4/16$).

---

## 🎯 1. Contrato e Protocolo Experimental

* **Modelo Base**: `Qwen/Qwen3.5-0.8B-Base` (`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`)
* **Hardware**: NVIDIA GeForce RTX 3090 (24GB VRAM)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Adaptadores de Entrada**:
  - `Adapter 1`: `target_mlp_only` (Math Specialist, 17/32 no GSM8K)
  - `Adapter 2`: `target_attn_only` (QA Specialist, 5/16 no QA)
* **Métodos de Fusão Comparados**:
  1. `NAIVE_LINEAR_AVERAGE`: Média direta ponderada $\frac{A + B}{2}$.
  2. `DARE_MERGE`: Poda aleatória Bernoulli ($p=0.5$) com re-escala $\frac{1}{1-p}$.
  3. `TIES_MERGE`: Trimming dos 30% menores pesos + Consenso de Sinal majoritário.
* **Métricas**:
  1. `merged_math_score`: Acertos no GSM8K (32 problemas).
  2. `merged_qa_score`: Acertos no QA Protegido (16 perguntas).
  3. `composite_score`: Soma combinada dos acertos no modelo fundido.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Preservação Matemática**: Escore GSM8K no modelo fundido $\ge 14/32$.
2. **Gate de Preservação de QA**: Escore na suite protegida $\ge 4/16$.
3. **Superioridade sobre Média Linear**: TIES ou DARE superando a média direta em pelo menos $\ge 15\%$.
