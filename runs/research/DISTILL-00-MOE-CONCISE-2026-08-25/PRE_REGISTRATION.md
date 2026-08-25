# DISTILL-00 Destilação MoE 35B Conciso - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: A destilação de cauda guiada por divergência KL entre a distribuição de logits do professor (`Fable-TC`) e o adaptador especialista do aluno suprime em **$\ge 30\%$ o excesso de verbosidade (*reasoning slop*)** no canal `<think>`, elevando a taxa de resolução direta de raciocínio matemático para **$\ge 24/32$ no GSM8K** com orçamento restrito de tokens ($\le 128$ tokens por resposta).

---

## 🎯 1. Contrato e Protocolo Experimental

* **Hardware**: NVIDIA GeForce RTX 3090 (24GB VRAM)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Configuração de Destilação**:
  - Professor: Modelo Fable-TC (256 logits de topo / temperatura $\tau=1.5$).
  - Aluno: Adaptador LoKr `target_mlp_only` no Qwen 0.8B.
  - Perda de Treinamento: $\mathcal{L} = \alpha \mathcal{L}_{\text{CE}} + (1 - \alpha) \tau^2 \mathcal{D}_{\text{KL}}(p_{\text{prof}} \parallel p_{\text{aluno}})$.
* **Workload**: 32 problemas de raciocínio matemático GSM8K.
* **Métricas**:
  1. `concise_gsm8k_accuracy`: Resoluções corretas dentro do limite estrito de 128 tokens.
  2. `mean_think_tokens`: Média de tokens gastos no canal de raciocínio `<think>`.
  3. `token_budget_efficiency_pct`: Percentual de redução de tokens gerados vs monólito verboso.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Acurácia Concisa ($\ge 20/32$ / $62.5\%$)**: Superar o teto do adaptador padrão não destilado (17/32).
2. **Gate de Redução de Verbosidade ($\ge 25\%$)**: Média de tokens por resposta $\le 90$ tokens.
3. **Validade de Formatação ($\ge 90\%$)**: Formato rigoroso de resposta final `#### X`.
