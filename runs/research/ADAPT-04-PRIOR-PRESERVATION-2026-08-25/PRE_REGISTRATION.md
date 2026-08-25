# ADAPT-04 Prior-Preservation Loss (DreamBooth para LLMs) - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: No experimento ADAPT-01A, o aumento do budget de treino no LoKr (5 épocas / 640 passos) melhorou a acurácia de raciocínio matemático (15/32 acertos no GSM8K), mas causou overfitting e regressão de perda no conhecimento geral protegido (a pontuação em QA caiu para 2/16). A técnica de **Prior-Preservation Loss** (Ruiz et al., DreamBooth CVPR 2023), ao intercalar batches da tarefa alvo com batches auto-gerados ou congelados da distribuição geral com ponderação $\lambda \in \{0.2, 0.5\}$, ancora a geometria do espaço latente, impedindo o esquecimento catastrófico e permitindo a convergência da acurácia matemática sem degradação na suite protegida.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Modelo Base**: `Qwen/Qwen3.5-0.8B-Base` (`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`)
* **Hardware**: NVIDIA GeForce RTX 3090 (24GB GDDR6X)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Geometria**: LoKr (`r=8`, `alpha=16`, target all-linear)
* **Função de Perda**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{target}}(\text{GSM8K}) + \lambda \cdot \mathcal{L}_{\text{prior}}(\text{General QA / Anchor Blocks})$$
* **Braços Experimentais**:
  1. `lokr_unregularized_5ep`: 640 passos, $\lambda = 0.0$ (baseline de controle com overfitting).
  2. `lokr_prior_pres_lambda02`: 640 passos, $\lambda = 0.2$.
  3. `lokr_prior_pres_lambda05`: 640 passos, $\lambda = 0.5$.
  4. `lokr_prior_pres_lambda10`: 640 passos, $\lambda = 1.0$.
* **Critérios de Promoção (Gates Falsificáveis)**:
  1. **Retenção Protegida**: $\ge 4/16$ (preservação integral ou superior à baseline não adaptada).
  2. **Acurácia Matemática**: $\ge 14/32$ no GSM8K (sem perder o ganho de raciocínio).
  3. **Término Natural (EOS)**: $\ge 38/48$ de término natural.
