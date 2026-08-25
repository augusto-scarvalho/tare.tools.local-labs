# ADAPT-02 Hybrid Module Targeting - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: A escolha de quais sub-módulos lineares são adaptados em PEFT determina o balanço entre capacidade de raciocínio relacional e retenção de conhecimento factual. Adaptar exclusivamente os módulos de atenção (`q_proj, k_proj, v_proj, o_proj`) preserva o conhecimento factual armazenado nos MLPs com menos parâmetros ($~140\text{k}$ vs $359\text{k}$), enquanto adaptar exclusivamente os MLPs (`gate_proj, up_proj, down_proj`) altera o mapeamento de conhecimento. Este bakeoff compara 4 estratégias de targeting em LoKr sob o mesmo budget de 384 passos no `Qwen/Qwen3.5-0.8B-Base`.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Modelo Base**: `Qwen/Qwen3.5-0.8B-Base` (`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`)
* **Hardware**: NVIDIA GeForce RTX 3090 (24GB GDDR6X)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Geometria**: LoKr (`r=8`, `alpha=16`, 3 épocas = 384 passos, $LR = 2\times 10^{-4}$)
* **Braços Experimentais**:
  1. `TARGET_ALL_LINEAR`: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (359.040 parâmetros).
  2. `TARGET_ATTN_ONLY`: `q_proj, k_proj, v_proj, o_proj` (~138.240 parâmetros).
  3. `TARGET_MLP_ONLY`: `gate_proj, up_proj, down_proj` (~220.800 parâmetros).
  4. `TARGET_QV_GATE`: `q_proj, v_proj, gate_proj` (~122.880 parâmetros).
* **Critérios de Promoção (Gates Falsificáveis)**:
  1. **Eficiência de Parâmetros**: Identificar se `ATTN_ONLY` ou `QV_GATE` alcançam $\ge 90\%$ da acurácia do `ALL_LINEAR` com $< 50\%$ dos parâmetros treináveis.
  2. **Retenção de Conhecimento Geral**: Preservação $\ge 3/16$ na suite protegida de QA.
  3. **Término Natural**: $\ge 38/48$ de término natural (sem loops de geração).
