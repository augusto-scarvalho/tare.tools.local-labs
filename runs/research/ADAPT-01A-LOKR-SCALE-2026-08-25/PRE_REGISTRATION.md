# ADAPT-01A LoKr Scaling & Training Budget - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: No experimento preliminar ADAPT-00C, a decomposição por Produto de Kronecker (LoKr) obteve 15/32 acertos exatos no GSM8K em apenas 1 época (128 pares de treino, 359.040 parâmetros), liderando todas as 6 geometrias avaliadas e ficando a apenas 1 acerto do piso de qualificação (16/32). Aumentar o budget de treinamento para 3 épocas (com cosine decay de learning rate e regularização de gradientes) fornecerá a convergência necessária para cruzar o piso de 16/32 e atingir $\ge 40/48$ de término natural (EOS) sem degradar a suite protegida de QA geral.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Modelo Base**: `Qwen/Qwen3.5-0.8B-Base` (revisão `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Geometria**: LoKr (`r=8`, `alpha=16`, target modules `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`)
* **Treinamento**:
  - Braço A (LoKr 1 epoch - baseline de reprodução)
  - Braço B (LoKr 3 epochs - budget escalado)
  - Braço C (LoKr 5 epochs - budget estendido)
* **Critérios de Promoção (Gates Falsificáveis)**:
  1. **Acurácia Alvo**: $\ge 16/32$ (50.0%) de acertos exatos no GSM8K (painel congelado).
  2. **Formato Estrito**: $\ge 24/32$ de emissão do formato `#### <valor>`.
  3. **Término Natural**: $\ge 40/48$ de EOS natural (sem atingir o teto de 192 tokens).
  4. **Retenção Protegida**: Não regredir em relação ao base na suite protegida de 16 itens.
  5. **Comprimento de Resposta**: Mediana de tokens $\le 1.25\times$ a mediana do professor Fable (142.5 tokens).

---

## 📊 2. Baselines de Comparação

- Controle Base (Qwen 0.8B sem adaptação): 4/32 GSM8K, 3/16 Protected QA.
- LoKr 1-epoch (Codex em 24/08): 15/32 GSM8K, 4/16 Protected QA, 35/48 EOS.
- Professor Fable-TC (27B): Mediana de 142.5 tokens no GSM8K.
