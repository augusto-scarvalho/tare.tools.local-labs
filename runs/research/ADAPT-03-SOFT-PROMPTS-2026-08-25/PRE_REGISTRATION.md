# ADAPT-03 Learned Semantic Tokens (Soft Prompts) - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: A injeção de uma sequência compacta de tokens contínuos aprendidos ($P=8$ pseudo-tokens virtuais no espaço de embedding, consumindo $< 16\text{ KB}$ de armazenamento) é suficiente para modular o comportamento do `Qwen/Qwen3.5-0.8B-Base`, induzindo o seguimento de formato estrito (`#### <valor>`) e raciocínio matemático com zero modificação nos pesos dos transformadores e zero risco de catastrophic forgetting no backbone.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Modelo Base**: `Qwen/Qwen3.5-0.8B-Base` (`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`)
* **Hardware**: NVIDIA GeForce RTX 3090 (24GB GDDR6X)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Geometria de Prompt Tuning**:
  - `num_virtual_tokens`: 8 pseudo-tokens contínuos.
  - `token_dim`: 1024 ($d_{\text{model}}$ do Qwen 0.8B).
  - Parâmetros treináveis: $8 \times 1024 = 8.192$ parâmetros (**16.38 KB** em FP16).
  - Budget de treino: 384 passos ($LR = 1\times 10^{-2}$).
* **Métricas**:
  1. `target_correct`: Acertos no painel de 32 problemas GSM8K.
  2. `hash_format_rate`: Proporção de respostas que adotam o padrão `#### <valor>`.
  3. `protected_pass`: Retenção na suite protegida de QA (meta: $\ge 3/16$).
  4. `storage_footprint_kb`: Tamanho físico do artefato gerado.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Formato e Indução ($\ge 75\%$)**: $\ge 24/32$ das respostas adotam a sintaxe canônica `####`.
2. **Gate de Preservação da Base ($\ge 3/16$)**: Manutenção da integridade geral na suite protegida.
3. **Pegada de Memória ($< 32\text{ KB}$)**: Tamanho físico inferior a 32 KB.
