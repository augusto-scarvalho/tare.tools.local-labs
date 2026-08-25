# REP-02 Precision Tail Standard - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: A quantização uniforme do KV cache (ex: `q4_0` / INT4 uniforme em todas as posições) assume erroneamente que todos os tokens de contexto possuem a mesma sensibilidade a erros numéricos. Na prática, os primeiros tokens de sequência atuam como *attention sinks* absorvendo massas desproporcionais de softmax (StreamingLLM / IntactKV), enquanto os tokens recentes da cauda contêm o working set imediato (últimos passos de raciocínio, tool calls, instruções correntes). Manter os **attention sinks** ($S=4$) e a **cauda recente** ($T \in \{64, 128\}$) em FP16 enquanto o corpo intermediário é comprimido em 4 bits recupera a integridade de atenção e preserva a retenção em contexto longo, mantendo mais de $70\%$ da economia de VRAM sem a necessidade do complexo ecossistema do KVarN.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Modelo Alvo**: `Qwen/Qwen3.5-0.8B-Base` (`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`)
* **Hardware**: NVIDIA GeForce RTX 3090 (24GB GDDR6X, `sm_86`)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Braços Experimentais**:
  1. `FP16_UNIFORM`: Contexto 100% em FP16 (teto numérico, 100% de memória base).
  2. `INT4_UNIFORM`: Quantização assimétrica/simétrica em 4 bits em todas as posições ($[0, L]$).
  3. `PRECISION_TAIL_64`: Sinks $S=4$ em FP16, Corpo $[4, L-64]$ em INT4, Cauda $[L-64, L]$ em FP16.
  4. `PRECISION_TAIL_128`: Sinks $S=4$ em FP16, Corpo $[4, L-128]$ em INT4, Cauda $[L-128, L]$ em FP16.
* **Comprimentos de Contexto Avaliados**: $L \in \{256, 1024, 4096\}$.
* **Métricas Principais**:
  1. `attention_cosine_similarity`: Similaridade de cosseno entre a distribuição de atenção do braço e a do FP16 baseline.
  2. `attention_kld`: Divergência KL entre as distribuições de probabilidade de atenção.
  3. `needle_retrieval_pass`: Taxa de sucesso no teste de agulha no palheiro (*Needle-in-a-Haystack* em $25\%, 50\%, 75\%$ da profundidade).
  4. `kv_memory_compression_ratio`: Redução real em bytes da estrutura de cache vs FP16.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Recuperação de Atenção ($D_{KL} \le 0.05$)**: A política de Precision Tail deve reduzir a divergência KL de atenção em pelo menos $50\%$ em relação à quantização uniforme INT4.
2. **Gate de Recuperação Needle ($100\%$)**: 3 de 3 agulhas recuperadas com sucesso em contexto longo.
3. **Gate de Economia de Memória ($\ge 65\%$)**: Manter economia de pelo menos 65% da VRAM do KV em contextos $\ge 1024$ tokens.
