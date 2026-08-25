# SPEC-01 Speculative Evolution Pipeline (N-Gram + MTP) - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: Em tarefas com estruturas sintáticas repetitivas (código fonte, JSON, tags de raciocínio `<think>`), a combinação hierárquica de um cache de n-grams (Trie em RAM com custo de dispatch de $0.0\text{ µs}$ na GPU) com o modelo neural MTP (Multi-Token Prediction) acelera a taxa média de tokens aceitos por passo de verificação ($\bar{\tau}$) de **$1.65 \rightarrow \ge 2.20$**, proporcionando um speedup efetivo de throughput $\ge 2.0\times$ sobre a geração autoregressiva.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Ambiente**: Python 3.11 / PyTorch na máquina host
* **Arquitetura do Motor Híbrido**:
  - `TrieNGramDrafter`: Índice de n-grams ($N=3..5$) em janela de contexto recente.
  - `MTPNeuralDrafter`: Gerador multi-token preditivo (profundidade $K=4$).
  - `SpeculativeArbiter`: Chaveia para o Trie quando há casamento exato de prefixo; caso contrário, despacha o MTP neural.
* **Métricas**:
  1. `mean_accepted_tokens_per_step`: Média de tokens aceitos por verificação do modelo alvo.
  2. `speculative_speedup_factor`: Razão de tokens/s do pipeline híbrido vs autoregressivo.
  3. `ngram_hit_rate_pct`: Porcentagem de drafts atendidos com zero custo de GPU pelo Trie.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Throughput ($\ge 1.80\times$)**: Speedup $\ge 1.80\times$ vs autoregressivo puro.
2. **Gate de Taxa de Aceitação ($\bar{\tau} \ge 2.0$)**: Média $\ge 2.0$ tokens aceitos por passo.
3. **Fidelidade Exata**: $100\%$ de equivalência determinística com a saída do modelo alvo.
