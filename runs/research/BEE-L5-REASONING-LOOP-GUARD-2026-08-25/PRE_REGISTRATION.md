# BEE-L5 Reasoning-Loop Guard - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: Modelos dotados de capacidade de raciocínio em cadeia sofrem ocasionalmente de colapso de entropia e armadilhas de auto-reavaliação recursiva (loops infinitos dentro do bloco de pensamento). Um sentinela leve de inferência baseado em contagem de transições de dúvida cíclicas em janela deslizante de 32 tokens detecta patologias de loop em tempo real ($< 2 \mu\text{s}$ por token), forçando o fechamento imediato do canal de pensamento e salvando até $80\%$ dos tokens que seriam desperdiçados em gerações truncadas.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Gatilhos de Detecção de Loop**:
  1. `reversal_phrase_density`: Densidade de termos de recálculo cíclico ("wait", "let me reconsider", "alternatively", "on second thought", "mas espere", "recalculando") $\ge 3$ ocorrências em 32 tokens.
  2. `repetition_cycle_detector`: Detecção de n-grams idênticos de tamanho $N \in \{4, 8\}$ repetidos $\ge 3$ vezes contíguas.
  3. `entropy_collapse_indicator`: Queda sustentada da entropia de vocabulário em janela móvel.
* **Ação do Guard**:
  - Emissão de sinal `FORCE_CLOSE_THINKING` para injetar `\n</think>\n` e forçar a conclusão da resposta.
* **Corpus de Validação**:
  - 50 trajetórias de geração (25 raciocínios legítimos longos + 25 loops patológicos reais extraídos de modelos R1/ThinkingCap).

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Sensibilidade de Detecção (True Positive Rate $\ge 95\%$)**: Identificar e cortar pelo menos 24 de 25 loops patológicos.
2. **Especificidade (False Positive Rate $\le 2\%$)**: Não interromper prematuramente nenhum dos raciocínios legítimos.
3. **Passagem na Suite de Testes**: 100% de aprovação nos testes unitários do guard.
