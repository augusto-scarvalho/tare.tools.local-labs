# SLX-01B Stateful Serving Torture Matrix - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: Em servidores multi-slot de inferência stateful (`total_slots=4`), cancelamentos abruptos de clientes HTTP durante a fase de geração token a token (streaming truncado com TCP RST/FIN forçado) combinados com rajadas concorrentes de requests longos podem causar *zombie slot locking* (slot retido eternamente em estado de processamento), vazamento de descritores de KV cache ou corrupção de contexto entre requisições subsequentes. Esta suite de tortura aplica 50 ciclos de rajadas concorrentes com desconexões forçadas e verifica se o runtime recupera 100% dos slots para o estado `idle` com zero degradação de throughput e integridade de resposta.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Servidor Alvo**: `llm-inference.service` (`http://127.0.0.1:8080`, Fable-TC 27B Q4_K_M, total_slots=4)
* **Condições de Teste**:
  - 40 requisições completas concorrentes (4 workers simultâneos).
  - 20 requisições canceladas agressivamente após receber entre 1 e 5 tokens via streaming SSE.
  - 10 requisições com prompts de contexto extenso (> 2048 tokens).
* **Métricas Principais**:
  1. `slot_recovery_rate`: Proporção de slots que retornam a `idle` após cancelamento forçado (meta: 100%).
  2. `canary_pass_rate_post_torture`: Sucesso na geração de resposta canônica após a tempestade de cancelamentos.
  3. `vram_drift_mib`: Variação de uso de VRAM antes vs depois da tortura (meta: $\Delta \text{VRAM} \le 10 \text{ MiB}$).

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Integridade de Slots ($100\%$)**: Nenhum slot travado em estado zumbi (`is_processing == True` sem cliente ativo).
2. **Gate de Recuperação Pós-Tortura ($100\%$)**: 5 de 5 canaries pós-estresse respondidos com sucesso e formatação válida.
3. **Gate de VRAM Leak ($\le 20 \text{ MiB}$)**: Zero vazamento contínuo de memória após esvaziamento dos buffers.
