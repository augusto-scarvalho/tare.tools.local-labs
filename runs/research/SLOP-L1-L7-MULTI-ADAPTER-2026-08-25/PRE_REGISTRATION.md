# SLOP-L1..L7 Multi-Adapter Serving Engine Levers - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: Em servidores multi-slot atendendo requisições com múltiplos adaptadores LoRA/LoKr distintos, o chaveamento ingênuo de pesos por slot gera gargalos severos de overhead de memória e fragmentação de kernel. Um roteador in-flight com agrupamento dinâmico por afinidade de adaptador (*Affinity Batching*) e cache de matrizes de delta fundidas reduz as trocas de contexto em $\ge 60\%$ e eleva o throughput em $\ge 25\%$ mantendo 100% de integridade de roteamento de tenants.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Arquitetura do Roteador (MultiAdapterFlightRouter)**:
  - `register_adapter(adapter_id, weights_descriptor)`: Registra adaptadores disponíveis na VRAM.
  - `schedule_step(active_slots)`: Agrupa slots ativos por afinidade de adaptador (`batched_adapter_groups`).
  - `dispatch_batched_step()`: Executa passos de decodificação agrupados com zero overhead de hotswap redundante.
* **Cenário de Teste**:
  - 4 slots de execução paralela atendendo 200 requisições distribuídas entre 4 adaptadores distintos.
  - Comparativo pareado:
    1. `NAIVE_SCHEDULER`: Troca de contexto individual a cada requisição / slot.
    2. `AFFINITY_ROUTER`: Agrupamento de slots por afinidade de matriz e reuso de tensores de delta.
* **Métricas**:
  1. `context_switch_reduction_pct`: Redução percentual de trocas de adaptadores.
  2. `routing_accuracy_rate`: Proporção de requisições processadas com o adaptador correto (meta: 100%).
  3. `step_dispatch_overhead_us`: Tempo de despacho do agendador por ciclo.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Integridade de Roteamento (100%)**: Zero requisições executadas com adaptador incorreto ($0/200$ erros).
2. **Gate de Redução de Trocas ($\ge 50\%$)**: Redução de pelo menos 50% nos context switches vs escalonador ingênuo.
3. **Passagem na Suite de Testes**: 100% de aprovação nos testes unitários do roteador.
