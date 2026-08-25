# ADAPT-06 Adapter-Aware KV Cache & Tagging Controller - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: Em runtimes multi-tenant que suportam múltiplos adaptadores PEFT com prompt caching compartilhado, a ausência de segregação na chave de cache gera *Cross-Adapter KV Contamination* (tokens pré-calculados sob as matrizes do Adaptador A são incorretamente herdados por requisições do Adaptador B). Um controlador de identidade física de cache baseado em chave composta de 64 bits ($\text{Hash}(\text{adapter\_id} \mathbin{\Vert} \text{model\_id} \mathbin{\Vert} \text{prefix\_tokens})$) elimina 100% das colisões inter-adaptadores sem comprometer o reaproveitamento legítimo de prefixos sob o mesmo adaptador.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Estrutura da Chave de Cache (64-bit Hash Identifier)**:
  $$\text{CacheKey} = \text{SipHash64}(\text{model\_hash} \mathbin{\Vert} \text{adapter\_identity} \mathbin{\Vert} \text{prefix\_tokens})$$
* **Cenário de Teste**:
  - Simulação de 100 requisições multi-tenant com 3 adaptadores concorrentes:
    - Adaptador A (`lokr_math_r8`)
    - Adaptador B (`lokr_code_r8`)
    - Adaptador C (`base_backbone`)
  - Mistura de prompts com prefixos idênticos de sistema ("Você é um assistente...") e sufixos específicos de tarefa.
* **Métricas**:
  1. `cross_adapter_collision_count`: Número de contaminações de cache detectadas (meta: 0).
  2. `intra_adapter_hit_rate`: Taxa de reaproveitamento de KV sob o mesmo adaptador (meta: $\ge 80\%$).
  3. `lookup_overhead_ns`: Latência de resolução da chave de cache por bloco.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Isolamento Estrito (Zero Colisões)**: $0/100$ colisões cruzadas entre adaptadores distintos.
2. **Gate de Eficiência de Cache Hit**: $\ge 75\%$ de reaproveitamento legítimo de prefixos sob o mesmo adaptador.
3. **Passagem na Suite de Testes**: 100% de aprovação nos testes unitários do tagger de cache.
