# BEE-L4 Transactional Target + MTP Restore - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: Na decodificação especulativa multi-slot, o descarte de tokens rascunhados rejeitados exige retroceder ponteiros de KV cache de forma atômica. Se o rollback for realizado por sobrescrita direta sem demarcação transacional, condições de corrida entre slots concorrentes podem corromper os offsets de sequência. Um gerenciador de transações de KV cache com semântica de checkpoint (`checkpoint` $\rightarrow$ `draft_append` $\rightarrow$ `verify` $\rightarrow$ `commit` / `rollback`) elimina 100% dos erros de inconsistência de estado multi-slot com overhead de latência negligenciável ($\le 5 \mu\text{s}$).

---

## 🎯 1. Contrato e Protocolo Experimental

* **Semântica Transacional de Slot**:
  - `begin_speculative_step(slot_id) -> TxHandle` (Captura ponteiro de cauda $L_{\text{base}}$).
  - `append_draft_tokens(slot_id, tokens)` (Grava $K$ tokens temporários).
  - `commit_step(slot_id, accepted_count)` (Avança cauda para $L_{\text{base}} + \text{accepted\_count}$).
  - `rollback_step(slot_id)` (Restaura cauda para $L_{\text{base}}$ sem alterar slots vizinhos).
* **Cenário de Teste**:
  - 4 slots concorrentes executando 500 ciclos de especulação e rollback assíncronos.
  - Injeção forçada de taxas variáveis de rejeição (0% a 100% de rollback).
* **Métricas**:
  1. `cross_slot_leakage_count`: Contagem de buffers ou ponteiros corrompidos entre slots (meta: 0).
  2. `state_consistency_rate`: Taxa de conformidade de tokens persistidos vs aceitos (meta: 100%).
  3. `transaction_overhead_us`: Tempo médio gasto nas operações de checkpoint e rollback.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Isolamento Transacional (Zero Leaks)**: $0/2000$ operações de rollback com inconsistência de estado.
2. **Gate de Integridade de Sequência (100%)**: Todos os tokens confirmados coincidem com o histórico de commits.
3. **Passagem na Suite de Testes**: 100% de aprovação nos testes unitários do gerenciador transacional.
