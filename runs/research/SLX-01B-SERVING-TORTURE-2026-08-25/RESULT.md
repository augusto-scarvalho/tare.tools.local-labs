# SLX-01B Stateful Serving Torture Matrix - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `PROMOTED` — O runtime multi-slot (`llm-inference.service`, 4 slots) demonstrou resiliência perfeita sob condições extremas de concorrência, cancelamentos forçados e tempestades mistas.

---

## 🎯 1. Resumo Executivo

O experimento executou a matriz de tortura assíncrona [`tools/probes/slx01b_serving_torture.py`](../../tools/probes/slx01b_serving_torture.py) contra o servidor de inferência em produção na porta 8080 (Fable-TC 27B Q4_K_M, `total_slots=4`).

A hipótese de estabilidade sob desconexões de clientes e concorrência foi **CONFIRMADA**:
- **Fase 1 (Carga Concorrente)**: 20 de 20 requisições completadas com sucesso (latência média de ~3.4s com 4 workers simultâneos).
- **Fase 2 (Cancelamentos Forçados)**: 20 de 20 conexões SSE abortadas abruptamente no socket logo após o recebimento dos primeiros bytes, sem gerar travamento de socket ou exceções não tratadas no servidor.
- **Fase 3 (Tempestade Mista)**: 20 requisições simultâneas misturando inferências completas e cancelamentos no meio da geração.
- **Auditoria de Slots**: **100% dos 4 slots retornaram ao estado `idle`** sem qualquer slot retido em processamento zumbi (`is_processing == False`).
- **Recuperação Pós-Tortura**: **5 de 5 canaries (100%)** executados e validados com precisão exata pós-estresse.

---

## 📊 2. Tabela de Métricas do Experimento

| Fase / Teste | Concorrência | Requisições Disparadas | Sucesso / Conclusão | Slots Travados (Zumbi) | Canary Pós-Estresse |
|---|:---:|:---:|:---:|:---:|:---:|
| **Fase 1: Concorrência Pura** | 4 workers | 20 | **20/20 (100%)** | 0 | — |
| **Fase 2: Cancelamentos SSE** | 6 workers | 20 | **20/20 Abortados Limpos** | 0 | — |
| **Fase 3: Tempestade Mista** | 8 workers | 20 | **20/20 Processados** | 0 | — |
| **Pós-Tortura: Auditoria de Slots** | — | — | **4/4 Slots Idle** | **0** | **5/5 (100% PASS)** |

---

## 📁 3. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/SLX-01B-SERVING-TORTURE-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Script da Prova**: [`tools/probes/slx01b_serving_torture.py`](../../tools/probes/slx01b_serving_torture.py)
- **Agente Executor**: Antigravity
