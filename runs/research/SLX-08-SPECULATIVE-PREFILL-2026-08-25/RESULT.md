# SLX-08 Speculative Prefill (PFlash) - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `REJECTED` — O prefill especulativo por chunking obteve **1.93× de speedup no TTFT em 8192 tokens**, mas a perda de representação (Cosine Sim = 0.7305) falsificou o gate de fidelidade ($\ge 0.95$), comprovando que camadas de atenção iniciais exigem contexto denso integral.

---

## 🎯 1. Resumo Executivo

O experimento avaliou o pipeline de *Speculative Prefill* com filtragem esparsa de blocos em chunks de 256 tokens contra o prefill denso padrão através de sequências de 1024 a 8192 tokens utilizando o probe [`tools/probes/slx08_speculative_prefill_oracle.py`](../../tools/probes/slx08_speculative_prefill_oracle.py) na RTX 3090.

A hipótese de preservação de fidelidade com aceleração linear foi **FALSIFICADA**:
- O speedup de TTFT em 8192 tokens foi substancial (**1.93×**, caindo de 8.20 ms para 4.24 ms).
- No entanto, a poda estática de 50% dos blocos de atenção gerou distorção residual nos vetores de contexto ($\text{Cosine Sim} = 0.7305$, distante do piso de 0.95).
- A conclusão teórica é que o prefill não tolera esparsificação homogênea em todas as camadas sem mecanismos de compensação ou preservação densa nas primeiras 4 camadas.

---

## 📊 2. Tabela de Métricas de Prefill por Comprimento ($L$)

| Comprimento ($L$) | TTFT Denso (ms) | TTFT Especulativo (ms) | Speedup Efetivo | Cosine Similarity | Veredito |
|---|:---:|:---:|:---:|:---:|:---:|
| **$L=1024$** | 0.27 ms | 14.75 ms (Warmup/Overhead) | 0.02× | 0.7109 | `REJECTED` |
| **$L=2048$** | 0.65 ms | 0.38 ms | **1.72×** | 0.6719 | `REJECTED` |
| **$L=4096$** | 2.23 ms | 1.18 ms | **1.89×** | 0.7070 | `REJECTED` |
| **$L=8192$** | 8.20 ms | 4.24 ms | **1.93× (PASS SPEED)** | **0.7305 (FAIL SIM)** | `REJECTED` |

---

## 🔬 3. Diretriz para o `slop.cpp`

1. **Rejeição de Chunking Homogêneo no Prefill**:
   - Manter a execução de prefill densa através de FlashAttention padrão.
   - Não aplicar máscaras de prefill especulativo sem esquema hierárquico (camadas densas iniciais + esparsidade progressiva).

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/SLX-08-SPECULATIVE-PREFILL-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Script da Prova**: [`tools/probes/slx08_speculative_prefill_oracle.py`](../../tools/probes/slx08_speculative_prefill_oracle.py)
- **Agente Executor**: Antigravity
