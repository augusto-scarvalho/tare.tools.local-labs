# SLX-03 ReplaySSM State-Write Elision - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `PROMOTED` — O mecanismo de elisão de escrita de estado recorrente (*State-Write Elision*) em camadas Gated DeltaNet obteve **3.48× de speedup no loop de decodificação** e **99.22% de redução no tráfego de gravação em DRAM** (de 288 MB para 2.25 MB em 128 tokens).

---

## 🎯 1. Resumo Executivo

O experimento avaliou o impacto do tráfego de escrita de estado na memória global GPU através das 18 camadas lineares de Gated DeltaNet do `Qwen3.5-0.8B-Base` utilizando o probe de precisão [`tools/probes/slx03_state_write_oracle.py`](../../tools/probes/slx03_state_write_oracle.py) na RTX 3090.

A hipótese de gargalo de largura de banda por persistência excessiva foi **CONFIRMADA**:
- O baseline tradicional de persistência a cada token (`PERSIST_EVERY_TOKEN`) gastou 181.7 µs/passo gerando 288 MB de IO de memória global.
- A política **`ELISION_N16`** (persistindo estado em DRAM a cada 16 tokens) reduziu o tráfego em **93.75%**, acelerando a execução para **3.19× (56.9 µs/passo)**.
- A política **`EPHEMERAL_EOS_ONLY`** (estado mantido em registradores durante o streaming e gravado apenas no fim do prompt) alcançou **3.48× de aceleração (52.2 µs/passo)** com **99.22% de corte de IO**.

---

## 📊 2. Tabela de Métricas de Performance (128 Tokens / 18 Camadas GDN)

| Política de Escrita de Estado | Intervalo ($N$) | Latência por Passo (µs) | Tempo Total (ms) | Volume Escrito em DRAM (MB) | Speedup Efetivo | Redução de IO |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`PERSIST_EVERY_TOKEN`** | 1 | 181.7 µs | 23.26 ms | 288.00 MB | 1.00× (Base) | 0.0% |
| **`ELISION_N4`** | 4 | 119.3 µs | 15.27 ms | 72.00 MB | **1.52×** | **75.00%** |
| **`ELISION_N16`** | 16 | 56.9 µs | 7.28 ms | 18.00 MB | **3.19×** | **93.75%** |
| **`EPHEMERAL_EOS_ONLY`** | 128 | **52.2 µs** | **6.69 ms** | **2.25 MB** | **3.48×** | **99.22%** |

---

## 🔬 3. Diretriz para o `slop.cpp`

1. **Gestão Efêmera de Estado Recorrente**:
   - Implementar buffers de estado em registros/SRAM no kernel de decodificação de Gated DeltaNet do `slop.cpp`.
   - Persistir o estado do tensor na memória global apenas ao término do token stream ou em checkpoints de transação periódica ($N=16$).

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/SLX-03-STATE-WRITE-ELISION-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Script da Prova**: [`tools/probes/slx03_state_write_oracle.py`](../../tools/probes/slx03_state_write_oracle.py)
- **Agente Executor**: Antigravity
