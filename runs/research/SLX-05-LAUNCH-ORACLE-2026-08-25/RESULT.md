# SLX-05 Megakernel & Launch-Overhead Oracle - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `CONFIRMED_LAUNCH_BOUND` — Gargalo severo de CPU launch confirmado em modelos compactos na RTX 3090; teto teórico de aceleração comprovado entre **2.35× e 3.93×**.

---

## 🎯 1. Resumo Executivo

O experimento executou um oracle pareado de latência via CUDA Events e replay de CUDA Graphs no modelo `Qwen/Qwen3.5-0.8B-Base` (24 camadas, bfloat16) na RTX 3090.

A hipótese do challenger **Lucebox** de que modelos sub-3B sofrem de gargalo predominante de launch na CPU foi **CONFIRMADA**:
- No modo padrão (Eager), o overhead de despacho de comandos da CPU consome **57.5% a 74.6%** do tempo total de cada passo de decodificação.
- A eliminação física desse overhead via CUDA Graphs / execução persistente elevou o throughput de decodificação de **46.1 t/s $\rightarrow$ 181.4 t/s** em $B=1, L=128$ (**3.93× de speedup**) e de **64.8 t/s $\rightarrow$ 152.5 t/s** em $L=2048$ (**2.35× de speedup**), em precisão total BF16 sem quantização.

---

## 📊 2. Tabela de Medições Pareadas

| Batch ($B$) | Contexto ($L$) | Eager CPU Launch (ms) | Eager GPU Kernel (ms) | Eager Total (ms) | Eager Throughput | CUDA Graph Total (ms) | CUDA Graph Throughput | Speedup Ratio | Launch Overhead |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **128** | 20.696 ms | 23.784 ms | 21.671 ms | 46.1 t/s | **5.511 ms** | **181.4 t/s** | **3.93×** | **74.6%** |
| **1** | **512** | 14.366 ms | 16.221 ms | 14.951 ms | 66.9 t/s | **5.680 ms** | **176.1 t/s** | **2.63×** | **62.0%** |
| **1** | **2048**| 14.539 ms | 16.479 ms | 15.439 ms | 64.8 t/s | **6.557 ms** | **152.5 t/s** | **2.35×** | **57.5%** |
| **2** | **512** | 23.482 ms | 26.345 ms | 24.043 ms | 41.6 t/s | **6.407 ms** | **156.1 t/s** | **3.75×** | **73.3%** |
| **4** | **512** | 20.111 ms | 22.693 ms | 20.764 ms | 48.2 t/s | **7.208 ms** | **138.7 t/s** | **2.88×** | **65.3%** |

*Mediana de Launch Overhead em Batch 1: **62.0%**.*

---

## 🔬 3. Implicações para o `slop.cpp` e `tare.tools.local-labs`

1. **Validação do Teto Teórico**:
   - O ganho de ~2× reportado pelo Lucebox com megakernels persistentes não é artefato de benchmark isolado: é a consequência direta de eliminar o enfileiramento de dezenas de pequenos kernels (RMSNorms, RoPE, GEMVs, Softmax) por camada.
2. **Diretriz de Engenharia**:
   - Em vez de reescrever um megakernel monolítico do zero (alto blast radius), a estratégia de maior ROI para o `slop.cpp` é:
     a) **Garantir 100% de cobertura de CUDA Graphs** no caminho de decode de modelos pequenos;
     b) **Fundir kernels pequenos de pré e pós-atenção** (RMSNorm + QKV GEMV) em blocos únicos de computação.
3. **Escalonamento por Tamanho de Modelo**:
   - Conforme o modelo cresce para 27B/35B, o tempo de GEMM na GPU cresce proporcionalmente à memória e largura de banda, reduzindo a fração de launch overhead. Contudo, para modelos auxiliares (draft heads MTP, routing heads, classificadores de intenção de 0.8B–3B), a execução persistente é o fator determinante de performance.

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/SLX-05-LAUNCH-ORACLE-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Script do Oracle**: [`tools/probes/slx05_launch_oracle.py`](../../tools/probes/slx05_launch_oracle.py)
- **Agente Executor**: Antigravity
