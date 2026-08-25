# REP-04 KVarN Native Attention Kernel - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `REJECTED` — A fusão do codec KVarN reduziu o tráfego de DRAM em **72.93%**, mas a sobrecarga computacional da rotação de Walsh-Hadamard desacoplada em nível de PyTorch aumentou a latência do kernel de **142.1 µs $\rightarrow$ 261.6 µs (speedup de 0.54×)**, comprovando que o KVarN exige implementação em assembly PTX/Triton com rotação in-register dentro do warp.

---

## 🎯 1. Resumo Executivo

O experimento avaliou o kernel de atenção híbrido KVarN (Transformação de Walsh-Hadamard + Corpo INT4 + Cauda Recente FP16 $T=64$) em sequências de 8.192 tokens na RTX 3090 através de [`tools/probes/rep04_kvarn_native_kernel.py`](../../tools/probes/rep04_kvarn_native_kernel.py).

A hipótese de aceleração imediata sem kernel PTX dedicado foi **FALSIFICADA**:
- O kernel atingiu o objetivo de largura de banda, **eliminando 72.93% do tráfego de leitura de memória DRAM**.
- Contudo, a realização da rotação de Hadamard via multiplicação matricial no nível do host adicionou passos de cálculo intermediários, tornando o kernel Compute-Bound na RTX 3090 e elevando o tempo de atenção para **261.6 µs** (contra 142.1 µs do FlashAttention FP16).

---

## 📊 2. Tabela de Métricas do Kernel de Atenção ($L=8192$, RTX 3090)

| Kernel de Atenção | Latência por Passo (µs) | Speedup Efetivo | Redução de Tráfego DRAM | Similaridade Cosseno | Veredito |
|---|:---:|:---:|:---:|:---:|:---:|
| **`STANDARD_FLASH_ATTN_FP16`** | **142.1 µs** | 1.00× (Base) | 0.0% | 1.00000 | Referência |
| **`FUSED_KVARN_HYBRID_INT4`**   | 261.6 µs | **0.54× (Lento)** | **72.93% (PASS DRAM)** | 0.94141 | `REJECTED (FAIL SPEED)` |

---

## 🔬 3. Diretriz de Engenharia

1. **Requisito de Kernel Nativo em C++/CUDA**:
   - Não tentar integrar Hadamard KV no `slop.cpp` via invocações separadas de BLAS.
   - O KVarN só deve ser ativado quando fundido diretamente dentro do loop interno do kernel de decodificação FlashAttention (`mma.sync` nos Tensor Cores com dequantização e rotação via registradores de warp).

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/REP-04-KVARN-NATIVE-KERNEL-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Script da Prova**: [`tools/probes/rep04_kvarn_native_kernel.py`](../../tools/probes/rep04_kvarn_native_kernel.py)
- **Agente Executor**: Antigravity
