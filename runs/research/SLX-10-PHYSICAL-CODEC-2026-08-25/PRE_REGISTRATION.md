# SLX-10 Physical-Budget Codec Bakeoff - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: No limite físico de 24GB de VRAM da RTX 3090, a escolha do codec de compressão de pesos governa o teto de capacidade e a vazão de decodificação. Codecs vetoriais extremos de 2 bits (AQLM / QuIP# E8) reduzem a pegada de memória em até $55\%$ em relação ao GGUF Q4_K_M (permitindo modelos de até 35B parâmetros em 12GB), mas impõem overhead de descompressão de índices por codebook nos Tensor Cores. Este bakeoff quantifica o throughput efetivo (GB/s de descompressão) e a pegada física por parâmetro (bits/peso) dos principais codecs modernos na GPU Ampere (`sm_86`).

---

## 🎯 1. Contrato e Protocolo Experimental

* **Hardware**: NVIDIA GeForce RTX 3090 (24GB GDDR6X, GA102, 936 GB/s)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Codecs Avaliados (Matriz de Pesos $4096 \times 4096$)**:
  1. `FP16_UNCOMPRESSED`: 16.0 bits/peso (Referência não-comprimida).
  2. `GGUF_Q4_K_M`: 4.5 bits/peso (Escalares em bloco com super-blocos).
  3. `GGUF_IQ2_XXS`: 2.06 bits/peso (Quantização de importância vetorial).
  4. `AQLM_2BIT_2X8`: 2.12 bits/peso (Quantização Aditiva Vetorial com 2 codebooks de 8 dimensões).
  5. `QUIP_SHARP_E8P`: 2.00 bits/peso (Projeção em reticulado $E_8$).
* **Métricas**:
  1. `effective_decompression_bandwidth_gbs`: Largura de banda sustentada de descompressão na GPU.
  2. `vram_footprint_for_27b_gib`: Projeção de memória VRAM necessária para hospedar 27 bilhões de parâmetros.
  3. `decode_latency_us`: Latência de GEMV por token.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Envelope de 24GB para 27B/35B**: Identificar codecs que mantêm modelos $\ge 27\text{B}$ abaixo de $14\text{ GiB}$ de VRAM.
2. **Gate de Largura de Banda Mínima**: Descompressão $\ge 300\text{ GB/s}$ sustentados na RTX 3090.
3. **Mapeamento de Trade-off**: Determinação da fronteira de Pareto (Bits/Peso vs GB/s).
