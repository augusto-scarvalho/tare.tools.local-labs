# RSH-02 HyperQuant Entropy Coding - Pré-Registro

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Hipótese Causal**: A codificação por entropia de comprimento variável (Huffman / ANS) sobre símbolos quantizados atinge taxa de compressão teórica superior ($\le 2.2\text{ bits/valor}$), mas a serialização da descompressão bit-a-bit e a divergência de threads em warps da GPU geram uma queda de throughput de descompressão $\ge 10\times$ em comparação com a descompressão SIMD de tamanho fixo (`INT4`), inviabilizando o seu uso para serving de baixa latência.

---

## 🎯 1. Contrato e Protocolo Experimental

* **Hardware**: NVIDIA GeForce RTX 3090 (24GB VRAM)
* **Ambiente**: WSL2 `/home/augus/.venvs/adapt00-20260824`
* **Formatos de Empacotamento Comparados (Matriz $4096 \times 4096$)**:
  1. `FIXED_INT4_SIMD`: Formato de 4 bits de tamanho fixo (2 nibbles por byte, decodificação via bitwise shift paralelo).
  2. `VARIABLE_HUFFMAN_ANS`: Codificação por entropia com prefix-tree (símbolos frequentes em 1..2 bits, raros em 5..7 bits).
* **Métricas**:
  1. `compression_bits_per_element`: Taxa física de bits por valor no bitstream comprimido.
  2. `decode_throughput_gb_s`: Largura de banda sustentada de descompressão.
  3. `decode_latency_us`: Latência por token gerado.

---

## 🛑 2. Critérios de Promoção e Decisão (Kill Gates)

1. **Gate de Throughput ($\ge 100\text{ GB/s}$)**: A descompressão deve sustentar $\ge 100\text{ GB/s}$ para ser viável.
2. **Gate de Compressão ($\le 3.0\text{ bits/elem}$)**: Ganho de compressão significativo sobre INT4 fixo.
3. **Penalidade de Latência ($\le 2.0\times$ vs INT4)**: Não degradar a latência por token em mais de $2\times$.
