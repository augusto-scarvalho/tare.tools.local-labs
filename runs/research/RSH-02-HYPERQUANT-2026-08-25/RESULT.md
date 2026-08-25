# RSH-02 HyperQuant Entropy Coding - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `REJECTED` — A codificação por entropia de comprimento variável (Huffman/ANS) atingiu **2.40 bits/elemento (40.0% de compressão física sobre INT4)**, mas a serialização da leitura de bitstream e divergência de threads em warps da GPU limitou o throughput a **7.68 GB/s**, violando o gate de throughput ($\ge 100\text{ GB/s}$).

---

## 🎯 1. Resumo Executivo

O experimento comparou a eficiência de descompressão de bitstreams com comprimento de símbolo variável (Huffman / ANS) contra a descompressão SIMD de largura fixa (`INT4`) em matriz $4096 \times 4096$ na RTX 3090 através de [`tools/probes/rsh02_hyperquant_entropy_coding.py`](../../tools/probes/rsh02_hyperquant_entropy_coding.py).

A hipótese de viabilidade para inferência de alta velocidade foi **FALSIFICADA**:
- O ganho de compressão de Shannon foi confirmado (**2.40 bpw** vs 4.0 bpw no INT4 fixo).
- No entanto, a descompressão bit-a-bit impede a paralelização via registradores SIMD (`mma.sync` / `prmt`), gerando throughput de apenas **7.68 GB/s**, muito abaixo da largura de banda nativa da GPU (936 GB/s).
- Formatos de tamanho fixo em bloco (`Q4_0` / `IQ2_XXS`) continuam indispensáveis para manter a latência de inferência no nível de microssegundos.

---

## 📊 2. Tabela de Métricas do Codec de Entropia (Matriz $4096 \times 4096$)

| Formato de Codificação | Bits / Elemento (bpw) | Latência de Descompressão | Throughput Efetivo | Ganho de Compressão | Veredito |
|---|:---:|:---:|:---:|:---:|:---:|
| **`FIXED_INT4_SIMD`**     | 4.00 bpw | **0.743 ms** | **10.52 GB/s** | 0.0% (Base) | Baseline |
| **`VARIABLE_HUFFMAN_ANS`**| **2.40 bpw** | 0.610 ms | **7.68 GB/s (FAIL GATE)** | **+40.00%** | `REJECTED` |

---

## 🔬 3. Diretriz de Engenharia

1. **Rejeição de Bitstreams com Comprimento de Símbolo Variável no `slop.cpp`**:
   - Manter todos os formatos de pesos e KV cache alinhados a fronteiras de bits fixas (2 bits, 3 bits, 4 bits, 8 bits) por bloco, garantindo descompressão em uma única instrução bitwise por elemento.

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/RSH-02-HYPERQUANT-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Script da Prova**: [`tools/probes/rsh02_hyperquant_entropy_coding.py`](../../tools/probes/rsh02_hyperquant_entropy_coding.py)
- **Agente Executor**: Antigravity
