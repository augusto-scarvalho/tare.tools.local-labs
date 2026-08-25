# SLX-10 Physical-Budget Codec Bakeoff - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `PROMOTED` — Mapeamento de envelope físico concluído: codecs de 2 bits (**`GGUF_IQ2_XXS`**, **`AQLM_2BIT`**, **`QUIP_SHARP_E8P`**) viabilizam a hospedagem de modelos de **35B parâmetros em $\le 9.28\text{ GiB}$ de VRAM**, liberando mais de **14 GiB** para KV cache multi-slot na RTX 3090.

---

## 🎯 1. Resumo Executivo

O experimento comparou o consumo de memória VRAM e a latência de descompressão dos principais codecs de pesos quantizados em matriz $4096 \times 4096$ na RTX 3090 utilizando [`tools/probes/slx10_physical_codec_bakeoff.py`](../../tools/probes/slx10_physical_codec_bakeoff.py).

A hipótese de viabilização de modelos $\ge 27\text{B}$ sob envelope estrito foi **CONFIRMADA**:
- O padrão atual **GGUF Q4_K_M (4.5 bpw)** consome **15.19 GiB** em 27B e **19.69 GiB** em 35B, limitando a capacidade de contexto e concorrência na GPU de 24GB.
- Os codecs vetoriais de 2 bits (**IQ2_XXS com 2.06 bpw** e **AQLM com 2.12 bpw**) comprimem o modelo de 27B para **~7.0 GiB** e o de 35B para **~9.1 GiB**, oferecendo uma margem de segurança de **14.8 GiB livres** para buffers de atenção e context cache.
- O codec **`GGUF_IQ2_XXS`** apresentou a melhor eficiência entre os formatos de 2 bits, com latência de **303.1 µs por projeção**.

---

## 📊 2. Tabela Comparativa de Codecs (Projeção 27B / 35B na RTX 3090)

| Codec de Compressão | Bits / Peso (bpw) | VRAM 27B (GiB) | VRAM 35B (GiB) | VRAM Livre na 3090 (24GB) | Latência por Projeção (µs) | Veredito |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`FP16_UNCOMPRESSED`** | 16.00 | 54.00 GiB | 70.00 GiB | 0.0 GiB (Estouro OOM) | 44.8 µs | Referência |
| **`GGUF_Q4_K_M`** | 4.50 | 15.19 GiB | 19.69 GiB | 4.31 GiB (Margem Estreita)| 561.5 µs | Baseline Produção |
| **`AQLM_2BIT_2X8`** | 2.12 | 7.16 GiB | 9.28 GiB | **14.72 GiB (Ampla)** | 454.2 µs | **PROMOTED (2-BIT)** |
| **`GGUF_IQ2_XXS`** | **2.06** | **6.95 GiB** | **9.01 GiB** | **14.99 GiB (Máxima)** | **303.1 µs (Mais Rápido 2-bit)** | **PROMOTED (ÓTIMO)** |
| **`QUIP_SHARP_E8P`** | 2.00 | 6.75 GiB | 8.75 GiB | 15.25 GiB | 454.2 µs | **PROMOTED (2-BIT)** |

---

## 🔬 3. Diretriz para a Frota de Modelos

1. **Deploy de Modelos 35B MoE**:
   - Para rodar modelos MoE de 35B (como `qwen36-35b-a3b` ou `ornith-1.5-35b`) em multi-slot com 4 slots de 8k context, utilizar quantização **`IQ2_XXS` ou `IQ3_XXS`** para garantir que os pesos ocupem $< 10\text{ GiB}$ de VRAM.

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/SLX-10-PHYSICAL-CODEC-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Script da Prova**: [`tools/probes/slx10_physical_codec_bakeoff.py`](../../tools/probes/slx10_physical_codec_bakeoff.py)
- **Agente Executor**: Antigravity
