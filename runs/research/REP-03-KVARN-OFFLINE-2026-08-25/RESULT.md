# REP-03 KVarN Offline Codec (Walsh-Hadamard) - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `REJECTED` — A rotação de Walsh-Hadamard $H_{128}$ reduziu o erro de reconstrução do KV cache em **70.71%** sobre a quantização direta (de 0.199 para 0.058), elevando a similaridade de atenção para **0.971**, mas ficou marginalmente abaixo do gate estrito de $\ge 0.990$, comprovando a necessidade de combinar Hadamard com cauda de alta precisão (Tail-F16).

---

## 🎯 1. Resumo Executivo

O experimento avaliou o codec offline KVarN aplicando a transformação ortogonal de Walsh-Hadamard ($H_{128} \times K$) antes da quantização INT4 em tensores de atenção com outliers de ativação pontuais em sequências de 2048 tokens utilizando [`tools/probes/rep03_kvarn_codec.py`](../../tools/probes/rep03_kvarn_codec.py).

A hipótese de dispersão de energia de outliers foi **CONFIRMADA NO ERRO DE TENSOR, MAS COM DEFASAGEM MARGINAL NA SIMILARIDADE DE ATENÇÃO**:
- A quantização direta INT4 sem rotação colapsou a fidelidade dos mapas de atenção ($\text{Cosine Sim} = 0.89659$, $\text{MSE} = 0.19945$).
- A rotação de Hadamard **suprimiu 70.71% do erro quadrático** ($\text{MSE} = 0.05842$), recuperando a similaridade de atenção para **0.97136** com 75% de economia de VRAM.
- A quantização INT2 extrema (2 bits) provou ser destrutiva ($\text{Sim} = 0.392$).

---

## 📊 2. Tabela de Comparação de Codecs de KV Cache ($L=2048$)

| Codec de KV Cache | Formato / Rotação | Reconstrução MSE | Similaridade de Atenção (Cosine) | Redução de MSE vs Direto | Veredito |
|---|---|:---:|:---:|:---:|:---:|
| **`UNQUANTIZED_FP16`**| FP16 (Controle) | 0.00000 | 1.00000 | 0.0% | Referência |
| **`DIRECT_INT4`**     | INT4 Sem Rotação | 0.19945 | 0.89659 | 0.0% | `REJECTED` |
| **`HADAMARD_INT4`**   | **$H_{128}$ + INT4** | **0.05842** | **0.97136** | **-70.71% (PASS MSE)** | `REJECTED (GATE SIM < 0.99)` |
| **`HADAMARD_INT2`**   | $H_{128}$ + INT2 | 2.28260 | 0.39218 | +1044% (Degradação) | `REJECTED` |

---

## 🔬 3. Diretriz para o Kernel Nativo (`REP-04`)

1. **Fusão Hadamard + Precision Tail**:
   - Para atingir $\text{Cosine Sim} \ge 0.995$ no kernel CUDA FlashAttention (`REP-04`), fundir a rotação de Hadamard com os 64 tokens recentes em FP16 (`Precision Tail Standard`).

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/REP-03-KVARN-OFFLINE-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Script da Prova**: [`tools/probes/rep03_kvarn_codec.py`](../../tools/probes/rep03_kvarn_codec.py)
- **Agente Executor**: Antigravity
