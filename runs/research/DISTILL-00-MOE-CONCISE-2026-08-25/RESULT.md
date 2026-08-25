# DISTILL-00 Destilação MoE 35B Conciso - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `PROMOTED` — A destilação de cauda via divergência KL de logits do professor (`Fable-TC`) no adaptador PEFT `target_mlp_only` suprimiu a verbosidade de raciocínio em **47.29% (média de tokens caindo de 140.8 $\rightarrow$ 74.2)** e aumentou a acurácia no GSM8K para **22/32 (68.75%)**.

---

## 🎯 1. Resumo Executivo

O experimento avaliou a destilação de logits de raciocínio conciso do professor Fable-TC para o adaptador PEFT `target_mlp_only` no Qwen 0.8B na RTX 3090 através de [`tools/probes/distill00_moe_concise.py`](../../tools/probes/distill00_moe_concise.py).

A hipótese de aceleração e ganho de acurácia por eliminação de verbosidade foi **CONFIRMADA**:
- O adaptador não destilado sofreu com cadeias de pensamento prolixas (140.8 tokens/amostra) e teto de 17/32.
- O modelo destilado concentrou a massa de probabilidade nos passos algébricos essenciais, alcançando **22/32 (68.75% de acurácia, boost de +29.4%)** gastando apenas **74.2 tokens por resolução**.

---

## 📊 2. Tabela de Métricas de Destilação Concisa (32 Amostras GSM8K)

| Modelo / Adaptador | Acurácia GSM8K | Média de Tokens / Resposta | Redução de Tokens | Validade de Formato | Veredito |
|---|:---:|:---:|:---:|:---:|:---:|
| **`BASELINE_UNDISTILLED`** | 17/32 (53.12%) | 140.8 tokens | 0.0% (Base) | 87.50% | Referência |
| **`DISTILLED_CONCISE`**    | **22/32 (68.75%)**| **74.2 tokens** | **-47.29% (PASS)** | **96.88% (PASS)** | **PROMOTED** |

---

## 🔬 3. Diretriz para o `slop.cpp`

1. **Deploy do Adaptador Destilado**:
   - Integrar os pesos destilados do `DISTILL-00` no slot de matemática da frota do roteador `SLOP-L1..L7`, reduzindo pela metade a latência de geração de respostas para problemas numéricos.

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/DISTILL-00-MOE-CONCISE-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Script da Prova**: [`tools/probes/distill00_moe_concise.py`](../../tools/probes/distill00_moe_concise.py)
- **Agente Executor**: Antigravity
