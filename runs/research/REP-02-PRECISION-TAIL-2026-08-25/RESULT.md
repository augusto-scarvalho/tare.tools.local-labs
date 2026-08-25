# REP-02 Precision Tail Standard - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `REJECTED` — Em arquiteturas híbridas Recorrentes / Atenção Linear (como Qwen 3.5 com 18 camadas GDN e 6 camadas de Full Attention), o isolamento de sinks/tail sem rotação de Hadamard não produz redução causal de MSE ($\ge 50\%$) sobre o INT4 uniforme, embora preserve a recuperação de contexto (*Needle Recall 100%*).

---

## 🎯 1. Resumo Executivo

O experimento avaliou se preservar os *Attention Sinks* ($S=4$) e a *Recent Tail* ($T=64, 128$) em precisão total (FP16/BF16) durante a quantização em 4 bits do corpo intermediário do KV cache protege a representação de contexto longo em relação à quantização uniforme INT4 no `Qwen/Qwen3.5-0.8B-Base`.

A hipótese de redução de pelo menos $50\%$ no erro quadrático médio de logits sobre o INT4 uniforme foi **FALSIFICADA**:
- O teste de agulha no palheiro (*Needle-in-a-Haystack* em 2048 tokens a 50% de profundidade) passou com 100% de sucesso em todos os braços (`FP16`, `INT4_UNIFORM` e `PRECISION_TAIL_64`).
- A economia de memória no KV cache atingiu **70.8%** em 4096 tokens no `PRECISION_TAIL_64` (contra 72.0% no INT4 uniforme).
- Contudo, a redução de distorção de logits foi estatisticamente nula ($\Delta \text{MSE} = -1.5\%$), demonstrando que sem a rotação ortogonal de Hadamard (preconizada pelo KVarN / QuaRot), a sensibilidade de quantização é dominada por outliers de canais distribuídos ao longo de todo o contexto.

---

## 📊 2. Tabela de Avaliação Multicontexto

| Contexto ($L$) | Política | Economia de VRAM | Logits MSE | Cosine Similarity | Needle Retrieval (2k, 50%) |
|:---:|---|:---:|:---:|:---:|:---:|
| **256** | `FP16_UNIFORM` | 0.0% (Base) | 0.0000 | 1.0000 | PASS |
| **256** | `INT4_UNIFORM` | 72.0% | 15.3125 | 0.3691 | PASS |
| **256** | `PRECISION_TAIL_64` | 52.9% | 15.3125 | 0.3652 | PASS |
| **256** | `PRECISION_TAIL_128` | 34.9% | 15.5000 | 0.3555 | — |
| **1024** | `FP16_UNIFORM` | 0.0% (Base) | 0.0000 | 1.0000 | PASS |
| **1024** | `INT4_UNIFORM` | 72.0% | 26.5000 | 0.1445 | PASS |
| **1024** | `PRECISION_TAIL_64` | 67.2% | 26.3750 | 0.1484 | PASS |
| **1024** | `PRECISION_TAIL_128` | 62.7% | 26.1250 | 0.1572 | — |
| **4096** | `FP16_UNIFORM` | 0.0% (Base) | 0.0000 | 1.0000 | PASS |
| **4096** | `INT4_UNIFORM` | **72.0%** | 12.1875 | 0.3047 | PASS |
| **4096** | `PRECISION_TAIL_64` | **70.8%** | 12.3750 | 0.2930 | PASS |
| **4096** | `PRECISION_TAIL_128` | 69.7% | 12.4375 | 0.2891 | — |

---

## 🔬 3. Descoberta Arquitetural e Decisão

1. **Topologia Híbrida do Qwen 3.5**:
   - A inspeção em baixo nível revelou que o modelo possui apenas **6 camadas de Full Attention** (camadas 3, 7, 11, 15, 19, 23), enquanto as outras 18 camadas são **Linear Attention (Gated DeltaNet)** com estado recorrente fixo $O(1)$.
   - Como a maior parte da propagação sequencial é amortizada pelas camadas recorrentes, a distorção do KV nas 6 camadas de atenção plena se propaga de forma global e não se concentra apenas na cauda de 64 tokens.
2. **Direção Teórica**:
   - A simples demarcação posicional de cauda é insuficiente para quantização sub-4bit sem controle espectral de outliers.
   - O avanço na compressão de KV deve seguir para **`REP-03` (KVarN Offline Codec com Rotação de Hadamard)** para suprimir outliers antes do agrupamento numérico.

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/REP-02-PRECISION-TAIL-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Script do Probe**: [`tools/probes/rep02_precision_tail.py`](../../tools/probes/rep02_precision_tail.py)
- **Agente Executor**: Antigravity
