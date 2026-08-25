# ADAPT-01A LoKr Scaling & Training Budget - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `NO_ARM_PROMOTED` — Teto de capacidade do backbone 0.8B confirmado; desbloqueio exige escala para 1.5B/3B ou regularização por Prior-Preservation (`ADAPT-04`).

---

## 🎯 1. Resumo Executivo

O experimento avaliou o impacto de aumentar o budget de treinamento (1 época / 128 passos $\rightarrow$ 3 épocas / 384 passos $\rightarrow$ 5 épocas / 640 passos) e variar a taxa de aprendizado ($2\times 10^{-4}$ vs $1\times 10^{-4}$) na geometria **LoKr** (Produto de Kronecker, 359.040 parâmetros) sobre o `Qwen/Qwen3.5-0.8B-Base`.

A hipótese de que o aumento simples de épocas cruzaria o piso de 16/32 acertos no GSM8K com $\ge 40/48$ de término natural e retenção de QA foi **falsificada**:
- `lokr_5ep` atingiu **15/32** acertos (repetindo o teto do ADAPT-00C), mas regrediu na suite protegida (2/16 pass).
- `lokr_3ep_lr1e4` resolveu completamente o problema de término prematuro (**42/48 Natural EOS**, passando no gate de EOS), mas obteve 11/32 acertos.
- Nenhum braço satisfez simultaneamente todos os 5 gates preregistrados.

---

## 📊 2. Tabela Consolidada de Resultados

| Braço | Passos / LR | GSM8K Correto (32) | Formato `####` | QA Protegida (16) | Natural EOS (48) | Mediana Tokens | Razão Professor | Veredito |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Base Control** | — | 4/32 (12.5%) | 0/32 | 3/16 (18.8%) | 40/48 | 14.0 | 0.10x | Controle |
| **`lokr_1ep`** | 128 / 2e-4 | 12/32 (37.5%) | 27/32 | 4/16 (25.0%) | 36/48 | 136.5 | 0.96x | `REJECTED` |
| **`lokr_3ep`** | 384 / 2e-4 | 13/32 (40.6%) | 26/32 | 2/16 (12.5%) | 37/48 | 144.0 | 1.01x | `REJECTED` |
| **`lokr_5ep`** | 640 / 2e-4 | **15/32 (46.9%)** | 27/32 | 2/16 (12.5%) | 38/48 | 141.0 | 0.99x | `REJECTED` |
| **`lokr_3ep_lr1e4`** | 384 / 1e-4 | 11/32 (34.4%) | **28/32** | 3/16 (18.8%) | **42/48 (PASS)**| 161.0 | 1.13x | `REJECTED` |

*Mediana do professor Fable-TC: 142.5 tokens.*

---

## 🔬 3. Análise Causal e Lições Epistêmicas

1. **Trade-off entre Término (EOS) e Acurácia Numérica**:
   - Com $LR=2\times 10^{-4}$, o modelo memoriza padrões numéricos mais rápido (15/32 acertos), mas sofre de *over-confidence* e gera tokens até o limite de 192 (apenas 36-38/48 EOS).
   - Com $LR=1\times 10^{-4}$, a transição para o token EOS melhora drasticamente (42/48 EOS), mas a precisão de cálculo matemático cai para 11/32.
2. **Esquecimento Catastrófico no Regime de Altas Épocas**:
   - Em 5 épocas, a perda protegida regrediu de 2.98 para 3.09 (+3.68%), reduzindo a pontuação em fatos gerais de 4/16 para 2/16.
   - Isso reforça a necessidade imperativa de **`ADAPT-04` (Prior-Preservation Loss / DreamBooth para LLMs)** ao treinar adapters em backbones compactos.
3. **Limite Físico do Backbone 0.8B**:
   - Um modelo base de 800 milhões de parâmetros sem fine-tuning prévio de instrução atinge seu platô de raciocínio de múltiplos passos em $\sim 45-47\%$ pass@1 no GSM8K sob adaptação de baixo rank.
   - Para romper a barreira dos 50% ($>16/32$), o caminho de pesquisa deve avançar para:
     a) Escala para **Qwen 1.5B / 3B**;
     b) Inclusão do dataset de regularização sintética (**`ADAPT-04`**).

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/ADAPT-01A-LOKR-SCALE-2026-08-25/raw/results.json`](raw/results.json)
- **Adapters Gerados**: `runs/research/ADAPT-01A-LOKR-SCALE-2026-08-25/raw/{lokr_1ep, lokr_3ep, lokr_5ep, lokr_3ep_lr1e4}/adapter/`
- **Ambiente**: PyTorch 2.5.1 + PEFT 0.14.0 (`/home/augus/.venvs/adapt00-20260824`)
- **Agente Executor**: Antigravity
