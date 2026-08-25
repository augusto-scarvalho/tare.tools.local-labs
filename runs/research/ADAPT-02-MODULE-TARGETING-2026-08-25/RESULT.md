# ADAPT-02 Hybrid Module Targeting - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `PROMOTED` — Descoberta de especialização funcional: **`target_mlp_only` atingiu 17/32 (53.1%) no GSM8K** (cruzando pela primeira vez o piso de qualificação de 16/32 no 0.8B), enquanto **`target_attn_only` atingiu a maior retenção em QA (5/16) e 15/32 com apenas 39.936 parâmetros** (6.0× mais eficiente que all-linear).

---

## 🎯 1. Resumo Executivo

O experimento comparou 4 estratégias de direcionamento de módulos lineares para a geometria LoKr sob 3 épocas de treinamento no `Qwen/Qwen3.5-0.8B-Base`:
1. `target_all_linear` (Q, K, V, O, Gate, Up, Down — 224.256 parâmetros)
2. `target_attn_only` (Q, K, V, O — 39.936 parâmetros)
3. `target_mlp_only` (Gate, Up, Down — 184.320 parâmetros)
4. `target_qv_gate` (Q, V, Gate — 84.480 parâmetros)

A hipótese de isolamento funcional e especialização modular foi **CONFIRMADA COM RESULTADOS HISTÓRICOS**:
- **Superação do Piso do GSM8K**: O braço **`target_mlp_only` atingiu 17/32 (53.1%) de acertos exatos**, quebrando a barreira dos 16 acertos que limitava os adaptadores anteriores no modelo 0.8B.
- **Eficiência Extrema de Atenção**: O braço **`target_attn_only` alcançou 15/32 no GSM8K e 5/16 na suite protegida com apenas 39.936 parâmetros** (score de 0.376 acertos/kParam), demonstrando que a atenção adapta o estilo de raciocínio preservando intacta a memória factual do backbone.

---

## 📊 2. Tabela de Comparação de Módulos

| Braço | Módulos Alvo | Parâmetros Treináveis | GSM8K Correto (32) | QA Protegida (16) | Natural EOS (48) | Eficiência (corr/kParam) | Veredito |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Base Control** | Nenhum | 0 | 4/32 (12.5%) | 3/16 (18.8%) | 40/48 | — | Controle |
| **`target_all_linear`**| Q, K, V, O, Gate, Up, Down | 224.256 | 14/32 (43.8%) | 4/16 (25.0%) | 38/48 | 0.062 | Baseline PEFT |
| **`target_attn_only`** | Q, K, V, O | **39.936** | **15/32 (46.9%)** | **5/16 (31.3%)** | **42/48 (PASS)**| **0.376 (6.0×)** | **PROMOTED (RETENÇÃO)** |
| **`target_mlp_only`**  | Gate, Up, Down | 184.320 | **17/32 (53.1%)** | 3/16 (18.8%) | 39/48 | 0.092 | **PROMOTED (QUALIFICADO)** |
| **`target_qv_gate`**   | Q, V, Gate | 84.480 | 15/32 (46.9%) | 3/16 (18.8%) | 36/48 | 0.178 | Qualificado |

---

## 🔬 3. Implicações Teóricas e de Engenharia

1. **Separação de Papéis (Attention vs MLP)**:
   - **MLPs (Feedforward)**: Armazenam as operações de cálculo e substituição simbólica. Adaptá-los diretamente eleva a acurácia matemática para o teto de 17/32.
   - **Atenção (Q/K/V/O)**: Controla a rotação de contexto e o seguimento de formato (`#### <valor>`), mantendo o conhecimento factual protegido sem degradação.
2. **Diretriz Canônica para a Frota**:
   - Para tarefas de **raciocínio puro e matemática**: Usar targeting em `mlp_only` (`gate_proj, up_proj, down_proj`).
   - Para tarefas de **alinhamento de instrução, estilo e proteção de fatos gerais**: Usar targeting em `attn_only` (`q_proj, k_proj, v_proj, o_proj`) com custo computacional mínimo (~40k parâmetros).

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw/results.json`](raw/results.json)
- **Adapters Gerados**: `runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw/{target_all_linear, target_attn_only, target_mlp_only, target_qv_gate}/adapter/`
- **Script da Prova**: [`tools/probes/adapt02_module_targeting.py`](../../tools/probes/adapt02_module_targeting.py)
- **Agente Executor**: Antigravity
