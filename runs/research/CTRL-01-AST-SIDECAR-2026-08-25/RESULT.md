# CTRL-01 ControlNet / AST Grammar Sidecar - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `PROMOTED` — O módulo de validação sintática incremental AST/JSON garantiu **100.0% de validade sintática** nas saídas estruturadas (eliminando 28 violações de parsing) com sobrecarga média de validação de apenas **7.88 µs por token**.

---

## 🎯 1. Resumo Executivo

O experimento avaliou o sidecar de controle gramatical em tempo real implementado em [`tools/analysis/ast_grammar_sidecar.py`](../../tools/analysis/ast_grammar_sidecar.py) através de 50 testes de geração com injeção de ruído sintático.

A hipótese de garantia sintática com custo de latência desprezível foi **CONFIRMADA**:
- O fluxo desprovido de restrições (*Unconstrained*) sofreu colapso de parsing em 56% das tentativas (**44.0% de validade sintática**).
- O **AST Grammar Sidecar** interceptou 28 transições ilegais, elevando a taxa de sucesso para **100.0% (50/50 válidos em `json.loads`)**.
- A latência de verificação incremental consumiu apenas **7.88 µs/token** (muito abaixo do teto contratual de 500 µs).

---

## 📊 2. Tabela de Métricas do Sidecar Sintático (50 Trials)

| Modo de Geração | Validade Sintática de Parsing | Violações Interceptadas | Overhead Médio por Token (µs) | Veredito |
|---|:---:|:---:|:---:|:---:|
| **`UNCONSTRAINED_STREAM`** | 44.0% | 0 (Erros Vazados) | 0.00 µs (Base) | Baseline |
| **`AST_GRAMMAR_SIDECAR`**  | **100.0% (50/50)** | **28 (PASS)** | **7.88 µs (PASS)** | **PROMOTED** |

---

## 🔬 3. Diretriz para o `slop.cpp`

1. **Ativação de Máscara de Logits em Modos JSON**:
   - Integrar o analisador incremental no pipeline de amostragem (`temperature/top_p`) do endpoint `/v1/chat/completions` quando `response_format={"type": "json_object"}` estiver ativo.

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/CTRL-01-AST-SIDECAR-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Módulo do Sidecar**: [`tools/analysis/ast_grammar_sidecar.py`](../../tools/analysis/ast_grammar_sidecar.py)
- **Suite de Testes Unitários**: [`tests/test_ast_grammar_sidecar.py`](../../tests/test_ast_grammar_sidecar.py) (3/3 testes passando)
- **Agente Executor**: Antigravity
