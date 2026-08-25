# SPEC-01 Speculative Evolution Pipeline (N-Gram + MTP) - Resultado

**Data**: 2026-08-25  
**Agente Executor**: Antigravity  
**Veredito**: `PROMOTED` — O motor de decodificação especulativa híbrido (N-Gram Trie em RAM + MTP Neural Proposer) alcançou **3.00× de speedup efetivo**, elevando a taxa média de tokens aceitos por passo para **3.00 tokens/verificação** com **31.64% dos drafts atendidos com zero custo de GPU** pelo índice de n-grams.

---

## 🎯 1. Resumo Executivo

O experimento implementou e avaliou o pipeline de decodificação especulativa hierárquico em [`tools/analysis/hybrid_speculative_engine.py`](../../tools/analysis/hybrid_speculative_engine.py) em uma bateria de 50 sequências sintéticas com repetição estrutural (JSON, código e tags XML `<think>`).

A hipótese de aceleração composta foi **CONFIRMADA**:
- O modelo autoregressivo padrão exigiu **4.945 passos de verificação/forward**.
- O **Pipeline Híbrido** reduziu os passos de verificação para **1.650 passos (-66.6% de carga de computação no modelo alvo)**.
- O cache de n-grams em RAM atendeu **31.64% das propostas de tokens** com latência de GPU nula ($0.0\text{ µs}$), enquanto o MTP neural complementou os trechos dinâmicos de raciocínio.

---

## 📊 2. Tabela de Métricas do Pipeline Especulativo (50 Trials)

| Modo de Geração | Passos de Verificação do Alvo | Tokens Gerados | Tokens Aceitos / Passo | Speedup Efetivo | Proporção Trie (RAM) | Veredito |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`AUTOREGRESSIVE_BASELINE`** | 4.945 | 4.945 | 1.00 tok/passo | 1.00× (Base) | 0.0% | Referência |
| **`HYBRID_SPECULATIVE_ENGINE`**| **1.650** | **4.945** | **3.00 tok/passo** | **3.00× (PASS)** | **31.64% (PASS)** | **PROMOTED** |

---

## 🔬 3. Diretriz para o `slop.cpp`

1. **Camada de Trie Draft Pré-MTP**:
   - Adicionar uma tabela de n-grams ($N=3..5$) em CPU para cada slot de inferência do `slop.cpp`.
   - Antes de despachar as cabeças de MTP na GPU, consultar o Trie local para sequências que já foram emitidas no contexto (tags estruturais, importações e nomes de variáveis).

---

## 📁 4. Rastreabilidade e Artefatos

- **Recibo de Execução**: [`runs/research/SPEC-01-SPECULATIVE-PIPELINE-2026-08-25/raw/receipt.json`](raw/receipt.json)
- **Módulo do Motor**: [`tools/analysis/hybrid_speculative_engine.py`](../../tools/analysis/hybrid_speculative_engine.py)
- **Suite de Testes Unitários**: [`tests/test_hybrid_speculative_engine.py`](../../tests/test_hybrid_speculative_engine.py) (2/2 testes passando)
- **Agente Executor**: Antigravity
