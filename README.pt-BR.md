# tare.tools.local-labs

*English version: [README.md](README.md).*

[![CI](https://github.com/augusto-scarvalho/tare.tools.local-labs/actions/workflows/ci.yml/badge.svg)](https://github.com/augusto-scarvalho/tare.tools.local-labs/actions/workflows/ci.yml)

O laboratório de pesquisa e a bancada de benchmarks do tare.tools para rodar modelos de linguagem de pesos
abertos localmente: uma GPU de consumo (uma RTX 3090 com 24 GB) e 64 GB de RAM no host. Ele guarda os harnesses
de benchmark, as sondas e as estatísticas usadas para qualificar modelos, quantizações e alavancas do motor de
inferência, e os relatórios que explicam os resultados.

As afirmações são medidas contra o piso de ruído da máquina (cerca de 2,3 % de dispersão pareada) com
estatística não paramétrica: testes de sinal exatos, intervalos de confiança por bootstrap e delta de Cliff. Os
candidatos avançam por portões ordenados: elegibilidade, depois correção, depois qualidade, depois desempenho.

## O que é publicado

Este repositório publica as ferramentas, os testes e os relatórios. Os registros brutos das execuções (`runs/`),
o backlog de pesquisa legível por máquina e seus pacotes, os handoffs e registros de auditoria dos agentes, os
pacotes de evidência brutos, os scripts que só funcionam nas máquinas do mantenedor, os datasets de terceiros e
os pesos de modelos ficam privados. Os relatórios ainda citam os registros que resumem, mas esses registros não
estão neste repositório.

## Relação com os outros repositórios do tare.tools

- O [tare.tools.os](https://github.com/augusto-scarvalho/tare.tools.os) é o produto de agentes. Ele não fixa nem
  importa este repositório; alguns dos seus guias apontam para resultados publicados aqui, como o
  [estudo de localidade e custo de carregamento](docs/research/LOCALITY_COST_QUALIFICATION.md).
- O serviço do nó de GPU (gateway de modelos, lease compartilhado da GPU e gancho do ComfyUI) é o `node/` do
  [tare.tools.os](https://github.com/augusto-scarvalho/tare.tools.os/tree/main/node).
  Ele foi separado deste repositório em 2026-10-09. O catálogo de modelos qualificados dele cita o trabalho de
  qualificação feito aqui, e o `tools/agents/modelctl.py` precisa dele instalado.
- O [slop.cpp](https://github.com/augusto-scarvalho/slop.cpp), um fork do llama.cpp, é dono do código do motor,
  das flags de execução e dos builds. Este repositório é dono do desenho dos experimentos, da estatística e das
  decisões de promoção; [`patches/`](patches/README.md) só guarda recibos de patches arquivados.

## Requisitos

- Python 3.12 (a versão que a CI testa) e Git.
- Os testes precisam só de `pytest` e `numpy`. Eles não usam GPU, modelo, serviço de rede nem registro privado.
- Rodar os experimentos em si exige uma GPU NVIDIA, um build do slop.cpp ou do llama.cpp e os arquivos dos
  modelos. A maioria dos scripts foi escrita para o WSL2 (Ubuntu 24.04) numa estação Windows e usa por padrão o
  layout `~/src/slop.cpp*` e `~/models/...`; ajuste esses caminhos ou passe as opções do script.

## Instalar e testar

```bash
git clone https://github.com/augusto-scarvalho/tare.tools.local-labs.git
cd tare.tools.local-labs
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install pytest==9.1.1 numpy==2.5.1
python -m compileall -q src tools tests benchmark_harness_qa.py
python -m pytest -q
python tests/benchmark_harness/benchmark_harness_selftest.py
```

O último comando qualifica o harness de benchmark em poucos segundos sem GPU e termina com
`LAB-QA-001: 23/23 passed`; ele reescreve `tests/benchmark_harness/benchmark_selftest_report.json`. Esses são os
passos que a CI roda. A CI também roda o portão do backlog de pesquisa, que é pulado quando o backlog privado não
está presente.

Mudanças materiais precisam de um novo item em `## Unreleased` no [`CHANGELOG.md`](CHANGELOG.md). Instale o hook
de pre-push que confere isso com `python tools/install_git_hooks.py`; veja
[`docs/CHANGELOG_POLICY.md`](docs/CHANGELOG_POLICY.md).

## Datasets de benchmark

As cópias do GSM8K, do HumanEval+ e do MBPP+ que os benchmarks leem não são redistribuídas aqui. Gere-as de novo
com os exportadores (eles precisam do pacote `datasets` do Hugging Face, ou do `evalplus` para o HumanEval+ e o
MBPP+): `tools/analysis/export_gsm8k.py` (escreve `workloads/gsm8k.jsonl`),
`tools/analysis/export_humaneval_plus.py` (escreve `tools/benchmarks/workloads/humaneval_plus.jsonl`) e
`tools/benchmarks/export_mbpp_plus.py SAIDA`.

## Estrutura

```text
src/model_lifecycle/   pacote Python: portões de promoção e estatística, QA de benchmark, coletores,
                       planejador e guarda de execuções, geradores de relatório, adaptadores de servidor
tools/                 benchmarks, sondas, análise e portões (índice: tools/README.md)
ops/                   playbooks das campanhas: bring-up do Qwen3.8, modelos recorrentes, serving,
                       executor WSL (ops/README.md)
docs/                  relatórios, catálogo de pesquisa e contratos de método (mapa: docs/README.md)
config/                registro de modelos qualificados, política de prioridade do backlog, casos de smoke
                       do choice helper
workloads/             registro de identidade do MBPP+ e fixtures sintéticas de VLM
tests/                 testes unitários e a qualificação determinística do harness de benchmark
patches/, reports/     patch arquivado do motor e tabelas de evidência geradas
```

## Resultados

O [catálogo de pesquisa](docs/RESEARCH_CATALOG.md) indexa todas as campanhas; os principais resultados são:

- **Quantização.** Para o Qwen3.8-27B, o `UD-Q2_K_XL` (9,9 GB) manteve a qualidade em código e no MATH-500 L5 e
  a recuperação em contexto longo profundo, enquanto o `IQ2_M` perdeu agulhas a partir de 32k
  ([fronteira de quantização](ops/qwen38-bringup/QUANT_FRONTIER_CAMPAIGN.md)).
- **Orçamento de raciocínio.** Num conjunto exato de 60 problemas do HumanEval+, o Qwen3.8 em modo instruct fez
  95,0 %, acima do raciocínio baixo (93,3 %) e médio (88,3 %); o esforço máximo padrão truncou 31 das 60
  respostas ([relatório de bring-up](ops/qwen38-bringup/README.md),
  [curva de orçamento](ops/qwen38-bringup/BUDGET_CURVE_CLOSURE.md)).
- **Cache KV.** O KV simétrico `q4_0`/`q4_0` ficou no caminho fundido da GPU com recuperação de agulhas sem
  perda; o assimétrico `q8_0`/`q4_0` caiu para o caminho alternativo e custou cerca de 57 % do throughput
  ([quantização de KV A3](docs/campaigns/a3-kv-quant/A3_KV_QUANT.md)).
- **Decodificação especulativa.** A cabeça MTP embutida com `--spec-draft-n-max 3` deu cerca de 2,1x de decode
  em código para o build e o modelo registrados; requalifique para cada modelo e build.
- **Alavancas do motor.** Buffers de KV fixados no host, prefetch de experts, cache de experts quentes e uma
  varredura GDN em blocos foram medidos na RTX 3090; veja o [relatório do fork](docs/research/FORK.md) para
  saber quais ajudam e onde.

O lançador de serving com essas configurações é [`ops/qwen38-bringup/serve.sh`](ops/qwen38-bringup/serve.sh).

## Licença

Licenciado sob a [Apache License, Version 2.0](https://opensource.org/licenses/Apache-2.0).
