# MiniCPM5-2B original: compactação nativa e chamadas de ferramentas

Data: 2026-09-14. Continuação da investigação do MiniCPM, com o **original da
OpenBMB**, conforme esclarecimento do operador. Não foi usado o derivado NPU2.

## Resultado

A compactação determinística funciona no Runtime nativo, inclusive com o modelo
real, sem uma chamada auxiliar de inferência. Isso não qualifica o MiniCPM como
implementador autônomo nem demonstra economia integral. O modelo continua
irregular em edição, interpretação de ferramentas e cumprimento das etapas.

Foram seis sessões experimentais, 140 gerações físicas incluindo quatro chamadas
de sumarização. Nenhum teto agregado de chamadas, tempo, estagnação ou compactação
foi aplicado; retries foram zero. Não houve reparo manual do código produzido.

## Configuração observada

- Repositório oficial: `openbmb/MiniCPM5-2B-GGUF`.
- Revisão: `2079a22f3beaa4e306449978533478fe0522f4b3`.
- Arquivo: `MiniCPM5-2B-Q8_0.gguf`.
- SHA-256: `c5415f8989bf88a8288f1b55a3cc371af53c07b0faa220a63bd7a990cfaba078`.
- slop.cpp `b10165-71676e46c`, template Jinja do artefato, thinking habilitado.
- `/props`: janela efetiva **32768**, uma sessão; não é uma afirmação de limite
  máximo do modelo original. Identidade da resposta conferida em todas as gerações.
- GPU, flash attention, KV K/V `q8_0`, batch 512, ubatch 128; seed 42;
  temperature 1, top-p .95, top-k 40, min-p 0, repeat penalty 1.05, repeat-last-n 64.
- Saída por resposta 4096; trigger .8, target .5, margem 64, último turno completo
  preservado. O cache do servidor permaneceu disponível entre execuções.

## Implementação

Runtime: `ContextPolicy.compaction` aceita `model` (padrão compatível) ou
`deterministic`. O projetor é injetado pelo host; não há import do Kernel/OS.
O candidato inteiro é contado antes da ativação atômica. Instruções, fontes
obrigatórias, mensagens explícitas do usuário, turnos recentes e o journal
original permanecem preservados. Estado excessivo não é descartado silenciosamente
para forçar admissão. A retomada não inventa inferência nem repete efeitos.

Kernel: o projetor `native-task-state/1` deriva arquivos/hashes observados,
edições, verificações, buscas, recusas e efeitos incertos. Ele declara que não
resume decisões semânticas anteriores. Recusados por argumentos incompletos agora
informam campos ausentes/inesperados e repositórios válidos. As descrições finais
explicam caminhos relativos à raiz e números de linha apenas de apresentação.
Sumarização OpenAI-compatible recebe instruções próprias e não oferece ferramentas;
o contrato Converse não foi alterado por essa mudança.

OS: seleção explícita em `context_policy.compaction`. O novo modo é opcional;
os testes não justificam torná-lo obrigatório para todo trabalho.

Código preservado localmente, sem publicação ou instalação global nesta rodada:

| Repositório | Base | Entrega |
|---|---|---|
| agent-runtime | `2554110` | `978bd981` |
| kernel | `cac5099` | `c027c60a` |
| os | `ee12f0f` | `aac4731f` |

Branch comum: `fix/deterministic-context-20260914`. A implementação deve ser
integrada/publicada com versões compatíveis dos três pacotes; mínimos publicados
anteriores não entregam o novo modo. Os checkouts antigos com WIP foram preservados.

Validação final: Runtime **209** testes, Kernel **109** focados, OS **54** de
integração nativa; `compileall` do Runtime e `git diff --check` aprovados.

## Sessões reais

Tarefas dirigidas, com caminhos explícitos: nenhuma comparação de descoberta,
SpecGraph ou busca ampla. Verificadores congelados antes de cada execução.

O teste de pressão usa três arquivos de 200 linhas com marcadores, pedidos em
etapas e edição final de `answer.json`. É um ensaio sintético de acumulação e
recuperação de contexto. O verificador confere o arquivo; o journal também foi
inspecionado para detectar edição prematura. Essa auditoria adicional foi feita
após as execuções, sem converter falhas em passes.

| Variante | Tarefa/modo | Gerações | Segundos de sessão | Tokens entrada + saída | Compactações / resumos LLM | Resultado |
|---|---|---:|---:|---:|---:|---|
| Inicial | Unicode / determinístico | 4 | 7.17 | 6414 | 0 / 0 | Arquivo e 6 grupos de testes corretos |
| Inicial | Duração / determinístico | 63 | 131.14 | 836600 | 0 / 0 | Falhou, terminou propondo entrega com SyntaxError |
| Inicial | Pressão / determinístico | 19 | 49.45 | 273195 | 2 / 0 | Arquivo correto; sem edição antecipada |
| Inicial | Pressão / modelo | 11 | 47.81 | 140226 | 1 / 1 | Resumo válido; edição recusada por caminho incorreto; arquivo vazio |
| Descrições claras | Pressão / determinístico | 23 | 77.56 | 339399 | 3 / 0 | Arquivo correto, mas tentou editar antes da etapa autorizada |
| Descrições claras | Pressão / modelo | 20 | 80.31 | 264613 | 2 / 3 | Arquivo correto, editado prematuramente; sessão bloqueada por candidato grande demais |

As duas últimas execuções não são passes completos. A melhora pontual de chamada
não equivale a melhora demonstrada da tarefa inteira. As descrições finais
eliminaram nesses dois ensaios as recusas por campos ausentes e caminho com prefixo
de repositório, mas não eliminaram erros de comportamento.

O parser recebeu 7 exemplos válidos e 14 inválidos, incluindo `2ms`, `1hour`,
`2ss`, `2m garbage` e `1hfoo`. Houve 29 recusas por argumentos ausentes e cinco por
search sem correspondência única. O código incorreto e os argumentos incompletos
vieram do modelo; as edições recusadas não foram aplicadas. A proposta final não
foi tratada como aprovação do trabalho.

## Compactação e custo

Na primeira sessão determinística com pressão, os pedidos passaram de
**29159 para 10598** e de **28610 para 10532** tokens. Na variante final, foram
**29312 → 10528**, **30308 → 10730** e **29432 → 10642**. Não houve resumo por LLM.
A variante final com modelo ativou dois resumos; o terceiro candidato não coube,
pois o último turno continha três leituras grandes que deveriam permanecer inteiras.

O gatilho configurado foi 26214 tokens (80%). A avaliação ocorre antes da próxima
chamada, depois de receber o resultado da ferramenta; uma leitura pode levar o
pedido de menos de 80% para 89–92%. Esses pedidos foram compactados antes do envio,
não despachados com o contexto excessivo.

O checkpoint determinístico descartou prosa e provocou releituras; um dos resumos
por modelo reteve os marcadores corretamente. Portanto, “zero chamada de resumo”
não significa “menor custo total”. As taxas ponderadas de decodificação foram
aproximadamente 142–190 tok/s; são taxas do servidor, não do fluxo completo.

Tokens de argumentos de ferramentas estão na saída do provedor, e resultados de
ferramentas aparecem nas entradas seguintes. Não foram somados novamente. Resumos
reais estão no mesmo total. Cache, contagens e efeitos permanecem nos recibos.
O cronômetro de sessão exclui preparação dos fixtures e verificação independente
final; energia não foi medida. O ensaio não sustenta porcentagem de economia
integral, nem inferência causal a partir de uma trajetória por variante.

## Evidência e próximo uso

Resumo, patches, referências Git e recibos ficam em
`.reconciliation/evidence-store/reports/native-deterministic-20260914/`.
`RESULTS.json` identifica cada execução; requests, replies, journals, perfis e
verificadores são mantidos no arquivo consolidado indicado por `retention.json`.

O servidor temporário foi encerrado. O backend do Qwen permaneceu saudável com o
mesmo PID. A cópia duplicada de 2.68 GB dos pesos foi removida após verificar o
arquivo histórico que permite restaurá-la, sem criar outro arquivo de pesos.

Uso prudente desta evidência: manter o MiniCPM como candidato a tarefas pequenas
com resultado verificável e escopo estreito. Ainda falta qualificar sua qualidade
como subagente no paralelismo real. Não tornar compactação determinística padrão
universal nem tentar compensar toda falha de raciocínio com mais regras no harness.
