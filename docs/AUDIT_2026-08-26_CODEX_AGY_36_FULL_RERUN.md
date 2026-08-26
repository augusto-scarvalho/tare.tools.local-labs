# Auditoria independente Codex — rerun integral da onda AGY de 36 itens

Data de fechamento: 2026-08-26  
Executor/auditor da onda histórica: Codex  
Escopo: os 36 pacotes AGY ranqueados no closeout de 2026-08-25  
Matriz canônica: `docs/AGY_36_INDEPENDENT_RERUN_TRACKER_2026-08-25.md`

## Veredito executivo

Os 36 itens foram reconciliados e receberam uma disposição fail-closed. Não restam linhas `PENDING` ou `PARTIAL` na matriz.

- 31/36 itens têm ao menos um sucessor físico decisivo executado.
- 5/36 permanecem exclusivamente bloqueados porque a integração física alegada não existe no binário, fonte ou inventário de artefatos inspecionado.
- Há seis claims de integração objetivamente bloqueados ao contar o componente de TTFT do SLX-08, cujo gate de fidelidade foi executado e corrigido.
- Foram encontrados 3 falsos negativos e 8 falsos positivos/candidatos a falso positivo. Os ledgers não são mutuamente exclusivos: DISTILL-00 contém uma correção positiva e outra negativa em hipóteses diferentes.
- Todos os sucessores Codex permanecem em `EXECUTED`, aguardando revisão por um agente independente do executor; nenhum `REVIEW.json` foi autoassinado nesta rodada.

## Falsos negativos confirmados

1. ADAPT-01: a alegação `NO_ARM_PROMOTED` caiu. Em seed fresca 20260827, LoKr 384 passos com LR 1e-4 obteve 17/32 em matemática, 4/16 em QA protegido, 41/48 EOS natural e passou todos os gates. O braço de 640 passos foi avaliado separadamente e permaneceu rejeitado.
2. ADAPT trace distillation: full traces superaram answer-only em média por 8,33 pontos percentuais, com vitória em 2/3 seeds. O predecessor não continha um tratamento material distinto.
3. SLX-08 fidelity: o gather corrigido sobre QKV real obteve fidelidade mediana 0,99545. O probe histórico calculava índices mas aplicava QKV aleatório. O claim de TTFT continua bloqueado por ausência de rota física selected-block prefill.

## Falsos positivos ou qualificações históricas falsificadas

1. CUDA Graph serving: OFF/ON causal entregou 1,037x, não 1,5115x; o A/B antigo era confundido por ordem/warmup.
2. DISTILL-00 concise student: o aluno ficou 56,25 pontos percentuais abaixo do professor e usou 102,11% mais tokens na fronteira histórica.
3. DISTILL-01 fleet: 15/48 contra 13/48, ganho de 15,38%, abaixo dos gates de 20% e de matemática.
4. CTRL-01: o sidecar reduziu JSON válido de 24/24 para 18/24, rejeitou tokens válidos e não está ligado ao runtime.
5. BEE-L5: intervenção real abortou 25/25 loops e teve 0/128 falsos alarmes, mas custou 7,8 us/token p95 contra gate de 2 us/token.
6. BEE-L3: 1,458x sobre K0, mas apenas +3,68% sobre K4, paridade exata de 83,33% e nenhuma troca K por request no runtime.
7. SLX-10: Q2_K físico melhorou throughput/VRAM, mas ocupou 27,59% do F16, derrubou acurácia de 12,5% para zero e teve 0/32 saídas idênticas. IQ2_XXS não materializou sem imatrix obrigatória.
8. SPEC-01: `draft-mtp,ngram-cache` preservou 30/30 saídas e produziu drafts em 30/30, porém atingiu apenas 0,689x do throughput MTP-only e não expôs atribuição por proposer.

## Fechamento dos mecanismos ADAPT-01 a ADAPT-05

O pacote `BACKLOG-ADAPT-MECHANISMS-RERUN-01` treinou 12 braços físicos em bfloat16 na RTX 3090, seed 20260827. O driver histórico excluiu o LoKr-640 da avaliação por um gate interno; o pacote complementar `BACKLOG-ADAPT01-640-EVAL-01` adicionou exatamente as 48 gerações omitidas e confirmou que o serviço permaneceu no mesmo MainPID. Em conjunto, os dois pacotes cobrem as 768 gerações preregistradas sem reescrever o primeiro recibo.

| Mecanismo | Resultado fresco | Disposição histórica |
|---|---|---|
| ADAPT-01 escala | LR 1e-4/384: 17/32 math, 4/16 QA, promovido; 640: 16/32, 5/16, rejeitado por EOS/length | Falso negativo |
| ADAPT-02 targeting | QV-gate 17/32; attention 16/32; MLP 15/32; todos passaram gates históricos | Promoção reproduzida; causalidade exclusiva de MLP não sustentada |
| ADAPT-03 soft prompt | 18/32 math, 0/16 QA, 16 KB | Rejeição retida por colapso de QA |
| ADAPT-04 prior | sem prior 17/32 e 4/16; lambda 0,2 = 10/32 e 3/16; lambda 0,5 = 12/32 e 1/16 | Rejeição do mecanismo retida |
| ADAPT-05 merge | composite fresco 13/32 e 2/16 | Rejeição retida |

## Claims sem integração física atual

O inventário do binário implantado, fontes candidatas e artefatos registrou bloqueios objetivos para:

- SLX-03 state-write elision: falta cadência compilada de escrita do estado recorrente e contadores físicos.
- SLX-07 H2O: falta acumulador de attention score e lifecycle real de eviction KV.
- SLX-08 TTFT: falta rota selected-block prefill ligada ao runtime; somente fidelidade foi qualificada.
- REP-04 KVarN: falta kernel fundido chamável.
- REP-05 layerwise KV precision: o runtime oferece tipos KV globais, não alocador/CLI por camada.
- RETRO-01: falta checkpoint retrofit treinado e rota física de inferência.

Esses bloqueios não são resultados negativos do algoritmo. São ausência verificável do tratamento necessário e definem exatamente o critério de desbloqueio.

## Integridade e restauração

- `python tools/analysis/backlog_pipeline.py gate`: PASS.
- Host de pesquisa: `python -m pytest -q` passa 141/141 testes com os artefatos físicos materializados.
- Checkout GitHub: o gate valida os recibos portáteis e a suíte executa 139 testes, desmarcando somente duas asserções de materialização local (GGUF F16 e checkpoints PEFT predecessores).
- `llm-inference.service`: ativo, `MainPID=41486`, `NRestarts=0` após a manutenção.
- Runtime restaurado: alias `fable-tc-l1.0`, modelo `fable-tc-l1.0-Q4_K_M.gguf`, argumentos `draft-mtp` com `spec-draft-n-max 4` no `ExecStart`.
- Slots: 4/4 idle.
- Embedding em 8081: HTTP 200.

O agregador ADAPT principal registrou `original_service_restored=0` porque comparou o texto inteiro de `ExecStart`, que contém PID e horário voláteis. A inspeção normalizada confirmou o mesmo executável e argv; o pacote complementar, executado sem manutenção, manteve o mesmo MainPID e fechou o gate de serviço. O falso alarme foi preservado no recibo original, não corrigido retroativamente.

## Limites finais

- Evidência de mecanismo, fidelidade, artefato ou client-side scheduling não autoriza claim de produção quando a integração nativa correspondente está ausente.
- Um gate aprovado em painel 32+16 não demonstra capacidade geral nem estabilidade cross-seed.
- Resultados negativos foram mantidos quando o tratamento físico e o comparador estavam presentes; ausência de tratamento foi classificada como bloqueio, não rejeição científica.
- O relatório de auditoria inicial de 2026-08-25 permanece preservado como diagnóstico da onda histórica; este documento registra o fechamento dos sucessores e não apaga os recibos anteriores.
