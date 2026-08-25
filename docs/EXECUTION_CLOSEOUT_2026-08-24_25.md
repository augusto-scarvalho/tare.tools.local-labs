# Fechamento consolidado de execução — 2026-08-24/25

Este documento consolida o trabalho executado após o reboot do Windows, a
qualificação do baseline da RTX 3090, o fechamento do backlog HauhauCS/FastMTP/
RWKV7 e a análise sequencial do transcript BeeLlama/slop.cpp/PEFT. Os recibos
por experimento continuam sendo a fonte canônica para dados brutos e hashes;
este fechamento existe para deixar explícitos efeitos persistentes, falhas,
restaurações e pendências em um único lugar.

## Resultado executivo

- Driver operacional mantido: NVIDIA Studio `591.86`.
- Fan Control V272 com `backupzin.json` é o proprietário das curvas de
  ventoinha. MSI Afterburner 4.6.5 mantém apenas V/F e offsets de clock.
- Nenhuma curva de fan, V/F, power limit ou clock foi alterada na continuação
  pós-reboot nem nos experimentos de pesquisa.
- Fable-TC continua sendo o default amplo em 8080. HauhauCS permanece um perfil
  opcional de coding, não um substituto geral.
- Nenhuma alteração foi feita em `slop.cpp`; APEX4 e FastMTP não foram portados.
- Nenhum adapter foi promovido. ADAPT-01 permanece bloqueado pelo gate
  comportamental.
- A fila não-soak executável está vazia. O restante depende dos gatilhos
  explícitos em
  [`research/REMAINING_EXPERIMENTS_2026-08-24.md`](research/REMAINING_EXPERIMENTS_2026-08-24.md).

## Linha do tempo e ações executadas

| Ordem | Bloco | O que foi executado | Resultado |
|---:|---|---|---|
| 1 | Driver A/B | Preservação do pacote 591.86 e perfis; baseline 591.86; instalação limpa 610.88; tentativa 610.47; rollback via `pnputil`; restauração WSL/serviços | 591.86 mantido; 610.88 rejeitado; 610.47 `INSTALL_FAILED / NOT_BENCHMARKED` |
| 2 | Pós-reboot | Diagnóstico de Fan Control/Afterburner, verificação Windows/WSL/PCIe/BAR1, benchmark curto 591.86 e sondas reais de 8080/8081/8082 | baseline operacional fechado; ownership de fan corrigido |
| 3 | Qwen3.8 | Painel agent/tool Fable/HauhauCS/vanilla; GSM8K-200 pareado; restauração Fable entre braços | HauhauCS 8/8 em agent core, mas `MATERIAL_MATH_LOSS`; Fable mantido |
| 4 | FastMTP | Gate documental após os resultados anteriores | `NO-GO BEFORE INSTALL`; nenhum download/build/install |
| 5 | RWKV7 | Revalidação de licença e painel congelado de 48 itens no runtime isolado existente | licença liberada, qualidade 13/48; `HOLD_QUALITY` |
| 6 | Transcript | Hash e análise de `tare-tools-beellama-slop-peft-conversation-transcript-2026-08-24.md`; reconciliação com recibos locais | backlog dependency-gated criado |
| 7 | BEE-L0 | Clone temporário e arqueologia read-only do fork BeeLlama contra upstream | importação integral rejeitada; 872 commits e 607 paths no delta |
| 8 | SLX-01A | Auditoria read-only das lacunas de lifecycle/route receipts em `slop.cpp` | `GAP_CONFIRMED`; nenhum código alterado |
| 9 | SLX-02 | Clone APEX4, ambiente Python/CUDA 12.4 side-by-side, build `sm_86`, testes de kernel e preflight do checkpoint público | kernels corretos; checkpoint publicado internamente truncado; sem port |
| 10 | ADAPT-00A | Download pinado do Qwen3.5-0.8B Base, freeze de dados/seed e smoke LoRA | mechanics `PASS`; melhoria de loss alvo 38,62% |
| 11 | BEE-L2 | Design do contrato e implementação/teste do scorer full-distribution de KV | `DESIGN_COMPLETE`; execução aguarda codec físico |
| 12 | ADAPT-00B | Matriz sequencial LoRA, DoRA, LoHa, LoKr, BOFT, IA3 e trainable tokens | seis passes; DoRA non-finite no step 0 |
| 13 | ADAPT-00C | Painel comportamental Base/LoRA/LoKr/IA3 em GSM8K e QA protegida | `NO_ARM_PROMOTED`; LoKr 15/32, abaixo do piso 16/32 e do gate EOS |
| 14 | Fechamento | Testes locais, commits, pushes, consulta dos workflows e inventário read-only do estado vivo | commits e dois workflows anteriores verdes; baseline restaurado |

## Ledger de comandos e entrypoints materiais

O A/B de driver, incluindo backups, hashes, instaladores e recuperação, está
registrado comando a comando em
[`EXECUTION_LOG.md`](../runs/optimization/GPU-DRIVER-AB-2026-08-24/EXECUTION_LOG.md).
Os comandos materiais de maior impacto foram:

```powershell
wsl -d Ubuntu-24.04 -u root -- systemctl stop llm-inference.service
wsl -d Ubuntu-24.04 -- bash /mnt/c/projects/tare.tools.local-labs/ops/gpu-stability/uv_bench.sh driver-591.86
610.88-desktop-win10-win11-64bit-international-nsd-dch-whql.exe -s -noreboot -clean
# O 610.47 foi executado com os mesmos argumentos e retornou 1.
pnputil /add-driver <run-dir>\rollback-591.86\nv_dispig.inf /install
```

Após o reboot, o benchmark de confirmação foi:

```powershell
wsl -d Ubuntu-24.04 -u root -- systemctl stop llm-inference.service
wsl -d Ubuntu-24.04 -- bash /mnt/c/projects/tare.tools.local-labs/ops/gpu-stability/uv_bench.sh post-reboot-591.86-operational-plus350
wsl -d Ubuntu-24.04 -u root -- systemctl start llm-inference.service
```

As janelas APEX4 e ADAPT também pararam somente
`llm-inference.service` por `systemctl`, mantiveram `llm-embedding.service`
residente e restauraram Fable ao final. Os entrypoints versionados executados
ou produzidos para reprodução foram:

- `ops/qwen38-bringup/activate_qwen38_vanilla_canary.sh` e os helpers já
  existentes de HauhauCS/Fable para alternância controlada dos braços;
- `tools/benchmarks/rwkv7_quality_eval.py` para o painel RWKV7;
- `tools/probes/adapt00_lora_smoke.py` para ADAPT-00A;
- `tools/probes/adapt00_geometry_matrix.py` para ADAPT-00B;
- `tools/probes/adapt00_behavioral_panel.py` para ADAPT-00C;
- `tools/analysis/kv_qualification_metrics.py` e
  `tests/test_kv_qualification_metrics.py` para o gate BEE-L2.

No APEX4, foram executados o build oficial da extensão com
`CUDA_HOME=/usr/local/cuda-12.4` e `TORCH_CUDA_ARCH_LIST=8.6`, o teste auxiliar
`test_very_few_stages` e a matriz documentada `test_groups`. Esta última passou
60 combinações com assertions numéricas. A tentativa de carregar cada shard
com `safetensors.safe_open` falhou com `MetadataIncompleteBuffer`, antes de
qualquer avaliação do modelo.

No BOFT, o build opcional do backend FBD CUDA foi tentado e falhou na compilação
contra os headers congelados. O fallback PyTorch previamente permitido foi
usado e passou. DoRA foi interrompido no primeiro loss não finito e não recebeu
resgate pós-resultado.

## Falhas, tentativas inválidas e recuperação

- Após a troca live para 610.88, um contexto CUDA antigo no WSL causou
  `nvidia-smi` com código 139 e uma chamada real de embedding falhou. O WSL foi
  reiniciado e as sondas reais voltaram; `/health` sozinho foi classificado
  como evidência insuficiente.
- O instalador 610.47 retornou 1 e deixou temporariamente a GPU com problem code
  28. O rollback exportado 591.86 foi aplicado com `pnputil`; não houve DDU nem
  retry destrutivo.
- Duas sondas PCIe pós-reboot foram inválidas: JSON UTF-8 malformado e prompt
  acima do contexto. A terceira, com 6.012 tokens, observou Gen4 x16.
- O primeiro smoke de restauração após APEX4 usou thinking padrão, atingiu o
  limite e não produziu o canário esperado. A repetição com
  `enable_thinking=false` retornou exatamente `apex4-baseline-restored-ok`.
- O checkpoint APEX4 pinado tinha os mesmos tamanhos publicados, mas offsets
  internos além do payload. Retry de download não poderia reparar a origem.
- O teste APEX4 `test_very_few_stages` solicitou uma combinação não implementada
  pelo dispatcher liberado; isso não invalida o `test_groups` oficial.
- Em 2026-08-25, uma chamada read-only combinada ao WSL retornou
  `Wsl/Service/E_UNEXPECTED` em uma subconsulta. A listagem systemd subsequente
  confirmou inferência, embeddings e quatro runners ativos. Não houve mutação.
- Duas primeiras consultas de inventário em 2026-08-25 tiveram quoting incorreto
  entre PowerShell e bash. Uma delas começou `du` sobre o diretório errado e foi
  interrompida com Ctrl-C após duas leituras de 8,3 GiB. Eram consultas
  read-only; foram repetidas com paths explícitos.

## Mudanças persistentes no host e no WSL

### Windows

- Driver NVIDIA 591.86 ativo; 610.88 e 610.47 não ficaram instalados.
- Fan Control permanece em `C:\FanControl\FanControl.exe`; a tarefa
  `FanControl` existe e o processo estava ativo na verificação de 2026-08-25.
- MSI Afterburner permanece em
  `C:\Program Files (x86)\MSI Afterburner\MSIAfterburner.exe`; tarefa e processo
  estavam ativos.
- `WSL-KeepAlive` estava `Running`.
- Backups e payloads locais do A/B permanecem excluídos do Git conforme o
  `.gitignore` do diretório da execução.

### WSL Ubuntu-24.04

O APEX4 exigiu CUDA 12.4 lado a lado; `/usr/local/cuda-13.3` não foi removido
nem substituído. Pacotes persistentes verificados:

| Pacote | Versão |
|---|---|
| `cuda-nvcc-12-4` | `12.4.131-1` |
| `cuda-libraries-dev-12-4` | `12.4.1-1` |
| `cuda-cudart-12-4` / `cuda-cudart-dev-12-4` | `12.4.127-1` |
| `cuda-driver-dev-12-4` | `12.4.127-1` |
| `libcublas-12-4` / `libcublas-dev-12-4` | `12.4.5.8-1` |
| `libcufft-12-4` / `libcufft-dev-12-4` | `11.2.1.3-1` |
| `libcurand-12-4` / `libcurand-dev-12-4` | `10.3.5.147-1` |
| `libcusolver-12-4` / `libcusolver-dev-12-4` | `11.6.1.9-1` |
| `libcusparse-12-4` / `libcusparse-dev-12-4` | `12.3.1.170-1` |
| `libnpp-12-4` / `libnpp-dev-12-4` | `12.2.5.30-1` |
| `libnvjitlink-12-4` / `libnvjitlink-dev-12-4` | `12.4.127-1` |

Artefatos locais retidos, não versionados:

| Path | Tamanho observado | Finalidade |
|---|---:|---|
| `/home/augus/.venvs/apex4-20260824` | 5,7 GiB | ambiente APEX4 pinado |
| `/home/augus/.venvs/adapt00-20260824` | 5,2 GiB | ambiente ADAPT-00 pinado |
| `/home/augus/models/apex4` | 2,1 GiB | checkpoint APEX4 inválido preservado como evidência |
| `/home/augus/models/adapt00` | 1,7 GiB | Qwen3.5-0.8B Base pinado |

Os adapters sob `runs/research/ADAPT-00*/raw/**/adapter/` somam 209.883.775
bytes (aproximadamente 200,2 MiB) e são ignorados pelo Git. Os JSONs de métricas
e resultados necessários para auditoria são versionados.

### Scratch fora do repositório

- `C:\projects\.codex-tmp\beellama-source-20260824`: clone read-only,
  aproximadamente 161,95 MiB.
- `C:\projects\.codex-tmp\apex4-w4a4-20260824`: clone de execução,
  aproximadamente 44,70 MiB.
- `adapt00-config-preflight.py` (1.227 bytes), `adapt00-download.sh` (709 bytes)
  e `verify-adapt00-restore.sh` (710 bytes): conveniência operacional temporária.

Esses arquivos não são canônicos e não foram commitados. Nenhum foi apagado
neste fechamento.

## Ações explicitamente não realizadas

- Nenhuma mudança em curvas de ventoinha, configuração `backupzin.json`, V/F,
  power limit ou offsets do Afterburner.
- Nenhum reboot adicional após o reboot solicitado pelo usuário.
- Nenhuma alteração ou push em `C:\projects\slop.cpp`.
- Nenhuma instalação, clone, build ou execução do FastMTP.
- Nenhum port APEX4, formato permanente de KV ou mudança de default.
- Nenhum download dos três checkpoints APEX4 restantes após o primeiro gate.
- Nenhum runtime/serviço RWKV novo.
- Nenhum ADAPT-01 e nenhuma promoção dos adapters 0.8B para Fable/27B.
- Nenhuma reliability soak nova; a fila autorizada excluía soaks.
- Nenhuma remoção dos ambientes, modelos, clones ou adapters locais listados.

## Estado vivo verificado em 2026-08-25

- Git: `master` e `origin/master` estavam em
  `83f4284539b398e697b7bc92eec43b1cc27189ac` antes deste fechamento.
- GPU: driver 591.86, 24.576 MiB totais, 20.890 MiB usados, 3.433 MiB livres,
  28 °C e 37,81 W no snapshot.
- `FanControl.exe` e `MSIAfterburner.exe`: ativos.
- Tarefas `FanControl`, `MSIAfterburner` e `WSL-KeepAlive`: presentes; as duas
  últimas estavam `Running`, FanControl estava `Ready` com processo ativo.
- `llm-inference.service` e `llm-embedding.service`: ativos.
- Quatro units `actions.runner.*`: ativas/running.

## Git, validação e CI já concluídos

| Commit | Conteúdo | CI |
|---|---|---|
| `3530dda093712f79291aa475482b276525f8f679` | pós-reboot, driver, Qwen3.8, FastMTP e RWKV7 | [`local-labs-ci` success](https://github.com/augusto-scarvalho/tare.tools.local-labs/actions/runs/32792075261) |
| `83f4284539b398e697b7bc92eec43b1cc27189ac` | transcript, BeeLlama, APEX4, ADAPT e BEE-L2 | [`local-labs-ci` success](https://github.com/augusto-scarvalho/tare.tools.local-labs/actions/runs/32798285257) |

Antes do segundo push, passaram 25 testes `pytest`, 23/23 checks do benchmark
harness, `compileall` e os selfchecks determinísticos aplicáveis. O commit deste
fechamento e o workflow correspondente são reportados no handoff final da
execução; o histórico GitHub é a autoridade para seu estado terminal.

## Pendências reais

| Prioridade | Item | Estado | Único gatilho de reabertura |
|---:|---|---|---|
| 1 | `ADAPT-01A-TRACE-DISTILLATION` | `BLOCKED_BEHAVIORAL` | nova hipótese preregistrada de budget/escala que produza finalista ADAPT-00C |
| 2 | Persistência MTP | `BLOCKED_MECHANISM` | hipótese falsificável de lifecycle com controles invariantes |
| 3 | ThinkingCap Qwen3.8 | `BLOCKED_UPSTREAM` | pesos oficiais e artefato compatível com RTX 3090 |
| 4 | Identidade MTP ThinkingCap legada | `BLOCKED_IDENTITY` | recibo do digest local exato |
| 5 | Builds de quantizadores terceiros | `UNKNOWN_BUILD` | recibos exatos do publisher |
| 6 | Calibração por juiz humano | `BLOCKED_HUMAN_INPUT` | 50–100 labels cegos congelados |
| 7 | RetNet | `BLOCKED_UPSTREAM` | checkpoint oficial Microsoft/TorchScale |
| — | BEE-L2 execução | `BLOCKED_CANDIDATE` | codec físico com formato imutável e route receipts |
| — | APEX4 | `BLOCKED_PUBLISHED_CHECKPOINT` | shards corrigidos e pacote end-to-end reproduzível |

Não há ação sequencial segura disponível sem um desses gatilhos. Limpeza de
ambientes/modelos/scratch é opcional e deve ser uma decisão separada, pois os
itens preservam reprodutibilidade e evidência de falha.

## Índice de recibos canônicos

- Driver: [`RESULT.md`](../runs/optimization/GPU-DRIVER-AB-2026-08-24/RESULT.md),
  [`EXECUTION_LOG.md`](../runs/optimization/GPU-DRIVER-AB-2026-08-24/EXECUTION_LOG.md) e
  [`POST_REBOOT_2026-08-24.md`](../runs/optimization/GPU-DRIVER-AB-2026-08-24/POST_REBOOT_2026-08-24.md).
- Qwen3.8/FastMTP/RWKV7: diretórios `runs/requalification/*-2026-08-24`.
- Transcript: [`research/BEELLAMA_SLOP_PEFT_ANALYSIS_2026-08-24.md`](research/BEELLAMA_SLOP_PEFT_ANALYSIS_2026-08-24.md).
- BeeLlama/APEX4/ADAPT/BEE-L2: diretórios `runs/research/*-2026-08-24`.
- Backlog: [`research/REMAINING_EXPERIMENTS_2026-08-24.md`](research/REMAINING_EXPERIMENTS_2026-08-24.md).
