# Handoff para o próximo agente — 2026-08-25

Este é o ponto de entrada operacional após o fechamento das rodadas de driver,
Qwen3.8, RWKV7, BeeLlama, APEX4 e PEFT, complementado com a consolidação exaustiva
do transcript de pesquisa em [`research/MASTER_RESEARCH_BACKLOG_2026.md`](research/MASTER_RESEARCH_BACKLOG_2026.md).
Não reconstrua o estado pelo histórico do chat. Leia primeiro este arquivo, depois o
[`research/MASTER_RESEARCH_BACKLOG_2026.md`](research/MASTER_RESEARCH_BACKLOG_2026.md) e o
[`EXECUTION_CLOSEOUT_2026-08-24_25.md`](EXECUTION_CLOSEOUT_2026-08-24_25.md).
Consulte os recibos individuais somente para o item efetivamente retomado.

Snapshot live deste handoff: **2026-08-25 01:28:00 -03:00**.

## 0. Resumo para entrada imediata

- Repositório: `master`, `HEAD` e `origin/master` em
  `8bb0197d4a280aafb20e118db8ff5a7fc21d0631`; worktree limpo antes da criação
  deste handoff.
- CI do fechamento: [`local-labs-ci` success](https://github.com/augusto-scarvalho/tare.tools.local-labs/actions/runs/32805077537).
- GPU: RTX 3090, driver 591.86, 24.576 MiB totais, 20.888 MiB usados,
  3.435 MiB livres, 28 °C e 35,61 W no snapshot.
- 8080: `llm-inference.service` ativo; `/health` HTTP 200; `/props.model_path`
  aponta para `/home/augus/models/merges/fable-tc-l1.0-Q4_K_M.gguf`;
  `total_slots=4`.
- 8081: `llm-embedding.service` ativo; `/health` HTTP 200.
- 8082: **divergência aberta**. `llm-locale-proxy.service` está enabled, mas
  `inactive (dead)`; `/health` não conecta. O journal mostra parada limpa por
  `SIGTERM` em 2026-08-24 21:30:35 -03, sem crash registrado.
- Quatro units GitHub Actions runner estão ativas/running.
- `FanControl.exe` e `MSIAfterburner.exe` estão ativos.
- Tarefas `MSIAfterburner` e `WSL-KeepAlive`: `Running`; tarefa `FanControl`:
  `Ready`, com processo Fan Control ativo.
- Não existe experimento não-soak pronto para continuar automaticamente.

Este handoff e a atualização do ponteiro `docs/HANDOFF.md` são as mudanças
esperadas no worktree após o snapshot acima. Verifique `git status` antes de
qualquer edição e preserve mudanças adicionais do usuário.

## 1. Primeira checagem do próximo agente

Execute somente leitura antes de qualquer mutação:

```powershell
Set-Location C:\projects\tare.tools.local-labs
git status --short
git log -3 --oneline --decorate
git rev-parse HEAD
git rev-parse origin/master

nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,power.draw --format=csv,noheader
Get-Process -Name FanControl,MSIAfterburner -ErrorAction SilentlyContinue |
  Select-Object ProcessName,Id,StartTime,Path
Get-ScheduledTask -TaskName FanControl,MSIAfterburner,WSL-KeepAlive -ErrorAction SilentlyContinue |
  Select-Object TaskName,State

wsl -d Ubuntu-24.04 -- systemctl is-active llm-inference.service llm-embedding.service llm-locale-proxy.service
wsl -d Ubuntu-24.04 -- systemctl list-units --type=service --state=running --no-legend --no-pager
```

Depois confira os endpoints, sem usar apenas `/health` para qualificar CUDA
após qualquer futura troca de driver:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health -TimeoutSec 5
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8081/health -TimeoutSec 5
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8082/health -TimeoutSec 5
```

## 2. Primeira pendência operacional: proxy 8082

Não declare os três endpoints restaurados. No snapshot deste handoff, somente
8080 e 8081 estavam disponíveis. O proxy 8082 foi parado de forma limpa durante
uma janela anterior e não voltou.

Diagnóstico já executado:

```powershell
wsl -d Ubuntu-24.04 -- systemctl is-enabled llm-locale-proxy.service
wsl -d Ubuntu-24.04 -- systemctl --no-pager --full status llm-locale-proxy.service
wsl -d Ubuntu-24.04 -- journalctl -u llm-locale-proxy.service -n 30 --no-pager
```

Resultado: `enabled`, `inactive (dead)`, último processo encerrado por `TERM` e
unit `Deactivated successfully`. Não há evidência suficiente para atribuir a
parada a crash. Nenhum restart foi feito durante a criação deste handoff.

Se uma solicitação posterior autorizar restaurar o baseline de três endpoints,
use o unit, nunca um processo Python manual:

```powershell
wsl -d Ubuntu-24.04 -u root -- systemctl start llm-locale-proxy.service
wsl -d Ubuntu-24.04 -- systemctl is-active llm-locale-proxy.service
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8082/health -TimeoutSec 5
```

Depois faça uma chamada real via 8082 e confirme o contrato PT-BR; health 200
sozinho não fecha a restauração. Registre o resultado antes de mudar qualquer
documento que ainda diga que 8082 está ativo.

## 3. Invariantes de operação

1. Fan Control V272 com `backupzin.json` é o proprietário das ventoinhas.
   MSI Afterburner 4.6.5 controla apenas V/F e offsets de clock.
2. Não altere fan curves, V/F, power limit, clocks, BIOS, XMP, DDR5 ou Vcore
   como efeito colateral de uma campanha de modelo.
3. Pare ou reinicie `llm-inference.service` somente por `systemctl`; nunca mate
   o filho `llama-server` de um unit com `Restart=always`.
4. Não use `pkill -f llama-server` no host inteiro.
5. Preserve `llm-embedding.service` e a porta 8081 durante janelas de texto.
6. Use porta experimental explícita quando possível e confira
   `/props.model_path` antes de coletar evidência.
7. Para restaurar Fable, use
   `ops/qwen38-bringup/restore_fable_service.sh` e valide chamadas reais.
8. Preserve recibos inválidos e resultados negativos com seus rótulos; não
   apague, não sobrescreva e não os misture com resultados válidos.
9. Não trate diretórios locais de modelo/venv como mudanças promovidas.
10. Não faça push remoto sem autorização explícita da solicitação corrente.
11. **Rastreabilidade Obrigatória de Agente**: Todo item de backlog executado, testado ou concluído deve registrar explicitamente o agente responsável pela execução (ex: `Codex`, `Antigravity`, `Sonnet`, etc.) nos recibos em `runs/` e nas tabelas canônicas.

## 4. O que está concluído

| Frente | Estado final | Agente Executor | Recibo principal |
|---|---|:---:|---|
| Driver A/B e pós-reboot | 591.86 mantido; 610.88 rejeitado; 610.47 install failed | **Codex** | [`GPU-DRIVER-AB`](../runs/optimization/GPU-DRIVER-AB-2026-08-24/POST_REBOOT_2026-08-24.md) |
| Ownership Fan Control/Afterburner | corrigido e verificado | **Codex** | [`POST_REBOOT`](../runs/optimization/GPU-DRIVER-AB-2026-08-24/POST_REBOOT_2026-08-24.md) |
| Agent/tool Fable/HauhauCS/vanilla | 7/8, 8/8, 8/8; gate maior não abriu | **Codex** | [`QWEN38-AGENT`](../runs/requalification/QWEN38-AGENT-REGRESSION-2026-08-24/RESULT.md) |
| GSM8K-200 | Fable 195/200; HauhauCS 191/200 com 8 truncations | **Codex** | [`QWEN38-MATH`](../runs/requalification/QWEN38-HAUHAUCS-MATH-2026-08-24/RESULT.md) |
| FastMTP | `NO-GO BEFORE INSTALL` | **Codex** | [`DECISION.md`](../runs/requalification/QWEN38-HAUHAUCS-FASTMTP-2026-08-24/DECISION.md) |
| RWKV7 | licença liberada; `HOLD_QUALITY` 13/48 | **Codex** | [`RWKV7`](../runs/requalification/RWKV7-SERVING-QUALITY-2026-08-24/RESULT.md) |
| Transcript | reconciliado e transformado em fila dependency-gated | **Codex / Antigravity** | [`Analysis`](research/BEELLAMA_SLOP_PEFT_ANALYSIS_2026-08-24.md) / [`Master Backlog`](research/MASTER_RESEARCH_BACKLOG_2026.md) |
| BeeLlama | arqueologia concluída; whole-fork import rejeitado | **Codex** | [`BEE-L0`](../runs/research/BEE-L0-SOURCE-ARCHAEOLOGY-2026-08-24/RESULT.md) |
| slop.cpp lifecycle receipts | `GAP_CONFIRMED`; nenhum código alterado | **Codex** | [`SLX-01A`](../runs/research/SLX-01A-GAP-AUDIT-2026-08-24/RESULT.md) |
| APEX4 | kernels passam; checkpoint publicado inválido; sem port | **Codex** | [`SLX-02`](../runs/research/SLX-02-APEX4-2026-08-24/RESULT.md) |
| ADAPT-00A/B/C | mechanics e matriz concluídos; nenhum arm promovido | **Codex** | [`ADAPT-00C`](../runs/research/ADAPT-00C-BEHAVIORAL-2026-08-24/RESULT.md) |
| BEE-L2 | scorer e design completos; execução sem candidato | **Codex** | [`BEE-L2`](../runs/research/BEE-L2-KV-QUALIFICATION-DESIGN-2026-08-24/RESULT.md) |
| Consolidação Backlog Mestre (46 itens) | Concluída exaustivamente com papers, excertos e ROI | **Antigravity** | [`MASTER_RESEARCH_BACKLOG_2026.md`](research/MASTER_RESEARCH_BACKLOG_2026.md) |

Commits publicados:

- `3530dda093712f79291aa475482b276525f8f679` — fechamento pós-reboot e backlog
  Qwen3.8/RWKV7;
- `83f4284539b398e697b7bc92eec43b1cc27189ac` — transcript, APEX4, ADAPT e
  BEE-L2;
- `8bb0197d4a280aafb20e118db8ff5a7fc21d0631` — closeout consolidado e
  changelog.

Todos os três workflows associados terminaram em `success`.

## 5. Backlog consolidado e ordem de execução (Custo / ROI na RTX 3090)

O catálogo exaustivo com todos os 46 experimentos, excertos, papers acadêmicos e status está formalizado em:
👉 [`research/MASTER_RESEARCH_BACKLOG_2026.md`](research/MASTER_RESEARCH_BACKLOG_2026.md).

A fila de execução priorizada por relação custo-benefício computacional na estação é:

| Rank | Código | Frente | Custo Estimado | ROI Potencial | Próximo Passo Imediato / Gatilho |
|:---:|---|---|:---:|:---:|---|
| **#1** | **`ADAPT-01`** | **Retomada LoKr Reasoning** | ~25 min | **Extremo (9.5/10)** | Treinar LoKr com 3 épocas no Qwen 0.8B / 1.5B (superar 16/32 no GSM8K). |
| **#2** | **`SLX-05`** | **Launch-Overhead Oracle (Lucebox)**| ~15 min | **Muito Alto (8.8/10)** | Nsight trace no Qwen-0.8B para medir teto de CPU launch bottleneck. |
| **#3** | **`REP-02`** | **Precision Tail Standard** | ~45 min | **Muito Alto (8.7/10)** | Protótipo de sinks (4 tokens) + cauda F16 (64 tokens) em contexto longo. |
| **#4** | **`BEE-L1`** | **Effective Route Receipts** | ~30 min | **Alto (8.3/10)** | Shadow parser Python para emitir receipts dos 4 níveis de config. |
| **#5** | **`ADAPT-04`**| **Prior-Preservation Loss (DreamBooth)**| ~35 min | **Alto (8.2/10)** | Injetar regularizador sintético no treino do LoKr para proteger QA geral. |
| **#6** | **`SLX-01B`**| **Stateful Serving Torture Matrix** | ~40 min | **Alto (8.0/10)** | Fuzzing assíncrono com 4 slots e cancelamentos forçados via `curl`. |
| **#7** | **`BEE-L3`** | **Adaptive MTP Profit Controller** | ~40 min | **Alto (7.9/10)** | Redução dinâmica de profundidade MTP sob taxa de rollback alta. |
| **#8** | **`SLX-09`** | **Sparsidade Estruturada 2:4 Ampere** | ~1h 00m | **Alto (7.8/10)** | Oracle de GEMM 2:4 com calibração imatrix na 3090 (`sm_86`). |

Fontes canônicas complementares:
- [`research/MASTER_RESEARCH_BACKLOG_2026.md`](research/MASTER_RESEARCH_BACKLOG_2026.md) (catálogo completo de 46 itens);
- [`research/REMAINING_EXPERIMENTS_2026-08-24.md`](research/REMAINING_EXPERIMENTS_2026-08-24.md) (registro de bloqueios);
- [`research/BEELLAMA_SLOP_PEFT_ANALYSIS_2026-08-24.md`](research/BEELLAMA_SLOP_PEFT_ANALYSIS_2026-08-24.md) (análise inicial).

## 6. Artefatos persistentes e scratchpads do Codex

A inspeção confirmou os seguintes artefatos que devem ser preservados para reprodução e evidência:

- `C:\projects\.codex-tmp\`:
  - `adapt00-config-preflight.py` (teste de configs PEFT);
  - `verify-adapt00-restore.sh` (canary do Fable 8080);
  - `adapt00-download.sh` (helper de download da base);
  - `beellama-source-20260824/` e `apex4-w4a4-20260824/` (clones git).
- `C:\projects\.codex-ops-backups\`: backups de units systemd, Afterburner e MSI do WSL.
- WSL `/home/augus/.venvs/`:
  - `apex4-20260824` (~5,7 GiB, PyTorch + CUDA 12.4 compilado);
  - `adapt00-20260824` (~5,2 GiB, Transformers + PEFT / LyCORIS).
- WSL `/home/augus/models/`:
  - `adapt00/qwen3.5-0.8b-base-dc7cdfe` (~1,7 GiB);
  - `apex4/qwen2.5-7b-g128` (~2,1 GiB, shards corrompidos preservados).
- `tare.tools.local-labs/runs/research/`:
  - `ADAPT-00A`, `ADAPT-00B` e `ADAPT-00C` contendo adapters `.safetensors` reais (`lokr`, `loha`, `boft`, `ia3`, `trainable_tokens`).

Não apague nada disso sem solicitação explícita; preservam reprodução e evidência.

## 7. Validação e critérios para declarar recuperação

O último fechamento passou localmente:

- `python -m compileall -q src tools tests benchmark_harness_qa.py`;
- `python -m pytest -q`: 25/25;
- `python tests/benchmark_harness/benchmark_harness_selftest.py`: 23/23;
- GitHub Actions `deterministic-qa`: success.

Ao tocar no baseline de serving, a recuperação só está completa quando:

1. units esperadas estão active/running;
2. `/props.model_path` confirma o artefato correto;
3. 8080 responde um canário real com `enable_thinking=false` quando o canário
   exige texto exato;
4. 8081 devolve um embedding real de 768 dimensões;
5. se 8082 estiver no escopo, uma chamada real confirma o contrato PT-BR;
6. Fan Control e Afterburner continuam ativos com seus papéis separados;
7. os quatro runners continuam ativos;
8. o resultado e qualquer desvio são documentados antes de commit/push.

## 8. Regra de continuidade

Se o usuário pedir apenas status, análise ou diagnóstico, permaneça read-only.
Se pedir implementação, execute somente a mudança em escopo e restaure o
baseline. Se pedir continuar experimentos, primeiro revalide o gatilho e a
existência de job real; recibos históricos não provam que algo está rodando.

Não há trabalho científico meio executado para “retomar”. A única divergência
operacional observada neste handoff é o proxy 8082 inativo; tratá-la depende da
próxima solicitação do usuário.
