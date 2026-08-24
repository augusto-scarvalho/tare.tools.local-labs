# Handoff — reboot e qualificação pós-driver — 2026-08-24

Este é o handoff operacional para continuar imediatamente após o reboot do
Windows que encerra a rodada de A/B dos drivers NVIDIA. Não refaça a investigação
a partir do chat. Leia primeiro este arquivo e depois o
[`RESULT.md`](../runs/optimization/GPU-DRIVER-AB-2026-08-24/RESULT.md) e o
[`EXECUTION_LOG.md`](../runs/optimization/GPU-DRIVER-AB-2026-08-24/EXECUTION_LOG.md).

Última verificação pré-reboot: **2026-08-24 18:59:51 -03:00**.

## 0. Resultado e próxima ação

- 610.88 Studio: reprovado neste host com MSI Afterburner 4.6.5; apresentou
  regressão bruta de 3.77% em prefill e 4.83% na mediana de decode, e não
  reaplicou o controle salvo de fan/VF.
- 610.47 Studio: `INSTALL_FAILED / NOT_BENCHMARKED`; a tentativa de downgrade
  deixou temporariamente a RTX 3090 com código 28.
- 591.86 Game Ready: restaurado pelo pacote de driver-store exportado;
  Windows e WSL o reportam novamente e a placa está `CM_PROB_NONE`.
- Próxima ação obrigatória: reboot real do Windows, logon e validação do
  Afterburner antes de qualquer benchmark ou carga longa.

Não atualizar BIOS, XMP, DDR5, Vcore, WSL vCPU/memory-reclaim ou Afterburner
neste handoff. Esses itens são campanhas separadas.

## 1. Estado pré-reboot que deve ser recuperado

| Componente | Estado pré-reboot |
|---|---|
| GPU Windows | RTX 3090, 591.86, `CM_PROB_NONE`, PL 420 W |
| GPU WSL | 591.86, CUDA funcional |
| Kernel WSL | `6.6.114.1-microsoft-standard-WSL2` |
| 8080 | `llm-inference.service` ativo, Fable-TC |
| 8081 | `llm-embedding.service` ativo, Nomic Embed |
| 8082 | `llm-locale-proxy.service` ativo |
| Runners | quatro units ativas, `NRestarts=0` após último WSL boot |
| Gateway | `local_agent_fleet_gateway.py` ativo |
| Keepalive | `WSL-KeepAlive` em `Running` |
| Afterburner | processo fechado; tarefa no logon habilitada com `/s` |
| Eventos GPU | sete históricos em 14 dias; nenhum novo na rodada |

O processo do Afterburner foi deixado fechado de propósito. O controle stock de
fan é seguro em idle, mas não execute carga longa até confirmar que o perfil foi
reaplicado após o logon.

## 2. Backups e recuperação

Diretório da execução:

```text
C:\projects\tare.tools.local-labs\runs\optimization\GPU-DRIVER-AB-2026-08-24
```

Receipts principais:

- `rollback-591.86/`: driver-store exportado antes da primeira troca;
- instalador oficial `591.86-desktop-win10-win11-64bit-international-dch-whql.exe`;
- `afterburner-backup/VEN_10DE&DEV_2204&SUBSYS_39873842&REV_A1&BUS_1&DEV_0&FN_0.cfg`;
- `afterburner-backup/MSIAfterburner-profiles.cfg`;
- `RESULT.md` e `EXECUTION_LOG.md`.

Hashes do backup correto do Afterburner:

- perfil por GPU: `D9A872A964775B31FE01BA77B1A7A11BF206A543B1E7DA3EC98A433247DAB6C2`;
- configuração de profiles: `7F8A7A975DE315A471D940C53F0DEA6C978C133AC09DBD805AD66FBAB992A417`.

Não use `afterburner-profile-pre61047.cfg` como backup canônico; ele foi
capturado durante a investigação, depois da primeira falha de reaplicação.

## 3. Checklist pós-boot obrigatório

### 3.1 Windows, driver e dispositivo

```powershell
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
nvidia-smi --query-gpu=name,driver_version,pci.bus_id,pstate,temperature.gpu,fan.speed,power.draw,power.limit,memory.used,memory.total,clocks.sm,clocks.mem --format=csv,noheader
Get-PnpDevice -InstanceId 'PCI\VEN_10DE&DEV_2204&SUBSYS_39873842&REV_A1\4&2635B274&0&0008' |
  Select-Object Status,Class,FriendlyName,Problem,InstanceId
```

Critérios:

- driver exatamente 591.86;
- RTX 3090 em `CM_PROB_NONE`;
- nenhum fallback para Microsoft Basic Display Adapter;
- power limit 420 W.

### 3.2 Afterburner antes de carga

```powershell
Get-Process MSIAfterburner -ErrorAction SilentlyContinue |
  Select-Object Id,StartTime,Responding,Path
Get-ScheduledTask -TaskName MSIAfterburner |
  Select-Object TaskName,State
```

Validar no perfil por GPU:

- `PowerLimit=100`;
- `CoreClkBoost=-190000`;
- `MemClkBoost=0`;
- `VFCurve` presente e não vazia;
- `FanMode=1`;
- `FanSpeed=48`.

Depois do logon, a fan deve sair do estado stock 0% se o perfil for aplicado
como no baseline. Se o Afterburner estiver ativo mas a fan continuar 0%, não
rode benchmark. Compare o arquivo vivo com o backup canônico e aplique o perfil
pela interface. Se ainda falhar, mantenha a GPU stock, registre a falha e pare a
qualificação; não invente outra curva.

### 3.3 WSL, CUDA e serviços

```powershell
wsl -d Ubuntu-24.04 -- uname -r
wsl -d Ubuntu-24.04 -- nvidia-smi --query-gpu=name,driver_version,pstate,memory.used,memory.total --format=csv,noheader
wsl -d Ubuntu-24.04 -- systemctl show llm-inference.service llm-embedding.service llm-locale-proxy.service -p Id -p ActiveState -p SubState -p MainPID -p NRestarts
```

Esperado:

- kernel `6.6.114.1-microsoft-standard-WSL2`;
- driver 591.86 no WSL;
- três units `active/running`;
- endpoints 8080, 8081 e 8082 saudáveis.

Não aceite apenas `/health` para CUDA. Faça uma requisição real de embedding em
8081 e uma chat completion curta em 8080. Na troca anterior, `/health` ficou 200
mesmo com o contexto CUDA quebrado.

### 3.4 Runners, gateway e keepalive

```powershell
wsl -d Ubuntu-24.04 -- systemctl --no-pager --type=service --state=running
wsl -d Ubuntu-24.04 -- pgrep -a Runner.Worker
wsl -d Ubuntu-24.04 -- pgrep -a -f local_agent_fleet_gateway.py
Get-ScheduledTask -TaskName WSL-KeepAlive,WSL2-KeepAlive |
  Select-Object TaskName,State
```

Se o gateway não estiver presente, relançar exatamente:

```powershell
Start-Process -FilePath 'C:\Windows\System32\wsl.exe' `
  -ArgumentList '-d Ubuntu-24.04 --exec python3 /home/augus/.local/share/local-agent-fleet/local_agent_fleet_gateway.py --config /home/augus/.local/share/local-agent-fleet/local-agent-modelcards.json --state-dir /home/augus/.local/state/local-agent-fleet' `
  -WindowStyle Hidden
```

Se `WSL-KeepAlive` estiver apenas `Ready`, iniciar e verificar `Running`:

```powershell
Start-ScheduledTask -TaskName WSL-KeepAlive
```

### 3.5 PCIe/ReBAR e eventos

```powershell
nvidia-smi -q -d PCI
Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='nvlddmkm';StartTime=(Get-Date).Date} -ErrorAction SilentlyContinue
Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-WHEA-Logger';StartTime=(Get-Date).Date} -ErrorAction SilentlyContinue
```

Confirmar PCIe Gen4 x16 sob carga e BAR1/ReBAR de 32 GiB. Qualquer novo
`nvlddmkm` 153, Xid, WHEA corrigido/erro de barramento ou falha de driver deve ser
registrado com timestamp antes de continuar.

## 4. Confirmação curta do 591.86

Só executar depois que o perfil do Afterburner estiver comprovadamente aplicado.
Parar somente o serviço 8080 e deixar o embedding residente:

```powershell
wsl -d Ubuntu-24.04 -u root -- systemctl stop llm-inference.service
wsl -d Ubuntu-24.04 -- bash /mnt/c/projects/tare.tools.local-labs/ops/gpu-stability/uv_bench.sh post-reboot-591.86
wsl -d Ubuntu-24.04 -u root -- systemctl start llm-inference.service
```

Referência, não limiar inventado:

- prefill estável observado: 5740–5788 tok/s;
- decode observado: 198–200 tok/s;
- pico de potência: 307–315 W;
- pico térmico: 45–50 C.

Após o benchmark, repetir as sondas reais de 8080/8081, verificar 8082, runners,
gateway, keepalive e eventos. Atualizar `RESULT.md` com uma seção pós-reboot e
marcar a rodada concluída somente se todo o baseline estiver restaurado.

## 5. Guardrails

1. Parar `llm-inference.service` por `systemctl`; nunca matar seu processo filho.
2. Não usar `pkill -f llama-server`.
3. Preservar o embedding 8081 durante o benchmark curto.
4. Não repetir instalação do 610.47/610.88 nesta continuação.
5. Não usar DDU sem nova autorização e um plano explícito de recuperação.
6. Não alterar curva V/F, memory clock ou fan speed para “melhorar” o resultado.
7. Não misturar esta rodada com BIOS/microcode, RAM/XMP ou Vcore.
8. Não commitar os instaladores de aproximadamente 1 GB nem o pacote exportado
   de aproximadamente 2.8 GB. Esses arquivos são receipts locais, não artefatos
   de Git.

## 6. Estado do repositório

SHA pré-reboot: `ed1fb68b2b7c73ffa311b14e0c7fceb32a7a62f5`.

Os artefatos desta rodada estão não commitados. Não houve commit nem push. Antes
de qualquer commit, separar documentação pequena dos instaladores e do
driver-store exportado; confirmar a política de retenção/ignore com o usuário.

## 7. Fechamento pós-reboot

Este handoff foi executado e fechado em 2026-08-24. A qualificação operacional
do 591.86 passou. O recibo canônico da continuação é
[`POST_REBOOT_2026-08-24.md`](../runs/optimization/GPU-DRIVER-AB-2026-08-24/POST_REBOOT_2026-08-24.md).

Correção importante: Fan Control V272 é o proprietário das curvas de
ventoinha; MSI Afterburner mantém V/F e clocks. As instruções anteriores que
atribuem fan control ao Afterburner ficam superseded nesse ponto específico.
O perfil operacional `mem +350` foi preservado e a confirmação pós-reboot foi
registrada como tuple distinta do baseline temporário `mem=0`.
