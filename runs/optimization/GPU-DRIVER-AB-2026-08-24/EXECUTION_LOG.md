# Registro de execução — A/B de drivers NVIDIA — 2026-08-24

Este arquivo registra o que foi feito durante a qualificação dos drivers NVIDIA
no host Windows/WSL com RTX 3090. A conclusão resumida está em [`RESULT.md`](RESULT.md)
e o procedimento de continuação após o reboot está em
[`docs/HANDOFF_2026-08-24_GPU_DRIVER_AB_REBOOT.md`](../../../docs/HANDOFF_2026-08-24_GPU_DRIVER_AB_REBOOT.md).

## 1. Escopo e exclusões

Escopo autorizado:

- medir uma linha de base no driver 591.86;
- instalar e testar drivers NVIDIA Studio mais novos;
- preservar rollback, perfil do Afterburner, hashes e evidências;
- restaurar o baseline operacional do WSL.

Fora do escopo e não alterado:

- BIOS, microcode, Intel Default Settings, XMP ou frequência DDR5;
- Vcore/undervolt do CPU;
- teste de estabilidade de memória;
- configuração de 20/24 vCPUs, `autoMemoryReclaim` ou `swappiness`;
- HAGS, ASPM, ReBAR, plano de energia ou pagefile;
- versão do MSI Afterburner;
- código, modelos ou configuração permanente dos serviços de inferência.

Não foi usado DDU. Não houve alteração de BIOS, firmware ou driver Intel.

## 2. Estado inicial observado

- GPU: NVIDIA GeForce RTX 3090, PCI `00000000:01:00.0`.
- Driver: 591.86, CUDA reportada pelo host: 13.1.
- Power limit: 420 W.
- WSL: Ubuntu-24.04, kernel `6.6.114.1-microsoft-standard-WSL2`.
- `llm-inference.service`: ativo em 8080.
- `llm-embedding.service`: ativo em 8081.
- Quatro serviços de GitHub Actions runner ativos e nenhum `Runner.Worker` em execução.
- Afterburner em execução com o perfil por GPU contendo power limit 100%,
  `CoreClkBoost=-190000`, `MemClkBoost=0`, curva V/F personalizada e fan control.
- Histórico anterior à rodada: sete eventos `nvlddmkm` nos 14 dias anteriores;
  o mais recente era de 2026-08-23 00:40:21. Nenhum evento novo surgiu na rodada.

## 3. Preservação e downloads

Antes da primeira troca:

```powershell
pnputil /export-driver oem25.inf <run-dir>\rollback-591.86
```

O pacote exportado contém o driver-store completo do 591.86 e foi usado na
recuperação do dispositivo. Foram baixados os instaladores oficiais:

| Pacote | Tamanho | SHA-256 | Assinatura |
|---|---:|---|---|
| Game Ready 591.86 | 918,362,520 bytes | `A50C89C9D254F33CC8A8E638F7CC1981A76263005FCEC102AD8C8B45626D53E0` | NVIDIA válida |
| Studio 610.47 | 978,481,008 bytes | `59AC4A1659664AAD0A6FC525E5DF99B3FA76887BDE663F9E36E0E7EBB5DBA937` | NVIDIA válida |
| Studio 610.88 | 979,580,584 bytes | `6E6E7AEB03FA8788F0E97BF0D2F66852178AA05B7C17FB4A061E1BC1CF07EA0C` | NVIDIA válida |

### Correção do backup do Afterburner

O primeiro comando tentou copiar `Profiles\*` usando `Copy-Item -LiteralPath`.
Como `LiteralPath` não expande wildcard, esse comando não copiou o perfil por
GPU. A falha foi descoberta durante a recuperação e registrada, sem declarar o
backup incompleto como válido.

Antes do reboot, o perfil limpo foi copiado novamente com o caminho exato:

| Arquivo | SHA-256 |
|---|---|
| `afterburner-backup/VEN_10DE&DEV_2204&SUBSYS_39873842&REV_A1&BUS_1&DEV_0&FN_0.cfg` | `D9A872A964775B31FE01BA77B1A7A11BF206A543B1E7DA3EC98A433247DAB6C2` |
| `afterburner-backup/MSIAfterburner-profiles.cfg` | `7F8A7A975DE315A471D940C53F0DEA6C978C133AC09DBD805AD66FBAB992A417` |

O arquivo por GPU preservado contém a curva V/F existente, power limit 100%,
core offset `-190000`, memory offset `0`, `FanMode=1` e `FanSpeed=48`.

## 4. Linha de base 591.86

O serviço 8080 foi parado pelo unit systemd, nunca matando o filho com
`Restart=always`:

```powershell
wsl -d Ubuntu-24.04 -u root -- systemctl stop llm-inference.service
```

O embedding em 8081 permaneceu residente. O benchmark executado foi:

```powershell
wsl -d Ubuntu-24.04 -- bash /mnt/c/projects/tare.tools.local-labs/ops/gpu-stability/uv_bench.sh driver-591.86
```

Identidade congelada:

- `llama-bench` build `068764d92` (`b10159`), SHA-256
  `12beac933d456cae61f046a0de4ea9a1d3e245d46eb312ac8fd8046afc27fdb7`;
- modelo `gpt-oss-20b-Q4_K_M.gguf`, SHA-256
  `c27536640e410032865dc68781d80a08b98f8db5e93575919af8ccc0568aeb4f`;
- argv: `-ngl 99 -fa on -p 2048 -n 512 -r 3`.

Resultados:

| Run | pp2048 | tg512 | Potência | Temperatura | SM |
|---|---:|---:|---:|---:|---:|
| a | 5787.83 ± 7.70 tok/s | 197.94 ± 0.37 tok/s | 306.93 W | 45 C | 1830 MHz |
| b | 4058.23 ± 3065.23 tok/s | 199.25 ± 0.71 tok/s | 308.02 W | 48 C | 1830 MHz |
| c | 5740.01 ± 86.02 tok/s | 199.98 ± 0.47 tok/s | 315.45 W | 50 C | 1815 MHz |

O prefill do run b é inválido por variância extrema. A média dos dois prefills
estáveis é 5763.92 tok/s; a mediana de decode é 199.25 tok/s.

## 5. Instalação e qualificação do 610.88

O Afterburner foi fechado e o instalador executado com:

```powershell
610.88-desktop-win10-win11-64bit-international-nsd-dch-whql.exe -s -noreboot -clean
```

O instalador encerrou com código 0. Windows passou a reportar 610.88 e o WSL
também o reconheceu depois de uma reinicialização do WSL.

### Falha do contexto CUDA antigo

Imediatamente após a troca ao vivo, `nvidia-smi` dentro do WSL saiu com código
139. O `/health` do embedding ainda retornava HTTP 200, mas uma requisição real
de embedding derrubou o processo com:

```text
CUDA error: unknown error
cudaStreamSynchronize(cuda_ctx->stream())
```

Foi confirmado que não havia `Runner.Worker` nem benchmark/treino ativo. O WSL
foi reiniciado e então `nvidia-smi`, 8080, 8081 e os runners voltaram. Essa
ocorrência prova que `/health` sozinho não qualifica CUDA depois de live swap.

### Benchmark 610.88

| Run | pp2048 | tg512 | Potência | Temperatura | SM |
|---|---:|---:|---:|---:|---:|
| a | 5575.28 ± 41.61 tok/s | 189.62 ± 0.68 tok/s | 331.90 W | 58 C | 1845 MHz |
| b | 5502.34 ± 50.48 tok/s | 184.37 ± 10.99 tok/s | 336.93 W | 63 C | 1845 MHz |
| c | 5561.88 ± 48.29 tok/s | 190.12 ± 1.30 tok/s | 346.34 W | 66 C | 1845 MHz |

Resultado bruto contra o baseline válido:

- prefill: 5546.50 versus 5763.92 tok/s, regressão de 3.77%;
- mediana de decode: 189.62 versus 199.25 tok/s, regressão de 4.83%.

### Incompatibilidade prática com o Afterburner

Após a instalação limpa, o Afterburner 4.6.5 abriu, mas não reaplicou fan/curva.
A fan ficou em 0% a aproximadamente 50 C. Fechar e relançar, usar `/s` e carregar
explicitamente um slot temporário copiado de `Startup` não aplicaram o fan fixo.

Por isso potência e temperatura não formam um A/B controlado apenas por driver.
Mesmo assim, o candidato foi reprovado operacionalmente: throughput menor e
perda do caminho atual de controle do Afterburner.

## 6. Tentativa do 610.47 e recuperação

O 610.47 era o candidato citado originalmente por declarar otimizações para
llama.cpp. Seu instalador foi executado com os mesmos argumentos limpos, mas
encerrou com código 1 durante o downgrade. A transição removeu o driver ativo e
deixou temporariamente a RTX 3090 com:

```text
CM_PROB_FAILED_INSTALL / device problem code 28
```

Não foi feito retry destrutivo nem DDU. O pacote exportado antes da rodada foi
reaplicado diretamente:

```powershell
pnputil /add-driver <run-dir>\rollback-591.86\nv_dispig.inf /install
```

O comando instalou o pacote 591.86 no dispositivo e restaurou
`CM_PROB_NONE`. O WSL foi reiniciado novamente para descartar qualquer contexto
CUDA intermediário.

## 7. Limpeza e restauração

- O slot temporário `Profile4` foi removido do arquivo por GPU.
- O pequeno stub criado para o slot foi movido, não apagado, para
  `Profile4-temporary-removed.cfg` como evidência recuperável.
- `Startup` foi deixado com os valores existentes: PL 100%, core `-190000`,
  memória `0`, curva V/F preservada, fan mode 1 e fan speed 48.
- O Afterburner foi deixado fechado; a tarefa `MSIAfterburner` está habilitada e
  executará `MSIAfterburner.exe /s` no próximo logon.
- `llm-inference.service`, `llm-embedding.service` e
  `llm-locale-proxy.service` foram restaurados e passaram sondas reais.
- Os quatro runners foram restaurados com `NRestarts=0` após o último WSL boot.
- O gateway `local_agent_fleet_gateway.py` foi relançado.
- `WSL-KeepAlive` foi iniciado e verificado como `Running`.
- Não houve novo evento `nvlddmkm` durante a rodada.

## 8. Estado imediatamente antes do reboot

Verificado em `2026-08-24 18:59:51 -03:00`:

- repositório: `ed1fb68b2b7c73ffa311b14e0c7fceb32a7a62f5`;
- alteração do trabalho: somente o diretório de execução e a documentação ainda
  não commitados;
- Windows: RTX 3090, driver 591.86, `CM_PROB_NONE`, power limit 420 W;
- WSL: driver 591.86, kernel `6.6.114.1-microsoft-standard-WSL2`;
- 8080/8081/8082: ativos;
- runners: quatro ativos;
- gateway: PID 603;
- `WSL-KeepAlive`: `Running`;
- `MSIAfterburner`: tarefa `Ready`, processo propositalmente ausente;
- fan: 0% em idle sob controle automático stock, enquanto aguarda o reboot;
- eventos: sete `nvlddmkm` históricos, último em 2026-08-23 00:40:21; um WHEA
  informativo ID 3 no dia, sem WHEA de erro registrado nesta rodada.

## 9. Decisão

- manter 591.86;
- reprovar 610.88 com o Afterburner 4.6.5 atual;
- classificar 610.47 como `INSTALL_FAILED / NOT_BENCHMARKED`;
- reiniciar o Windows antes de qualquer nova carga de qualificação;
- após o logon, verificar primeiro a aplicação de fan/curva e somente então
  executar uma confirmação curta no 591.86.

## 10. Continuação pós-reboot e encerramento

Após o reboot real, o driver 591.86 voltou com a RTX 3090 em `CM_PROB_NONE`,
power limit de 420 W e CUDA funcional no WSL. A investigação corrigiu uma
atribuição do registro pré-reboot: Fan Control V272 (`backupzin.json`) controla
as ventoinhas; o MSI Afterburner controla a curva V/F e os offsets de clock.

O perfil operacional do usuário havia retornado a `MemClkBoost=350000`. Ele foi
preservado, sem alterar fan, V/F, power limit ou clocks. Por isso a confirmação
foi rotulada `post-reboot-591.86-operational-plus350` e não é uma réplica do
baseline temporário com memória em zero.

Resultado válido:

- prefill: `6273.75 ± 98.65 tok/s`;
- decode: `221.96 ± 1.74 tok/s`;
- pico de potência: `323.28 W`;
- pico térmico: `44 C`;
- pico SM: `1830 MHz`;
- `BENCH_RC=0`.

Uma requisição válida de 6.012 tokens observou PCIe Gen4 x16 sob carga. Duas
tentativas anteriores foram preservadas como inválidas: JSON com UTF-8 malformado
e prompt de 19.212 tokens recusado antes da inferência. BAR1 reportou 32 GiB.

O 8080 foi parado e restaurado apenas via `systemctl`; o embedding 8081 ficou
residente. O proxy 8082 parou com a janela de manutenção do 8080 e foi
explicitamente reiniciado. Ao final, chamadas reais retornaram `restored-ok`,
um embedding de 768 dimensões e `proxy-restored-ok`. Quatro runners, gateway e
keepalive estavam ativos, sem novo `nvlddmkm` ou WHEA de erro.

Recibo detalhado: [`POST_REBOOT_2026-08-24.md`](POST_REBOOT_2026-08-24.md).

Decisão final: manter 591.86; 610.88 continua rejeitado; 610.47 continua
`INSTALL_FAILED / NOT_BENCHMARKED`; qualificação operacional pós-reboot concluída.
