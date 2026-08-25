# SLX-01C serving torture — preregistration

Executor: Codex  
Predecessor: `SLX-01B` (`SUPERSEDED_GATE_DEFECT`)  
Target: canonical Fable-TC service at `http://127.0.0.1:8080`

## Frozen identity and command

- Repository HEAD before run: `8bb0197d4a280aafb20e118db8ff5a7fc21d0631`.
- Probe SHA-256: `94062910711b23a4bd59d54ef2c72dea74ae89dbf1e263be100a03055dd026e9`.
- Provenance helper SHA-256: `230ae7266707f4469f9add33ca8a4e0ed9148760e8a85d81598915cc931d9fb9`.
- Command:

```powershell
python tools/probes/slx01b_serving_torture.py `
  --endpoint http://127.0.0.1:8080 `
  --expected-slots 4 `
  --max-vram-drift-mib 20 `
  --settle-seconds 10 `
  --systemd-unit llm-inference.service `
  --wsl-distro Ubuntu-24.04 `
  --output runs/research/SLX-01C-SERVING-TORTURE-2026-08-25/raw/receipt.json
```

## Gates

All must pass: 18/20 or better normal completions; 20/20 explicit clean aborts; mixed phase 10/10 normal completions and 10/10 explicit clean aborts; four slots before and after; every final slot explicitly reports `is_processing == false`; 5/5 exact canaries; unchanged systemd MainPID and restart counter; service active/running before and after; post-settle GPU-memory drift no greater than 20 MiB; complete provenance.

The run must not restart or stop the service and must leave embedding 8081 untouched.
