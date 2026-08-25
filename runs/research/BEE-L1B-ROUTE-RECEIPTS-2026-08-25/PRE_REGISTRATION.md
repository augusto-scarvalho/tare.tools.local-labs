# BEE-L1B effective-route receipt — preregistration

Executor: Codex  
Predecessor: `BEE-L1` (`SUPERSEDED_GATE_DEFECT`)  
Target: canonical Fable-TC service at `http://127.0.0.1:8080`

## Frozen identity and command

- Repository HEAD before run: `8bb0197d4a280aafb20e118db8ff5a7fc21d0631` (dirty remediation worktree will be recorded).
- Verifier SHA-256: `dc4bf49e340725224c97466f95c6f96eb2f338dae5c9a67d512a1b2298e452b5`.
- Provenance helper SHA-256: `230ae7266707f4469f9add33ca8a4e0ed9148760e8a85d81598915cc931d9fb9`.
- Expected model SHA-256: `052c08ca13d75d8d88c9cc3f201d7bfa9167e2a1e69ad3e1e1f26ff73c1b390b`.
- Expected build family: `b10159`.
- Command:

```powershell
python tools/analysis/effective_route_verifier.py `
  --endpoint http://127.0.0.1:8080 `
  --expected-model fable-tc-l1.0-Q4_K_M.gguf `
  --systemd-unit llm-inference.service `
  --wsl-distro Ubuntu-24.04 `
  --hash-model `
  --expected-model-sha256 052c08ca13d75d8d88c9cc3f201d7bfa9167e2a1e69ad3e1e1f26ff73c1b390b `
  --output runs/research/BEE-L1B-ROUTE-RECEIPTS-2026-08-25/raw/receipt.json
```

## Gates

All must pass: systemd unit active/running; effective argv and live `/proc` argv captured; model path agrees across argv and `/props`; model content hash matches the frozen digest; `/props.build_info` agrees with the executable path; non-empty slot allocation agrees with `total_slots`; every slot exposes positive `n_ctx`; strict canary content equals `route-receipt-ok`; provenance envelope is complete.

Verdict is `VERIFIED` only if there are zero divergences. Any missing evidence is `UNVERIFIED` or `DIVERGENT`.
