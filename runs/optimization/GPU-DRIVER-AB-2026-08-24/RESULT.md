# GPU driver A/B — 2026-08-24

Procedimento e cronologia: [`EXECUTION_LOG.md`](EXECUTION_LOG.md). Continuação
pós-reboot: [`docs/HANDOFF_2026-08-24_GPU_DRIVER_AB_REBOOT.md`](../../../docs/HANDOFF_2026-08-24_GPU_DRIVER_AB_REBOOT.md).

## Scope

- Host: NVIDIA GeForce RTX 3090, Windows 11 + WSL2 Ubuntu-24.04.
- Baseline driver: GeForce Game Ready 591.86 WHQL.
- Candidates: NVIDIA Studio 610.88 WHQL and NVIDIA Studio 610.47 WHQL.
- Benchmark: `llama-bench` build `068764d92` (`b10159`), model `gpt-oss-20b-Q4_K_M.gguf`, `-ngl 99 -fa on -p 2048 -n 512 -r 3`.
- Binary SHA-256: `12beac933d456cae61f046a0de4ea9a1d3e245d46eb312ac8fd8046afc27fdb7`.
- Model SHA-256: `c27536640e410032865dc68781d80a08b98f8db5e93575919af8ccc0568aeb4f`.

## Raw benchmark results

| Driver / run | pp2048 tok/s | tg512 tok/s | Peak power | Peak temp | Peak SM |
|---|---:|---:|---:|---:|---:|
| 591.86 a | 5787.83 ± 7.70 | 197.94 ± 0.37 | 306.93 W | 45 C | 1830 MHz |
| 591.86 b | 4058.23 ± 3065.23 | 199.25 ± 0.71 | 308.02 W | 48 C | 1830 MHz |
| 591.86 c | 5740.01 ± 86.02 | 199.98 ± 0.47 | 315.45 W | 50 C | 1815 MHz |
| 610.88 a | 5575.28 ± 41.61 | 189.62 ± 0.68 | 331.90 W | 58 C | 1845 MHz |
| 610.88 b | 5502.34 ± 50.48 | 184.37 ± 10.99 | 336.93 W | 63 C | 1845 MHz |
| 610.88 c | 5561.88 ± 48.29 | 190.12 ± 1.30 | 346.34 W | 66 C | 1845 MHz |

The 591.86-b prefill result is invalid because its within-run variance is extreme. Using the two stable 591.86 prefill runs, the baseline is 5763.92 tok/s. The 610.88 mean is 5546.50 tok/s, a raw regression of 3.77%. Median decode fell from 199.25 to 189.62 tok/s, a raw regression of 4.83%.

## Qualification findings

1. Studio 610.88 installed successfully and was visible in both Windows and WSL after a WSL restart.
2. A live driver swap invalidated the old WSL CUDA context as expected. The 8081 `/health` endpoint still returned 200, but a real embedding request failed with `CUDA error: unknown error`; restarting WSL fixed it. This confirms that health-only validation is insufficient after a driver swap.
3. After the clean 610.88 install, MSI Afterburner 4.6.5 did not reapply the saved startup fan/VF settings. The fan stayed at 0% at approximately 50 C. Explicitly loading a temporary profile copied from `Startup` also did not apply it.
4. Because the effective GPU tuning state differed, the power and temperature comparison is not a controlled driver-only A/B. It is still a practical candidate failure for this host: lower throughput plus loss of the established Afterburner control path.
5. Studio 610.47 did not complete installation during the downgrade attempt (`installer exit 1`). The transition temporarily left the RTX 3090 at device problem code 28. The pre-exported 591.86 driver package was applied with `pnputil`, restoring `CM_PROB_NONE`, and the official 591.86 installer/package was retained.
6. No new `nvlddmkm` events appeared during the test window. The seven events in the preceding 14 days all predate this run.
7. The initial Afterburner directory backup command used a wildcard with PowerShell `LiteralPath`, so it preserved the install-level `MSIAfterburner.cfg` but did not copy the per-GPU profile as intended. This limitation was discovered and recorded during recovery. The live per-GPU file retained the existing VF curve and was cleaned back to its original `Startup` values; do not treat `afterburner-backup/` as a complete per-GPU profile backup.
8. Before reboot, the cleaned per-GPU profile and the profiles-level Afterburner configuration were copied again using exact literal paths. Their hashes are recorded in `EXECUTION_LOG.md`; these exact-path copies supersede the incomplete initial directory backup warning above.

## Current state

- Effective driver: 591.86.
- RTX 3090 device state: `CM_PROB_NONE`.
- WSL kernel: `6.6.114.1-microsoft-standard-WSL2`.
- WSL CUDA probe: passed on 591.86.
- `llm-inference.service`: active; real chat request returned HTTP 200.
- `llm-embedding.service`: active; real embedding request returned HTTP 200.
- Four GitHub runner services: active; no `Runner.Worker` job was present during maintenance.
- Local-agent-fleet gateway: restarted.
- `WSL-KeepAlive` scheduled task: running.
- Afterburner profile cleanup: temporary Profile4 removed; `Startup` retains power limit 100%, core offset `-190000`, memory offset `0`, the existing VF curve, fan mode 1, fan speed 48.
- Afterburner process: intentionally left stopped until Windows reboot; its scheduled task is enabled and runs `MSIAfterburner.exe /s` at logon.

## Decision

- Reject 610.88 for this host with the current Afterburner 4.6.5 setup.
- Do not retry 610.47 in the same live session after the failed downgrade/recovery.
- Keep 591.86. The real Windows reboot and post-boot operational verification
  completed successfully; see [`POST_REBOOT_2026-08-24.md`](POST_REBOOT_2026-08-24.md).

## Post-reboot closure

The post-reboot investigation corrected an ownership error in the original
handoff: Fan Control V272 owns the fan curves, while MSI Afterburner owns the
GPU V/F and clock profile. Statements above that attribute the fan-control path
to Afterburner are superseded only on that component attribution.

The live operational profile had memory `+350 MHz`, not the temporary `mem=0`
driver-A/B tuple. It was preserved. A short operational confirmation on 591.86
passed at 6273.75 ± 98.65 tok/s prefill and 221.96 ± 1.74 tok/s decode, with
323.28 W, 44 C, 1830 MHz, and benchmark RC 0. PCIe Gen4 x16 was observed under
load and BAR1 reported 32 GiB. No new NVIDIA or WHEA error event appeared.

After the benchmark, 8080, 8081, and 8082 passed real inference requests; the
four runner units, gateway, and keepalive were present. This closes the 591.86
operational qualification. The `+350 MHz` result is not pooled with the
temporary `mem=0` baseline.

## Retained rollback receipts

- Official 591.86 installer SHA-256: `A50C89C9D254F33CC8A8E638F7CC1981A76263005FCEC102AD8C8B45626D53E0` (valid NVIDIA signature).
- Studio 610.47 installer SHA-256: `59AC4A1659664AAD0A6FC525E5DF99B3FA76887BDE663F9E36E0E7EBB5DBA937` (valid NVIDIA signature).
- Studio 610.88 installer SHA-256: `6E6E7AEB03FA8788F0E97BF0D2F66852178AA05B7C17FB4A061E1BC1CF07EA0C` (valid NVIDIA signature).
- Exported 591.86 driver-store package: `rollback-591.86/`.
