# Post-reboot qualification — 2026-08-24

This receipt closes the real-Windows-reboot continuation for the NVIDIA driver
A/B. It supplements `RESULT.md` and `EXECUTION_LOG.md` without rewriting the
pre-reboot chronology.

## Control ownership correction

- Fan Control V272, configuration `backupzin.json`, is the fan-control owner.
- MSI Afterburner 4.6.5 owns the GPU V/F and clock profile; it is not the
  authoritative fan controller on this host.
- The pre-reboot handoff's attribution of fan control to Afterburner is
  superseded by this observed ownership split.
- No fan curve, V/F curve, power limit, or clock offset was changed during the
  post-reboot benchmark.

The live operational Afterburner tuple at qualification time was power limit
100%, core offset `-190000`, memory offset `+350000`, and a non-empty V/F curve.
The `+350 MHz` memory profile is the user's established operational profile and
is documented by earlier lab receipts. The temporary driver-A/B baseline had
used memory offset `0`; therefore the post-reboot result below is an operational
confirmation and must not be pooled as a fourth `mem=0` baseline replicate.

## Host and device

- Windows driver: 591.86.
- Device: NVIDIA GeForce RTX 3090, `CM_PROB_NONE`.
- WSL kernel: `6.6.114.1-microsoft-standard-WSL2`.
- WSL driver: 591.86; CUDA enumeration passed.
- Power limit: 420 W.
- BAR1 total: 32,768 MiB.
- PCIe link: Gen4 x16 observed during a valid 6,012-token prefill request;
  idle link correctly returned to Gen1 x16.

Two PCIe probe attempts were invalid and are excluded:

1. a Windows PowerShell request emitted malformed UTF-8 and the server rejected
   the JSON before inference;
2. the corrected ASCII request contained 19,212 prompt tokens and was rejected
   against the 8,192-token context before inference.

The corrected 6,012-token request completed and the sampler observed Gen1,
Gen2, and Gen4 transitions, always at x16 width.

## Short operational confirmation

Command:

```powershell
wsl -d Ubuntu-24.04 -u root -- systemctl stop llm-inference.service
wsl -d Ubuntu-24.04 -- bash /mnt/c/projects/tare.tools.local-labs/ops/gpu-stability/uv_bench.sh post-reboot-591.86-operational-plus350
wsl -d Ubuntu-24.04 -u root -- systemctl start llm-inference.service
```

The embedding service on port 8081 remained resident during the benchmark.

| Tuple | pp2048 | tg512 | Peak power | Peak temp | Peak SM | RC |
|---|---:|---:|---:|---:|---:|---:|
| 591.86, operational `mem +350` | 6273.75 ± 98.65 tok/s | 221.96 ± 1.74 tok/s | 323.28 W | 44 C | 1830 MHz | 0 |

The result is healthy and materially above the temporary `mem=0` reference,
but the tuning tuple differs, so the delta is not attributed to the driver.

## Restoration checks

- `llm-inference.service`: active/running; real chat returned `restored-ok`.
- `llm-embedding.service`: active/running; real embedding returned one
  768-dimensional vector.
- `llm-locale-proxy.service`: explicitly restarted after the 8080 maintenance
  window; real proxied chat returned `proxy-restored-ok`.
- Four GitHub Actions runner units: active/running.
- Local-agent-fleet gateway: present.
- `WSL-KeepAlive`: running.
- Fan Control: active and holding the GPU fan near 58% after the test.
- No new `nvlddmkm` or WHEA error event appeared during or after the benchmark.
  The day's WHEA ID 3 record is informational and predates this continuation.

## Final decision

- Keep 591.86 as the operational driver.
- Keep 610.88 rejected for this host.
- Keep 610.47 classified `INSTALL_FAILED / NOT_BENCHMARKED`.
- The real-reboot operational qualification is complete.
- The fan-control ownership correction supersedes only the earlier component
  attribution; it does not rehabilitate either rejected candidate.
