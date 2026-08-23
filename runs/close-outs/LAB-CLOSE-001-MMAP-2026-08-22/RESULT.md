# LAB-CLOSE-001 — mmap residual close-out

**Status:** `COMPLETE / QUALIFIED / HISTORICAL RESIDUAL CONFOUNDED`

The historical claim that `--no-mmap` imposed an approximately 10.4% decode cost
did not reproduce at its original Qwen3.6-35B-A3B placement. Warm-cache decode was
noise-equivalent. Fresh-process evidence instead supports `--no-mmap` for this exact
MoE/ncmoe=6 configuration because it reduced total process time and avoided a severe
lazy-page-in exposure. This finding does not transfer automatically to the current
dense Qwen3.8 service, whose defaults were not changed.

## Decode result

| Arm | n | Median tok/s | Range tok/s | Median process time | Median max RSS |
|---|---:|---:|---:|---:|---:|
| mmap ON | 6 | 99.41 | 17.98–103.88 | 11.12 s | 21,993 MiB |
| mmap OFF | 6 | 100.43 | 97.96–106.41 | 8.94 s | 4,159 MiB |

- All-pair median no-mmap decode delta: **+3.69%**, bootstrap 95% CI
  **−0.46% to +252.20%**. The extreme upper bound is caused by the retained cold
  mmap run and is not a stable throughput estimate.
- Warm-cache paired median (repetitions 1–5): **+0.18%**, bootstrap 95% CI
  **−0.87% to +43.34%**. The interval includes zero, so decode is classified
  `NOISE`; the old −10.4% no-mmap penalty is `CONFOUNDED`.
- Arm medians differ by only +1.03% for no-mmap, below the standing 2.3% hardware
  noise floor.

## Operational finding

`--no-mmap` reduced paired fresh-process elapsed time by a median **10.87%**,
bootstrap 95% CI **3.98% to 28.29%**. All six paired elapsed-time deltas favored
no-mmap. The first mmap process incurred 940 major faults, about 45 GB of filesystem
input and only 17.98 tok/s; the corresponding no-mmap process had 40 major faults
and 100.88 tok/s. Once cached, mmap generally recovered to about 99–104 tok/s, but
another retained mmap run fell to 68.34 tok/s without major faults.

Max RSS is not the same as committed anonymous memory—mapped file pages can be
shared/reclaimed—but it is a useful topology receipt: mmap exposed about 22 GB RSS,
whereas no-mmap materialized about 4.2 GB for the fixed six-layer CPU-MoE placement.

Therefore:

- use `--no-mmap` for the tested Qwen3.6-35B-A3B Q4_K_M / `ncmoe=6` fresh-process
  profile;
- do not claim a decode-speed advantage beyond noise;
- do not mutate the current dense Qwen3.8 service based on this MoE-specific result.

## Protocol and validity

- Build `5e7f6271c` (`b9863`); model 22,663,387,424 bytes.
- `-ngl -1 -ncmoe 6 -fa on -d 8192 -p 0 -n 64`, one fresh process per cell.
- Six paired repetitions with alternating order and 25-second cooldowns.
- 12/12 zero exits and parseable JSON rows; all cells read 420 W power limit,
  34–37 C idle-start temperature and 210 MHz idle SM clock.
- GPU clocks/voltage were observed, not mutated.
- The text service was stopped via `systemctl`; embedding port 8081 stayed healthy.
  The canonical Qwen3.8 service was then restored and verified healthy on 8080;
  board power remained 420 W.

## Evidence

- `PRE_REGISTRATION.md`: frozen protocol and decision rule.
- `raw.log`: complete stdout/stderr and `/usr/bin/time -v` receipts, SHA-256
  `88b512e717ce86b4dad51b4b303dd5b9f02ca25d3c802f374e98a4c18d9876a7`.
- `summary.json`: parsed rows, paired deltas and bootstrap output, SHA-256
  `c13713513cb2e940d3349b83efb1a2f4b7c544d2bc88baa0cb2cdebc411273d4`.
- Collector: `ops/close-outs/mmap_ab_qualified.sh`.
- Summarizer: `tools/analysis/mmap_ab_summarize.py`.

