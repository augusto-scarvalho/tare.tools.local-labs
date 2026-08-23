# LAB-OPS-003 result — 81,920 is the largest passing ladder point

Decision: **QUALIFIED RESOURCE CURVE / NO DEPLOY CHANGE**

All five live-equivalent Qwen3.8 startup cells loaded and cleaned up successfully while the embedding
endpoint remained healthy.

| Allocated context | Free VRAM | 4 GiB floor |
|---:|---:|---:|
| 65,536 | 4,599 MiB | PASS |
| 81,920 | 4,151 MiB | PASS |
| 90,112 | 3,927 MiB | FAIL |
| 98,304 | 3,703 MiB | FAIL |
| 131,072 | 2,807 MiB | FAIL |

The bounded recommendation is therefore 81,920 tokens if the existing 4,096 MiB reserve remains a
hard requirement. The exact crossing lies between 81,920 and 90,112 and was not estimated beyond the
frozen ladder. This is an allocation result only; it does not show that 81,920 has adequate effective
context quality, nor does it authorize changing the live 131,072 profile.

The canonical 131k service was restored unchanged. Ports 8080 and 8081 returned healthy, the mode
lock was coherent in SERVE, and the GPU power limit remained 420 W.

Evidence: `results.json` and `logs/ctx-*.log`.
