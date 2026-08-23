# LAB-OPS-003 — canonical context/VRAM envelope

Frozen before execution on 2026-08-22.

## Question

What is the largest preallocated context in a small fixed ladder that keeps at least 4,096 MiB of
free RTX 3090 VRAM for the live-equivalent Qwen3.8 service alongside the embedding endpoint?

## Fixed configuration

Use the canonical model/binary and flags: Qwen3.8 27B Q4_K_XL, q4_0/q4_0 KV, MTP n3, one slot,
32 checkpoints, flash attention, all GPU layers, batch 2,048, explicit ubatch 512 (the binary default),
default mmap and 420 W. Only context changes. Port 8092 is used in LAB mode; 8081 stays live.

## Ladder and decision

Measure contexts 65,536; 81,920; 90,112; 98,304; and 131,072. For each, record load success/time
and `nvidia-smi` used/free memory immediately after health. The largest ladder point with at least
4,096 MiB free is the bounded envelope recommendation. This is an allocation screen, not a quality
or long-context benchmark, and it does not mutate the systemd profile.

Abort on embedding health loss, cleanup failure or less than 16 GiB host RAM. Restore the canonical
service unchanged and verify both endpoints afterward.
