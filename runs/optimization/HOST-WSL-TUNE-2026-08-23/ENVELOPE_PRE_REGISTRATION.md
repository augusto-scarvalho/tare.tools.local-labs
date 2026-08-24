# Canonical-candidate context/VRAM envelope

Frozen after the 81,920 control failed its pre-registered 4 GiB reserve gate and before this ladder.

- Binary: canonical candidate `b10165-71676e46c`.
- Fixed runtime: MTP n3, ubatch 512, q4_0/q4_0 KV, one slot, batch 2,048.
- Ascending context ladder: 49,152; 57,344; 61,440; 65,536; 73,728; 81,920.
- Metric: clean-start free VRAM after health, with candidate teardown between points.
- Passing floor: at least 4,096 MiB free.
- Port 8081 is pre-existing down and remains untouched.

The largest passing point is only a resource-reserve candidate. No context default is changed by
this measurement.
