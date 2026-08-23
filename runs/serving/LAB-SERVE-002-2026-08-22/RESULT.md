# LAB-SERVE-002 workload-profile decision

Decision: **HOLD_MODEL_DRIFT / LEGACY CONFIG CHARACTERIZED / NO DEFAULT CHANGE**

LAB-SERVE-001c is qualified evidence for its exact Qwen3.6-35B-A3B MoE configuration: 12/12 cells
completed, request errors were zero, token accounting was 1.000, sustainable capacity was about
0.09 req/s, and queueing onset lay between 0.072 and 0.110 req/s. MTP improved median and p95 E2E
latency at every tested load point in that bounded campaign.

It cannot promote a workload profile for the live service. The current endpoint reports
`qwen38-27b`, 27,320,697,856 parameters and 131,072 context, whereas LAB-SERVE-001c used the
Qwen3.6-35B-A3B MoE candidate with `--n-cpu-moe 8`, Q8 KV, four statically partitioned slots and a
73,728 global context. The model, topology and memory regime all differ materially.

The historical labels in `DECISION_PACKET.md` are therefore descriptive guidance only. No
`SERVE_PROFILES` entry, systemd setting, context size, load limit or deployment default was changed.
The cancelled reliability run remains incomplete and was not used as a pass.

The next executable promotion experiment is a same-artifact Qwen3.8 open-loop packet at LOW and
NEAR load, three replicates, using the frozen gates in `DECISION_PACKET.md`. It should only run after
the canonical context policy chooses between the current 131k allocation and the measured 81,920
token 4 GiB-reserve point; otherwise the memory envelope would be an uncontrolled difference.

