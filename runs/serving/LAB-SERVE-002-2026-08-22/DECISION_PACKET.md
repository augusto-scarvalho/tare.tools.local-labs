# LAB-SERVE-002 workload-profile promotion packet

Status: **FROZEN BEFORE DECISION**

## Question

Can the completed LAB-SERVE-001c open-loop measurements be promoted into named workload-specific
serve profiles without weakening artifact identity, reliability, or deployment safeguards?

## Evidence admitted

- `runs/serving/LAB-SERVE-001c/PRE_REGISTRATION.md`
- `runs/serving/LAB-SERVE-001c/report/LAB-SERVE-001c.md`
- `runs/serving/LAB-SERVE-001c/campaign/normalized/summary.json`
- `runs/serving/LAB-SERVE-001c/workload/workload_manifest.json`
- `runs/reliability/LAB-REL-001-24h-2026-08-21/` as an incomplete, user-cancelled receipt only
- Current `/v1/models`, service state and LAB-PROV-001 artifact inventory captured on 2026-08-22

No new throughput result may be introduced during this decision. Existing measurements are not
transferred across model identities.

## Candidate labels

The old Qwen3.6-35B-A3B MTP configuration may receive descriptive, non-default labels only:

- `interactive-characterized`: LOW offered load 0.030 req/s; measured completion 0.031 req/s.
- `mixed-near-capacity-characterized`: offered 0.072 req/s; measured completion 0.070 req/s.
- `overload-boundary`: 0.110 req/s is not a promoted operating point; measured completion was about
  0.091 req/s and queue-tail inflation was visible.

These labels describe one artifact/configuration/workload. They are not aliases for the current
Qwen3.8 service and do not change `SERVE_PROFILES` or systemd defaults.

## Frozen promotion gates

Promotion requires all of the following:

1. Exact model content digest, runtime revision, flags, KV type, context topology and workload digest
   match the evidence packet.
2. The profile is measured on the model that will actually serve it. Historical Qwen3.6 results
   cannot promote Qwen3.8.
3. At LOW and NEAR load, all cells complete with zero request errors, token accounting 1.000, and
   completed rate at least 95% of offered rate.
4. Three clean replicates agree on the direction of median and p95 E2E latency. A two-replicate
   historical screen remains characterization evidence, not a promotion gate.
5. A bounded same-profile reliability run completes. The cancelled 369/369 partial run is not a
   substitute and is never relabeled PASS.
6. The 4 GiB VRAM-reserve policy is explicit. If the profile intentionally uses a smaller reserve,
   that exception must be named in the profile rather than hidden in a default.

Decision vocabulary: `PROMOTE`, `HOLD_MODEL_DRIFT`, `HOLD_RELIABILITY`, or `REJECT`.

