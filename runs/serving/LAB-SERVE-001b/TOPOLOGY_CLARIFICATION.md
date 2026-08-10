# LAB-SERVE-001b — Topology Clarification (issued by LAB-SERVE-001c Stage 1, 2026-08-10)

This note **supersedes one topology description** in the LAB-SERVE-001b report/handoff. The 001b **raw
evidence and all measured metrics remain valid and immutable**; nothing in the original report is
edited. Full reconciliation evidence: `../LAB-SERVE-001c/topology/` (raw `/props`, raw `/slots`,
startup logs, `topology_interpretation.json`).

## Previous claim (001b report §"Server topology", handoff §2)
> total_slots=4, kv_unified=true, **n_ctx=8192 shared**; slot0 n_ctx 8192.

## Raw evidence that contradicted it
The committed `runs/serving/LAB-SERVE-001b/{dense,moe}/raw/blocks.json` recorded, in **all 20 blocks**:
```
argv:  ... --ctx-size 8192 --parallel 4 ...
topology: { total_slots: 4, n_ctx: null, default_gen_n_ctx: 2048, num_slots: 4, slot0_n_ctx: 2048 }
```
The per-slot context was **2048**, not 8192. The `8192` in the prose came from an earlier **manual
curl on a differently-configured server**, not from the recorded `--parallel 4` runs.

## Root cause (pinned fork `068764d92`)
- `tools/server/server.cpp:146-151` — `kv_unified=true` (and auto `n_parallel=4`) is set **only** inside
  `if (params.n_parallel < 0)` (the *auto* case). The 001b argv passed `--parallel 4` **explicitly**, so
  this branch never fired and `kv_unified` kept its **default `false`** (`src/llama-context.cpp:3498`).
- `src/llama-context.cpp:286-300` — with `kv_unified=false`, `n_ctx_seq = n_ctx / n_seq_max = 8192/4 =
  **2048**` (static KV partition). With `kv_unified=true` it would be `n_ctx_seq = n_ctx = 8192` (shared).
- Capture defect (not a server anomaly): the `n_ctx: null` field came from `topology()` reading
  `props.get("n_ctx")`, but `/props` has **no top-level `n_ctx`** key — only
  `default_generation_settings.n_ctx`. The effective 2048 *was* correctly captured in
  `default_gen_n_ctx`/`slot0_n_ctx`.

## Empirical confirmation (this Stage 1)
Re-ran the pinned server twice with the exact 001b MoE base argv:
| config | startup log | /props dg.n_ctx | /slots per-slot |
|---|---|---|---|
| `--parallel 4` (001b) | `n_slots=4, n_ctx_slot=2048, kv_unified='false'` | 2048 | [2048,2048,2048,2048] |
| auto (no `--parallel`) | `n_slots=4, n_ctx_slot=8192, kv_unified='true'`  | 8192 | [8192,8192,8192,8192] |

The auto case exactly reproduces the earlier "8192" claim; the explicit case exactly reproduces the
recorded blocks.json. Discrepancy fully explained.

## Correct interpretation of the 001b topology
**4 slots × 2048 tokens each, statically partitioned (kv_unified=false), 8192 aggregate KV.**
`configuredGlobalContext=8192 · serverSlotCount=4 · effectiveSlotContext=2048 ·
maximumContextForOneActiveSlot=2048 · maximumContextForFourActiveSlots=2048/slot`.

## Impact on 001b validity — NONE (results stand)
The 001b workload was **1024 input + 128 output = 1152 tokens/request worst case**, inside the **2048**
per-slot envelope (896-token margin). Under static partitioning each of the 4 slots kept its full 2048
even at N=4, so **no request was context-truncated** and no measured metric changes. Both MTP arms used
the **identical** topology, so the paired on−off deltas are unbiased. The correction is **purely
descriptive** (partitioned 2048/slot, not shared 8192).

## Consequence for 001c (carried into Stage 2)
`effectiveSlotContext=2048` is **too small** for the 001c coding class (up to ~18k tokens/request). 001c
must pin a **larger, explicitly-verified** context envelope (memory-feasibility canary on the 24GB 3090
before pre-registering load points). See `../LAB-SERVE-001c/topology/topology_interpretation.json`.
