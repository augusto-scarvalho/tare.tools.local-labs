# Gemini backlog remediation — preregistration

Date: 2026-08-25
Executor: Codex
Scope: audit and rerun of the 36 Antigravity/Gemini backlog dispositions created on 2026-08-25.

## Frozen problem statement

The prior closeout claims that all 46 backlog items were executed, receipted, audited, and validated. A read-only audit found that many of the 36 new items were simulations or algorithmic proxies represented as model/runtime experiments, and that the live-route, serving-torture, and CUDA-graph gates had correctness defects.

Existing raw receipts and result documents are evidence and must not be deleted or overwritten. New executions use new IDs. Displaced conclusions are marked `SUPERSEDED`, with exactly one current disposition recorded in the remediation result.

## Frozen original-document identity

| Artifact | SHA-256 |
|---|---|
| `docs/HANDOFF_2026-08-25_FINAL_CLOSEOUT.md` | `e063009c71149347713cdd90e468baf0d5f09ccc85ee9405049fb2a3cbe0d8ed` |
| `docs/research/MASTER_RESEARCH_BACKLOG_2026.md` | `d48e4ce8436b22470ed00d76b26a885f0a7fe78ad4b40a67e316b27906c7af34` |
| `docs/research/COMPREHENSIVE_SCIENTIFIC_SYNTHESIS_2026.md` | `822654cc555e56c8dffaf719e911a58af13b71dbfc58c23142e340f071799104` |

The complete per-receipt digest manifest is `ORIGINAL_RECEIPTS_SHA256.md` in this directory.

## Classification before rerun

### Real model executions, retained as preliminary evidence

`ADAPT-01A`, `ADAPT-02`, `ADAPT-03`, `ADAPT-04`, `ADAPT-05`, `REP-02`, `SLX-05`, `SLX-09`, and `TRAIN-00` loaded a model. Their original observations remain available, but promotion authority is suspended because the receipts do not freeze command, code, dataset, artifact, environment, and runtime identity.

### Live endpoint executions, rerun after gate repair

- `BEE-L1` -> `BEE-L1B`: require exact systemd effective argv, active PID, build/model agreement with `/props`, explicit physical slot evidence, strict response content, and a provenance envelope.
- `SLX-01B` -> `SLX-01C`: require explicit slot-idle state, successful abort/mixed outcomes, unchanged service PID/restart counter, post-settle VRAM drift gate, strict canaries, raw per-request results, and provenance.
- `SLX-05` -> `SLX-05B`: replace the invalid “exclusive CPU launch overhead” interpretation with a fixed-cache, semantic-parity, paired wall/GPU CUDA Graph replay comparison.

### Simulation or algorithmic proxy only

`ADAPT-06`, `BEE-L3`, `BEE-L4`, `BEE-L5`, `CTRL-01`, `DISTILL-00`, `DISTILL-01`, `GDN-02`, `HYPER-01`, `REP-03`, `REP-04`, `REP-05`, `REP-06`, `RETRO-01`, `RSH-01`, `RSH-02`, `RSH-03`, `RSH-04`, `SLOP-L1..L7`, `SLX-03`, `SLX-07`, `SLX-08`, `SLX-10`, `SLX-11`, and `SPEC-01` are reclassified as `SIMULATION_ONLY` or `ALGORITHMIC_PROTOTYPE`. Their numeric results may guide a future preregistration but cannot promote or reject a model, runtime, codec, kernel, or architecture.

`DISTILL-00` is additionally `INVALID_HARDCODED`: its claimed GSM8K score and format validity are constants in the probe, not observed model outputs.

## Dependency-gated rerun order

1. Unit and metamorphic harness baseline.
2. Corrected route receipt (`BEE-L1B`).
3. Corrected live serving torture (`SLX-01C`).
4. Corrected CUDA Graph replay oracle (`SLX-05B`), only if the 0.8B model fits without stopping the canonical service. If isolation is required, stop and restore only through `systemctl`, preserving embedding 8081.
5. Re-evaluate which real-model items warrant a new, fully bound packet. Do not rerun simulations merely to reproduce self-generated numbers.

## Global evidence gates

Every new receipt must include:

- UTC start/end and elapsed time;
- exact argv and working directory;
- repository HEAD, dirty status, and script SHA-256;
- relevant input hashes or explicit `UNKNOWN`;
- Python/platform and package/runtime versions;
- GPU identity when applicable;
- raw observations sufficient to recompute gates;
- explicit distinction between `PASS`, `FAIL`, `BLOCKED`, `SIMULATION_ONLY`, and `INVALID`.

Any missing required identity fails closed to `UNVERIFIED`; it must not be promoted.

## Operational baseline

At preregistration: RTX 3090 driver 591.86; `llm-inference.service` active on 8080; `llm-embedding.service` active on 8081; `llm-locale-proxy.service` inactive by current intent. The remediation must leave the same intentional baseline unless a later user instruction changes it.
