# BeeLlama, slop.cpp, and PEFT transcript analysis - 2026-08-24

## Source identity

- Input: `tare-tools-beellama-slop-peft-conversation-transcript-2026-08-24.md`
- Source SHA-256: `f801a71ac934fd91871b6f9d36f4cf1f8c4b9a351572a5099945291ee4e5fc4d`
- Scope: proposals in the transcript are treated as hypotheses until reconciled
  with current local receipts and pinned upstream sources.

## Main finding

The transcript contains three related but distinct portfolios:

1. BeeLlama transfer: physical descriptors, route receipts, lifecycle, KV
   qualification, precision tails, and eventually KVarN.
2. Runtime challengers: stateful torture, APEX4, ReplaySSM, MoE dispatch,
   launch-overhead oracles, and historical-state recovery.
3. Adaptation research: adapter geometry, behavioral distillation, hybrid
   module targeting, prior preservation, composition, and adapter-aware cache
   identity.

The durable thesis is useful: explore broadly, but admit small falsifiable
packets and require evidence that a requested mechanism was physically
exercised. The transcript is not itself an execution plan because several
items were already completed locally and several implementation suggestions
depend on missing receipts.

## Reconciliation against current evidence

| Transcript item | Current evidence | Backlog disposition |
|---|---|---|
| `BEE-L0` source archaeology | Not previously recorded against a pinned BeeLlama head | Run first; completed in the linked receipt |
| `SLX-01` lifecycle and route receipts | Partial coverage exists in `bless_fork.sh`, recurrent rollback tests, slot save/restore tests, and local cache/cancel/reuse runs | `GAP_CONFIRMED`; define a bounded receipt-first packet before fork code |
| `SLX-02` APEX4 | Official code exists and explicitly targets RTX 3090 behavior | Run external reproducibility preflight before any port |
| `SLX-03` ReplaySSM | Upstream challenger is active, but the transcript itself gates prototypes behind lifecycle evidence | Keep dependency-gated behind `SLX-01` and a local state-traffic oracle |
| `SLX-04` MoE dispatch | Five local MoEs were already screened; routing was load-balanced and the dynamic-residency premise failed | `SUPERSEDED_CLOSED`; rerun only for a new model that fails the standing skew screen |
| `SLX-06` RNN-06D0 | D0 and D1 already qualified on synthetic Mamba-2; natural NoLiMa transfer later failed to detect useful historical signal | `SUPERSEDED`; no repeat of D0/D1 |
| `ADAPT-00` adapter geometry | New question; a suitable official 0.8B base exists, but no frozen local training packet exists | Run mechanics/data preflight after APEX4 |
| `ADAPT-01` ThinkingCap distillation | Strong causal motivation from failed rank-64 SVD extraction and successful full-rank task vector, but ADAPT-00C promoted no behavioral arm | `BLOCKED_BEHAVIORAL`; require a new preregistered budget or scale hypothesis |
| `ADAPT-02` hybrid module targeting | Scientifically useful, but confounded until an adapter method and retention protocol are selected | Gate behind `ADAPT-00/01` |
| KVarN and shared-capacity integration | Very large permanent surface and unresolved lifecycle risk | Research only; no fork integration before receipt and qualification gates |

## Prioritized sequential queue

1. `BEE-L0-SOURCE-ARCHAEOLOGY`: pin head/base, measure net delta, map contract
   surfaces, and classify transfers. Read-only.
2. `SLX-01A-GAP-AUDIT`: inventory current lifecycle tests and observability;
   mint the smallest missing packet, without changing execution.
3. `SLX-02A-APEX4-PREFLIGHT`: reproduce repository/environment/build and
   artifact availability on the RTX 3090 host. Stop before a costly matrix if
   the public package is not runnable or reproducible.
4. `ADAPT-00A-MECHANICS-PREFLIGHT`: freeze the official 0.8B base, environment,
   target/protected panels, equal-budget arms, and a small smoke training gate.
5. `BEE-L2-KV-QUALIFICATION-DESIGN`: extend existing KLD/retrieval/task harnesses
   only after the earlier gates identify a representation worth testing.

No item in this queue authorizes a permanent `slop.cpp` format, default change,
or production service mutation. Expensive follow-ons remain dependency-gated.

## Acceptance policy

- External claims require pinned source and artifact identity.
- Build support is not runtime qualification.
- Kernel speed without relevant end-to-end gain is a negative result.
- Training gain without protected-set retention is a negative result.
- Any silent fallback, partial state publication, cross-slot contamination, or
  incompatible cache reuse blocks promotion.
- Existing negative evidence is not rerun unless a concrete trigger changed.

## Linked receipts

- `runs/research/BEE-L0-SOURCE-ARCHAEOLOGY-2026-08-24/RESULT.md`
- `runs/research/SLX-01A-GAP-AUDIT-2026-08-24/RESULT.md`
- `runs/research/SLX-02-APEX4-2026-08-24/RESULT.md`
- `runs/research/ADAPT-00A-MECHANICS-2026-08-24/PRE_REGISTRATION.md`
- `runs/research/ADAPT-00A-MECHANICS-2026-08-24/RESULT.md`
- `runs/research/BEE-L2-KV-QUALIFICATION-DESIGN-2026-08-24/RESULT.md`
- `runs/research/ADAPT-00B-GEOMETRY-MATRIX-2026-08-24/RESULT.md`
- `runs/research/ADAPT-00C-BEHAVIORAL-2026-08-24/RESULT.md`
