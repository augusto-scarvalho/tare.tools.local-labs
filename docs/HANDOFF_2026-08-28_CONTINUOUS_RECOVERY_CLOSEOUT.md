# Continuous recovery closeout and next-agent handoff — 2026-08-28

Read this document first after a context reset. It supersedes the operational
queue in `HANDOFF_2026-08-28_REJECTED_HOLD_RECOVERY_WAVE.md`; the older handoff
remains the evidence-preserving account of the preceding six-packet audit.

## Current state

- Canonical ledger: `config/research_backlog.json`.
- Machine-readable continuation: `config/research_trails/continuous_recovery_2026-08-28.json`.
- Detailed scientific closeout: `docs/research/CONTINUOUS_RECOVERY_CLOSEOUT_2026-08-28.md`.
- Backlog population: 108 records — 24 `PROMOTED`, 21 `REJECTED`, 19
  `EXECUTED`, 39 `BLOCKED`, two `IMPLEMENTED`, and three `PROPOSED`.
- Pipeline gate: passing at closeout.
- Repository test suite: 274 passing tests at closeout.
- Serving baseline: qualified-model gateway healthy on port 8080 with
  `qwen38` resident; embeddings healthy on port 8081.
- No experiment watcher or research job is active. The only GPU workload is
  the qualified Qwen3.8 serving backend.

## What this wave established

Nine final successor packets were promoted and one diagnostic predecessor was
rejected. The decisive outcomes are:

1. Blind numeric relabeling recovered policy/scorer false negatives and fed
   immutable trace and Q8 final aggregators.
2. The trace deployment finalist gained 29/256 paired answers over its frozen
   comparator: delta 0.11328125, 95% bootstrap CI 0.046875 to 0.1796875.
3. Qwen3.8 Q8 KV remained noninferior on the broad retained panel and produced
   exact paired retrieval at 8k/16k and under two physically overlapping slots.
4. SLX-03 GDN fusion was separately established at build, live route and causal
   performance levels. Decode ratio was 1.051362, with 95% hierarchical-
   bootstrap CI 1.041026 to 1.062275 over 144 fixed-work requests.
5. Q8 long-context R1 is a preserved harness negative: raw `/completion`
   yielded 40/48 empty one-token EOS responses symmetrically. Chat-contract R2
   returned 48/48 correct non-empty responses. Do not cite R1 as Q8 inferiority.

The complete result table, review hashes, failed predecessor explanations and
claim boundaries are in the detailed closeout linked above.

## Canonical next queue

Three validated successors are admitted as dependency-ready `PROPOSED` work.
They deliberately reuse retained physical outputs and should not consume GPU:

1. `BACKLOG-FLEET-CONTEXT-ENVELOPE-04` (P0): reconstruct and independently
   recompute all 72 retained per-slot retrieval rows from a final immutable
   source set. This is the current output of `backlog_pipeline.py next`.
2. `BACKLOG-FLEET-CONTEXT-INTERFERENCE-02` (P0): reconstruct all 72 retained
   hard-decoy rows, including the exact 31-decoy construct, from immutable
   sources.
3. `BACKLOG-GATEWAY-ROUTE-STRESS-02` (P1): recompute the retained 30-switch,
   120-request campaign while separating HTTP transport and route identity from
   semantic content eligibility. Empty reasoning-only vision responses may not
   be counted as semantic success.

The first two may run in either order, but complete and independently review
each packet before signing a claim. Gateway R2 follows them by priority.

## Restart procedure

Run these commands from the repository root:

```powershell
python tools/analysis/backlog_pipeline.py gate
python tools/analysis/backlog_pipeline.py status
python tools/analysis/backlog_pipeline.py next
```

Expected `next` result:

```text
BACKLOG-FLEET-CONTEXT-ENVELOPE-04 P0
```

Then scaffold and preregister only that selected item. The executor stops at
`EXECUTED`; a fresh independent reviewer must recompute the raw evidence,
search for false positives and false negatives, and alone decide the terminal
scientific state.

## Non-negotiable boundaries

- Do not mutate or replace prior raw evidence. Import it by exact digest into a
  successor packet.
- Do not convert a provenance/harness failure into a scientific model failure.
- Do not infer scientific success from a passing unit test or watcher exit.
- No P3 successor authorizes fresh inference; all three are retained-evidence
  recomputations.
- The fleet context claims cover only the frozen associative-retrieval
  constructs and observed per-slot limits, not general reasoning or RAG.
- The gateway claim covers bounded transport, route identity and repeatability,
  not vision capability, model quality, multi-resident serving or production
  reliability.
- Serving work must preserve the gateway on 8080 and embeddings on 8081 and
  record restoration evidence.
- Commit and push remain separate maintainer actions. This handoff's publication
  commit does not authorize later automatic pushes.

## Watcher contract

For a future authorized continuous run, use the persisted watcher with a
300-second normal polling interval and compact notifications only for
completion, failure or an audit barrier. Full progress belongs on disk under
`runs/autonomous/`. The watcher may validate evidence and advance
`IMPLEMENTED -> EXECUTED`; it may not approve its own science, skip review, or
select work outside `backlog_pipeline.py next` without explicit reprioritization.
