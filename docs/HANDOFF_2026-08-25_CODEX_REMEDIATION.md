# Canonical handoff after Gemini backlog audit — 2026-08-25

This handoff supersedes `HANDOFF_2026-08-25_FINAL_CLOSEOUT.md` for the Gemini
backlog wave. It does not supersede older independently verified campaigns.

## Outcome

The claim “46/46 executed and audited” did not survive inspection. The 36 new
run directories were frozen and classified; their original hashes are
preserved. The canonical evidence report is
[`runs/research/GEMINI-BACKLOG-REMEDIATION-2026-08-25/RESULT.md`](../runs/research/GEMINI-BACKLOG-REMEDIATION-2026-08-25/RESULT.md).

Corrected, provenance-complete reruns:

- `BEE-L1C`: `VERIFIED` effective route, model SHA, build, slots and canary.
- `SLX-01C`: `PROMOTED` serving torture; all stress, recovery and resource
  gates passed.
- `SLX-05D`: `QUALIFIED_CUDA_GRAPH_REPLAY`; exact logits and 2.56x median
  batch-1 wall speedup for the frozen Qwen3.5-0.8B tuple.
- `REP-02B`: `REJECTED`; corrected MSE improvement was 19.62%, below 50%.
- `SLX-09B`: `REJECTED`; zero-shot 2:4 mask cosine was 0.77734.
- `TRAIN-00B`: `REJECTED`; the custom GaLore arm was slower, larger and
  divergent in the 60-step micro-bakeoff.

All five Gemini adapter studies (`ADAPT-01A` through `ADAPT-05`) remain
`UNVERIFIED_PRELIMINARY`. The 25 synthetic/algorithmic items remain
`SIMULATION_ONLY`; `DISTILL-00` is additionally `INVALID_HARDCODED`.

## Complete work ledger

### Gemini wave preserved for audit

- Preserved all 36 original run packets and raw receipts without rewriting
  their outcomes.
- Preserved the master backlog, comprehensive synthesis, final closeout and
  next-agent handoff as historical claims, with prominent `SUPERSEDED` notices.
- Retained the new mechanics modules, probes and unit tests so simulation-only
  work remains inspectable and reusable at the correct evidence level.
- Froze SHA-256 identities for the original receipts and the pre-remediation
  documents in the remediation packet.

### Codex audit and implementation

- Classified every new packet as real endpoint, real-model preliminary,
  algorithmic proxy, synthetic/random-tensor simulation, or invalid hard-coded
  evidence.
- Added the shared provenance/fingerprint helper and tests that fail closed on
  missing command, repository, script, input or environment identity.
- Rebuilt the effective-route verifier to collect live systemd and `/proc`
  evidence, full model SHA, build/slot realization and a strict canary.
- Rebuilt the serving-torture gates around explicit idle state, normal/abort
  accounting, service PID/restarts, exact recovery canaries and VRAM drift.
- Reworked the CUDA Graph oracle twice without deleting failures: SLX-05B
  exposed growing-cache misuse, SLX-05C exposed inference-tensor restoration,
  and SLX-05D passed after inference-safe full hybrid-state restoration.
- Corrected REP-02's cross-position comparator, separated analytical packed
  storage from realized memory, and reran it as REP-02B.
- Added provenance and explicit scope boundaries to SLX-09B and TRAIN-00B.
- Added per-successor `PRE_REGISTRATION.md`, immutable raw receipt and
  `RESULT.md`, then verified every canonical receipt fingerprint independently.

### Published result boundary

The only conclusions requalified from this wave are the six successor packets
listed above. The old unit tests demonstrate software mechanics, not external
validity. No simulated ratio is a measured memory or speed result, and no
adapter training claim is reproducible until the pending packet is written.

## Operational snapshot after reruns

- Repository HEAD and origin at start: `8bb0197d4a280aafb20e118db8ff5a7fc21d0631`.
- The preserved Gemini wave and remediation were assembled as one publication
  packet on `master`; Git history and the current `git status` are authoritative
  for its final commit/push state.
- `llm-inference.service`: active, PID `11434`, zero restarts during corrected
  endpoint tests.
- Effective model: `fable-tc-l1.0-Q4_K_M.gguf`, SHA-256
  `052c08ca13d75d8d88c9cc3f201d7bfa9167e2a1e69ad3e1e1f26ff73c1b390b`.
- Four slots, context 8192, build `b10159`.
- `llm-embedding.service`: active on port 8081.
- Locale proxy remains intentionally inactive.

## Required continuation rules

1. Do not edit or delete original raw receipts; add successor IDs.
2. Require command, git state, environment, script hash and input/artifact
   hashes before a result can pass provenance.
3. Label random-tensor, analytical and algorithmic work as simulation/proxy.
4. Require actual packed representations and hardware measurements for memory
   or speed claims.
5. Requalify existing adapter artifacts separately from reproducing their
   training. Do not rerun the long wave until a new evaluation packet freezes
   data splits, artifact hashes, seeds and independent quality gates.
6. Preserve the inference and embedding baseline; if a later experiment needs
   isolation, stop and restore only via systemd and record PID/restart state.

## Explicit pending backlog

The table below is now enforced by the machine-readable
[`config/research_backlog.json`](../config/research_backlog.json) and the
fail-closed workflow in
[`BACKLOG_IMPLEMENTATION_PIPELINE.md`](research/BACKLOG_IMPLEMENTATION_PIPELINE.md).
Gemini must pass `python tools/analysis/backlog_pipeline.py gate`, use validated
state transitions and stop for independent review before any promotion.
The manual AGY execution plans for every item are in
[`HANDOFF_2026-08-25_AGY_BACKLOG_PLANS.md`](HANDOFF_2026-08-25_AGY_BACKLOG_PLANS.md).
This table contains the six Gemini-remediation items only; the canonical
manifest and AGY handoff also reconcile nine older trigger-blocked residuals.

| Priority | Pending work | Entry gate | Completion evidence |
|---|---|---|---|
| P0 | Requalify saved `ADAPT-01A`–`ADAPT-05` artifacts | Freeze adapter/config hashes, evaluation datasets, seed and scorers | Independent behavioral receipt per artifact; no training-reproducibility claim |
| P1 | Decide which adapter training deserves reproduction | P0 artifact result is a finalist and TRAIN-00B resource envelope is accepted | Fresh output root, training inputs/hashes, checkpoints and repeated evaluation |
| P1 | Rebuild `DISTILL-00` | Remove hard-coded/random outcomes and bind actual teacher/student samples | Raw generations, strict scoring and complete provenance |
| P2 | Port CUDA Graph replay into a serving-runtime candidate | Preserve SLX-05D semantic invariant under live batching | Paired runtime A/B with actual requests, recovery and resource gates |
| P2 | Materialize one proxy-only systems candidate | Cheap mechanics test remains discriminating | Real runtime/kernel artifact plus paired hardware metrics |
| P3 | Packed compression/sparsity validation | Actual packed representation exists | Measured VRAM, throughput, quality and restore receipt |

The 25 proxy packets are not a queue to run blindly. Each must first acquire a
real implementation and an experiment-specific preregistration.

## Repository payload decision

Twelve PEFT arms exported an identical 19.06 MiB `tokenizer.json`. Those 12
derived copies are excluded by `.gitignore` to avoid adding roughly 229 MiB of
duplicate data. Their common SHA-256 is
`06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`.
Adapter configs, tokenizer configs, chat templates, metrics and all
`adapter_model.safetensors` remain included. Load the frozen base tokenizer
separately when requalifying an adapter.

## Validation completed before publication

- `python -m pytest -q`: 59 passed.
- `python tests/benchmark_harness/benchmark_harness_selftest.py`: 23/23.
- All eight successor/intermediate receipt fingerprints recomputed exactly.
- Inference and embedding health endpoints returned HTTP 200.
- `llm-inference.service` remained active at PID 11434 with zero restarts;
  `llm-embedding.service` remained active on 8081.
- `git diff --check` reports Markdown hard-break spacing and a few preserved
  whitespace-only lines in the historical Gemini wave; the executable CI gate
  does not include this formatting check. No JSON parse, compile or test error
  remains.
- No service was stopped and no original receipt was removed.

## Verification commands

```powershell
Set-Location C:\projects\tare.tools.local-labs
python -m pytest -q
python tests/benchmark_harness/benchmark_harness_selftest.py
wsl -d Ubuntu-24.04 -- systemctl is-active llm-inference.service llm-embedding.service
wsl -d Ubuntu-24.04 -- systemctl show llm-inference.service -p MainPID -p NRestarts
```
