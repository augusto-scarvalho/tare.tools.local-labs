# Gemini execution contract

This repository uses a fail-closed backlog pipeline. These rules are mandatory
for every research-backlog task executed by Gemini.

## Start every backlog session here

```powershell
python tools/analysis/backlog_pipeline.py gate
python tools/analysis/backlog_pipeline.py status
python tools/analysis/backlog_pipeline.py next
```

Stop if `gate` fails. Do not work around a failed gate, edit historical raw
receipts, or continue a task other than the dependency-ready item selected by
`next` unless the user explicitly changes priority.

The canonical backlog is `config/research_backlog.json`. Do not edit its states
or a packet's `PIPELINE.json` manually. State changes must use
`tools/analysis/backlog_pipeline.py advance` so illegal or incomplete changes
are rejected.

## Required state sequence

`PROPOSED -> PREREGISTERED -> IMPLEMENTED -> EXECUTED -> VERIFIED -> PROMOTED`

Failure closes as `EXECUTED -> REJECTED`. A genuine external dependency may
move a nonterminal item to `BLOCKED`; record the exact unblock condition.

1. Run `scaffold` for the selected task.
2. Replace every placeholder in `PRE_REGISTRATION.md`. Freeze inputs, hashes,
   exact command, controls, sample count, acceptance thresholds, abort
   conditions and allowed claims before implementation or execution.
3. Advance to `PREREGISTERED` before changing implementation code.
4. Implement only the preregistered scope. Tests and helpers are implementation
   artifacts and must be listed with repeated `--implementation` arguments.
5. Advance to `IMPLEMENTED`; this binds every implementation file by SHA-256.
6. Execute the frozen command. It must write `raw/receipt.json` using
   `local-labs-backlog-receipt-v1`, complete experiment provenance and a
   canonical fingerprint. Start from the generated `RECEIPT.template.json`;
   do not invent or remove gates/evidence keys. Never hard-code a decisive
   metric or substitute random/synthetic tensors for real model, runtime,
   kernel, VRAM or throughput evidence.
7. Advance to `EXECUTED`. The pipeline recomputes each frozen acceptance gate
   from the recorded actual value. A self-declared boolean is insufficient.
8. Write a bounded `RESULT.md`, then stop for independent review. Gemini cannot
   author or approve `REVIEW.json`, and cannot advance `VERIFIED` or `PROMOTED`.

Random-tensor, analytical and proxy work can test software mechanics only. It
cannot support model, hardware or production promotion. Preserve failed and
superseded receipts under successor IDs; never overwrite them.

If a run touches WSL services, use `systemctl`, record service identity/PID and
restart counts, and restore both inference and embedding baselines. Do not
commit or push unless the user explicitly requests it.

See `docs/research/BACKLOG_IMPLEMENTATION_PIPELINE.md` for commands, schemas and
the independent-review handoff.
