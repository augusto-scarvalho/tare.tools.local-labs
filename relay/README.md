# Local Labs Relay (tare.tools.local-labs)

This directory documents the **GitHub ↔ Google Drive relay** used to coordinate Local AI Lab work
between an auditor and an implementer. It is intentionally tiny: schema examples plus this note.

## What the relay is (and is not)

- **GitHub Issue = coordination / provenance, not Authority.** The Issue thread carries the message
  sequence (BACKLOG → ACK → RESULT → audit) and is the durable, ordered record of *what was agreed and
  done*. It does not itself grant authority; the authority envelope is stated explicitly in each backlog
  message and is bounded there.
- **Google Drive = bulky / transient evidence transport.** Large or throwaway evidence bytes (manifests,
  logs, diffs, diagnostics, rollback notes) live in Drive, not in Git. Git stays small and reviewable.
- **`READY.json` is written last.** An evidence package is only complete once every payload and hash is
  final and `READY.json` has been written as the final file. No `READY.json` ⇒ incomplete package.
- **No secret payloads.** Neither the Git candidate nor the Drive evidence carries credentials, tokens,
  private keys, model weights, or benchmark datasets. Secrets are sourced at runtime from the OS
  keyring / environment, never committed.

## Local Labs Drive logical path

Evidence is written only below the authorized **logical** path (a Windows junction):

```
C:\projects\tare-tools-relay\temporary-evidence\agent-relay\tare.tools.local-labs\
```

- The **same-user junction is operational path discipline, not OS confinement** — it is not a sandbox or
  security boundary. It keeps evidence writes scoped to the Local Labs subtree; it does not, by itself,
  prevent access elsewhere. The parent Drive tree is not enumerated.

## Normal state flow

```
AUDITOR posts BACKLOG (Issue)
  → IMPLEMENTER posts ACK / PLAN (Issue)
  → IMPLEMENTER creates candidate branch + tiny commit
  → IMPLEMENTER writes Drive evidence (READY.json last)
  → IMPLEMENTER opens DRAFT PR (no merge)
  → IMPLEMENTER posts RESULT (Issue)
  → AUDITOR independently verifies (Issue / Git / PR / Drive)
  → AUDITOR posts ACCEPT | ACCEPT_WITH_CAVEATS | CORRECTIVE_REQUIRED | BLOCKED | OWNER_AUTH_REQUIRED
```

Merge and Issue closure happen only after independent audit; the implementer never self-merges, never
marks a PR ready-for-review, and never force-pushes or rewrites history.

## Message examples

`examples/` contains schema-shaped JSON examples for each relay message kind: `ACK.json`, `RESULT.json`,
`BLOCKER.json`, `OWNER_AUTH_REQUIRED.json`. They are illustrative shapes only — placeholders, no live
identifiers, no secrets.
