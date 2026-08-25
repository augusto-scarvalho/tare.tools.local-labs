# SLX-01C serving-torture result

Verdict: `PROMOTED`

All corrected gates passed: 20/20 normal requests completed, 20/20 aborts
ended cleanly, and the mixed phase completed 10/10 normal requests plus 10/10
clean aborts. All four slots explicitly reported `is_processing=false`; 5/5
strict canaries passed. The service PID remained `11434`, `NRestarts` remained
zero, and VRAM drift was +14 MiB against the preregistered +20 MiB limit.

Provenance is complete. Receipt fingerprint:
`912a97124f5890c74965118f48d7f1229c57479262c1e0587d6ea3b7deadf7fc`.
This supersedes SLX-01B, whose idle and phase-success predicates were
insufficient.

