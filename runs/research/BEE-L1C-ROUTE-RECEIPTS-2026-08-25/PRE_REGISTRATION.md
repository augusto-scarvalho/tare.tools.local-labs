# BEE-L1C effective-route receipt — preregistration

Executor: Codex  
Predecessors: `BEE-L1` (`SUPERSEDED_GATE_DEFECT`) and `BEE-L1B` (`INVALID_CANARY_BUDGET`)

`BEE-L1B` bound the service/model/build successfully but its 16-token thinking-enabled canary produced no final content. This correction freezes `chat_template_kwargs.enable_thinking=false` and 64 output tokens; all other BEE-L1B gates remain unchanged.

- Verifier SHA-256: `7bc17992f190d5493bbf1cf34f4660067f40f08367c9ddac5a03e5c827b1ffb9`.
- Expected model SHA-256: `052c08ca13d75d8d88c9cc3f201d7bfa9167e2a1e69ad3e1e1f26ff73c1b390b`.
- Output: `runs/research/BEE-L1C-ROUTE-RECEIPTS-2026-08-25/raw/receipt.json`.

Command and all fail-closed identity, runtime, allocation, semantic-canary, and provenance gates are otherwise identical to `BEE-L1B/PRE_REGISTRATION.md`.
