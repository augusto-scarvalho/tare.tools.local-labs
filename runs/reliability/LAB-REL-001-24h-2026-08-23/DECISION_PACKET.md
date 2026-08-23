# LAB-REL-001 fresh 24 h reliability soak

**SUPERSEDED / CANCELLED:** the user clarified on 2026-08-23 that soaks were excluded. The mistaken
launch was stopped after 8/8 partial operations; this packet no longer authorizes execution.

Frozen: 2026-08-23

## Authority and order

This section records the interpretation that caused the mistaken launch and is superseded by the
cancellation above. It does not append to or reinterpret the cancelled 2026-08-21 partial.

## Subject

- Canonical `llm-inference.service` on port 8080.
- Model/profile: Qwen3.8-27B UD-Q4_K_XL, 131,072 context, q4_0 K/V cache, native MTP n=3,
  one server slot, exact service configuration in `/etc/default/llm-inference`.
- Embedding service on port 8081 remains resident and is not mutated.
- SERVE/LAB lock must be coherent in SERVE mode before launch.

## Frozen run

- Harness: `tools/benchmarks/reliability_soak.py`.
- Duration: 24 hours from fresh start.
- Interval: 60 seconds.
- Long known-answer request every 10th operation.
- Other operations rotate all eight qualified LAB-AGENT-001-v2 cases.
- Append-only `records.jsonl`; atomic `summary.json`; stdout/stderr retained.

## Decision

- `PASS`: process reaches `COMPLETE`, operation failures = 0, health failures = 0.
- `FAIL`: any operation or health failure, even if later requests recover.
- `INTERRUPTED` or `CRASHED`: not a PASS; preserve evidence and diagnose.
- LAB-REL-002 remains blocked until this fresh 24 h predecessor completes cleanly.
