# LAB-REL dependency-gated soak sequence

**CANCELLED / NOT RUN FORWARD:** user clarified on 2026-08-23 that soak experiments were excluded.

Frozen: 2026-08-23

Order: fresh LAB-REL-001 24 h, then LAB-REL-002 48 h, then LAB-REL-002 72 h.

Each successor opens only when its predecessor has status `COMPLETE`, at least one operation, exactly
all operations passing, zero operation failures, zero health failures, and the full requested duration.
Any terminal failure or a `RUNNING` summary stale for 15 minutes blocks the remaining stages. Partial
observations remain evidence but are never relabeled PASS. The sequencer writes an atomic
`sequence_summary.json` and runs under its own transient systemd unit.
