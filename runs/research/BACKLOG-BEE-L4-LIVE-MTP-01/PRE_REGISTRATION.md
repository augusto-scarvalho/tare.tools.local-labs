# BACKLOG-BEE-L4-LIVE-MTP-01 preregistration

Task: Qualify BEE-L4 observable slot isolation on the live MTP runtime
Evidence class: `serving_runtime`

## Hypothesis

The active four-slot `draft-mtp` runtime will preserve deterministic per-slot state across 100 concurrent physical requests despite real rejected draft tokens: every repeated output must match its slot baseline, every response must contain only its own code word, all slots must return idle, and the service must not restart. Failure confirms that the historical simulation-only promotion was a false positive for the live runtime.

## Frozen inputs

- `runs/research/BEE-L4-TRANSACTIONAL-MTP-2026-08-25/PRE_REGISTRATION.md`
- `runs/research/BEE-L4-TRANSACTIONAL-MTP-2026-08-25/RESULT.md`
- `runs/research/BEE-L4-TRANSACTIONAL-MTP-2026-08-25/raw/receipt.json`
- `tools/analysis/transactional_mtp_manager.py`
- `tests/test_transactional_mtp_manager.py`

- Historical preregistration/result/receipt SHA-256: `cc6d18fd717c34411d1cd682097978315b4616596649c5584992bb0eeae76bad`, `65ce6535303a8f7d782553175360a94c19f63c51a2b07b16ee8c116fecbd476a`, `4b806de6b1bcd5b109d96f002e86cdb9aec5d8242b9cfbd226b6f552149a41fd`.
- Historical manager/test SHA-256: `4b7cb18323dfb70912f140965d4967739443c9dc7d30321f6d065e4ec6076d66`, `a873602bdc9efebf8a160bc8e0bdee5c4f66edffe5ca2fa6c17ab50abcde3cd7`.
- Active endpoint: `http://127.0.0.1:8080`; systemd unit and binary/argument identities are captured before execution and must remain unchanged.

## Command

```powershell
python tools/research/run_bee_l4_live_mtp.py --outdir runs/research/BACKLOG-BEE-L4-LIVE-MTP-01
```

## Factors

- Four fixed physical slots `0..3`, with code words `SAFFRON`, `COBALT`, `AMBER`, and `VIOLET` respectively.
- Twenty-five concurrent rounds; each round submits exactly one deterministic `/completion` request per slot (`temperature=0`, `top_k=1`, `seed=0`, `n_predict=32`, `cache_prompt=true`). Total: 100 live requests.
- Freeze the first-round output per slot as the paired baseline. All later outputs for that slot must match it byte-for-byte.
- Every output must contain its own code word case-insensitively and no code word assigned to another slot.
- Read `draft_n` and `draft_n_accepted` from each physical response. At least 80 requests must use draft tokens and at least 25 must reject one or more draft tokens, establishing observable rollback pressure.
- Record request/response bodies, timings, slot IDs, round ordering, wall latency, `/slots` before/after, systemd `MainPID/NRestarts/ExecStart`, binary SHA-256, service logs and 8080/8081 health.
- After all rounds, poll until all four slots are idle. The same PID, zero restarts and both health endpoints HTTP 200 are mandatory.

## Acceptance gates

- `request_coverage`: `live_requests eq 100`
- `slot_coverage`: `physical_slots eq 4`
- `speculation_coverage`: `requests_with_draft_tokens ge 80`
- `rollback_coverage`: `requests_with_rejected_draft_tokens ge 25`
- `state_consistency`: `exact_repeat_rate eq 1.0`
- `nonce_integrity`: `own_nonce_rate eq 1.0`
- `cross_slot_isolation`: `cross_slot_leakage_count eq 0`
- `service_integrity`: `service_restarts eq 0`
- `idle_recovery`: `idle_slots_after eq 4`

## Abort conditions

- Active route is not four-slot `draft-mtp`, slot endpoint unavailable, any request errors/times out, another workload occupies a slot, service identity changes, or 8081 health fails.
- Any threshold, prompt, code word, slot mapping, request count or decode setting changes after outputs are observed.

## Allowed claims

- `BEE_L4_LIVE_SLOT_ISOLATION_QUALIFIED_R1`
- `BEE_L4_FALSE_POSITIVE_CONFIRMED_R1`

Claims outside these codes are forbidden even if a metric looks favorable.

This test can qualify observable live slot isolation only; it cannot measure internal pointer atomicity or the historical Python manager's microsecond overhead.
