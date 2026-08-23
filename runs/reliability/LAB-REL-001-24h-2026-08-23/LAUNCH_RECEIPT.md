# LAB-REL-001 launch receipt

**Final disposition:** `CANCELLED_BY_USER / INCOMPLETE_NOT_PASS`. At `2026-08-23T03:52:00Z`, both
transient units were stopped, canonical 8080 was stopped, mode returned to LAB, and embedding 8081
remained healthy. The official partial ended at 8/8 operations with zero health failures.

## Canonical endpoint

- SERVE/LAB mode changed from LAB to SERVE with reason `authorized reliability soak sequence 2026-08-23`.
- `llm-inference.service` started through systemd as root; canonical PID at initial launch 437952.
- Port 8080 and auxiliary embedding port 8081 both returned `{"status":"ok"}`.

## Preflight evidence

- Windows-host preflight: `COMPLETE`, 3/3 operations, 0 health failures.
- WSL preflight: 0/1 because the canonical server aborted while loading an LCP prompt checkpoint:
  `common_prompt_checkpoint::load_tgt` -> `CUDA error: unknown error`. The service restarted under
  `Restart=always`. Receipt: `runs/reliability/LAB-REL-001-24h-2026-08-23-WSL-PREFLIGHT/` and journal
  timestamp `2026-08-23T03:40:45Z`.
- This event is not pooled into the official 24 h denominator, but is retained as a real preflight
  failure and strengthens the prior MTP/persistent-state concern.

## Official campaign

- Initial systemd launch attempts were preserved as `launch_attempt1_*` and `launch_attempt2_*`.
  They failed before iteration 0 because the clean systemd PATH omitted `nvidia-smi.exe`, then
  `powershell.exe`; these are launcher-environment failures, not endpoint observations.
- Historical unit (now stopped): `lab-rel-001-20260823-a3.service`.
- Invocation ID: `a6179a48cb324aa886a6cc02b3903149`.
- Harness PID at launch: 438296.
- Start: `2026-08-23T03:43:52.737590Z`.
- Frozen duration/interval: 86,400 s / 60 s.
- First two official operations: 2/2 pass, 0 operation failures, 0 health failures.

## Successor automation

- Historical unit (stopped before successors ran): `lab-rel-sequence-20260823.service`.
- Invocation ID: `646aa0078a324684867e227cbca207ee`.
- It waits for a clean full 24 h summary, then runs 48 h and 72 h sequentially; any failure or stale
  receipt blocks successors. Live state: `../LAB-REL-SEQUENCE-2026-08-23/sequence_summary.json`.
