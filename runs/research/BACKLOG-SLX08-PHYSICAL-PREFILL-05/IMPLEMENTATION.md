# BACKLOG-SLX08-PHYSICAL-PREFILL-05 implementation

R5 preserves the immutable R4 experimental server binary and changes only the
experiment driver and acceptance policy:

- native streaming measures monotonic time to the first non-empty content;
- eight predicted tokens allow the frozen single-digit answers to materialize;
- absolute 90% semantic floors prevent empty relative parity;
- restoration compares stable systemd argv, gateway role/model and HTTP health,
  while retaining volatile PID/time fields as evidence rather than identity.

Bound files:

- `tools/research/run_slx08_physical_prefill_r5.py`: `42a0c405bedd7003a6d28b17152261ea690bf1fb5cd6ca6f3e18f6451fd94848`
- `tests/test_slx08_physical_prefill_r5.py`: `9716f437e1eb65a48382461bbd2d517837b303eeacf65d7b6ac811946d0984d6`
- experimental binary: `4395a601202ec76bcaef1d10db97849a92b311d8c31e4afce4d8b961609807a1`

R4 remains immutable and ABORTED. R5 performs a fresh 128-response campaign.
