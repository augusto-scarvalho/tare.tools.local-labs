# Aborted after calls, before receipt

All 24 real requests and controls ran, but the final model-identity assertion compared the complete `/v1/models` payload. Its `data[].created` field is volatile, so the assertion failed even though systemd reported the same `MainPID=29428`, `NRestarts=0`, and stable model metadata. No samples or receipt were written because the harness is fail-closed.

After transition to `BLOCKED`, the shared runner was changed to compare a stable identity projection. A clean successor must rerun all samples under a new implementation digest.
