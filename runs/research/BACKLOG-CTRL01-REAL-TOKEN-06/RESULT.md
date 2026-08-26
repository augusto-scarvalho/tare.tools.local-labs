# BACKLOG-CTRL01-REAL-TOKEN-06 result

## Verdict

`CTRL01_FALSE_POSITIVE_CONFIRMED_R6` pending independent AGY review.

Independent recovery reproduced all source metrics exactly from 36 immutable physical rows. Raw real-model JSON validity was `1.000000`; applying the sidecar reduced complete validity to `0.750000`. Valid-token acceptance was `0.901639`, valid-control exact preservation was `0.833333`, p95 overhead was `34.850` microseconds/token, and production runtime binding was `False`.

Failed mandatory gates: `real_validity, valid_control_recall, valid_control_semantics, runtime_binding`. This recovers canonical evidence from the already completed physical run; it performs no new inference and makes no Python-mode claim.
